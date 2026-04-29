"""llama.cpp log capture, classification, and routing.

Two sources of llama.cpp-ish output are handled here:

* The official log callback (`_global_llama_log_cb`) — model loader, metadata,
  perf stats, decode, sampler, backend init. Installed process-wide at import.
* Stderr fd-level writes from mtmd.dll (image tokenize/encode/decode) that
  bypass the callback — captured by `FdStderrCapture`.

Runner threads register themselves via `register_thread_runner()` so the
global callback can dispatch lines to the right per-agent bucket.
"""
import ctypes
import io
import os
import sys
import threading

import llama_cpp


LLAMA_LOG_LEVEL_NAMES = {2: "error", 3: "warn", 4: "info"}


# Thread-keyed log routing: each RunnerService thread registers its ModelRunner
# here. The global callback dispatches to the runner on the calling thread, so
# logs from one agent never appear under another when multiple are loaded.
_thread_log_map: dict[int, object] = {}
_thread_log_lock = threading.Lock()


def register_thread_runner(runner) -> None:
    with _thread_log_lock:
        _thread_log_map[threading.get_ident()] = runner


def unregister_thread_runner() -> None:
    with _thread_log_lock:
        _thread_log_map.pop(threading.get_ident(), None)


# ---- Pattern tables ----
#
# Each tuple is (category, keyword-tuple, target_bucket).
# target_bucket: 'basic' → <basic_log>; 'verbose' → <llama_cpp> wrapped.
# The category name is the matching ModelConfig.log_<name> field.

_LLAMA_PATTERNS = (
    ("log_backend",    ("ggml_", "cuda_", "vulkan_", "clblast",
                        "rocm_", "hip_"), "basic"),
    ("log_model_load", ("llama_model_loader:", "llm_load_", "print_info:",
                        "llama_new_context", "llama_kv_cache_init",
                        "llama_context:"), "verbose"),
    ("log_perf_stats", ("llama_perf_",), "basic"),
    ("log_decode",     ("llama_decode:", "llama_sampler_", "decode:",
                        "eval:", "llama_kv_cache_update"), "verbose"),
)

_MTMD_PATTERNS = (
    ("log_image_progress", ("encoding image", "image slice encoded",
                            "decoding image", "image decoded"), "basic"),
    ("log_prompt_debug",   ("add_text:",), "verbose"),
    ("log_tensor_debug",   ("image_tokens->", "batch_f32 size"), "verbose"),
)


# Fields on ModelConfig that are log filters — popped from Llama() kwargs.
LOG_CATEGORY_FIELDS = frozenset({
    "log_image_progress", "log_prompt_debug", "log_tensor_debug",
    "log_backend", "log_model_load", "log_perf_stats", "log_decode",
})


def _classify(line: str, patterns: tuple) -> tuple[str, str] | None:
    """Return (category, target_bucket) for the first matching pattern, or None."""
    for category, keywords, bucket in patterns:
        for kw in keywords:
            if kw in line:
                return category, bucket
    return None


def _emit_basic(runner, line: str) -> None:
    runner._output_callback(f"<basic_log>[{runner.name}] {line}\n</basic_log>")


def _emit_verbose(runner, line: str, level_name: str = "info") -> None:
    runner._output_callback(
        f'\n<llama_cpp model="{runner.name}" index="{runner._runner_index}"'
        f' level="{level_name}">{line}</llama_cpp>'
    )


# ---- Global llama.cpp log callback ----

@llama_cpp.llama_log_callback
def _global_llama_log_cb(level, message, user_data):
    """Single global llama.cpp log callback; dispatches by calling thread."""
    tid = threading.get_ident()
    with _thread_log_lock:
        runner = _thread_log_map.get(tid)
    if runner is None:
        return
    try:
        msg = message.decode("utf-8", errors="replace").strip()
    except Exception:
        return
    if not msg:
        return

    level_name = LLAMA_LOG_LEVEL_NAMES.get(level, str(level))

    # Errors/warnings always pass through as verbose (UI still shows errors in
    # non-verbose modes via the existing rendering rules).
    if level <= 3:
        _emit_verbose(runner, msg, level_name)
        return

    cfg = runner.model_config

    # Master override: verbose=True bypasses category toggles, shows everything.
    if cfg.verbose:
        _emit_verbose(runner, msg, level_name)
        return

    # Non-verbose: only show categories the user has toggled on.
    classified = _classify(msg, _LLAMA_PATTERNS)
    if classified is None:
        return
    category, bucket = classified
    if not getattr(cfg, category, False):
        return
    if bucket == "basic":
        _emit_basic(runner, msg)
    else:
        _emit_verbose(runner, msg, level_name)


def reinstall_log_callback() -> None:
    """Re-install the global callback after Llama() resets it (verbose=False case)."""
    llama_cpp.llama_log_set(_global_llama_log_cb, ctypes.c_void_p(0))


# Install at import time so any subsequent llama_cpp use is covered.
reinstall_log_callback()


# ---- fd-level stderr capture for mtmd.dll printfs ----

class FdStderrCapture:
    """Redirect fd 2 (C-level stderr) into a pipe and route lines via the log rules.

    llama-cpp-python's mtmd image-tokenize path writes unconditional fprintf(stderr)
    messages that bypass both Python's sys.stderr and the llama log callback.
    We dup fd 2 to a pipe for the duration of load/generate, and a daemon thread
    pumps lines through _classify / _MTMD_PATTERNS.

    fd redirection is process-global; safe here because the runner is the only
    stderr writer during the wrapped call (GIL + single-threaded inference path).
    """

    def __init__(self, runner):
        self._runner = runner
        self._saved_fd: int | None = None
        self._read_fd:  int | None = None
        self._write_fd: int | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._read_fd, self._write_fd = os.pipe()
        self._saved_fd = os.dup(2)
        os.dup2(self._write_fd, 2)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        try:
            sys.stderr.flush()
        except Exception:
            pass
        # Restore stderr so further writes go back to the real terminal
        if self._saved_fd is not None:
            os.dup2(self._saved_fd, 2)
        # Close our pipe-write ref to signal EOF to the pump
        if self._write_fd is not None:
            os.close(self._write_fd)
            self._write_fd = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._read_fd is not None:
            os.close(self._read_fd)
            self._read_fd = None
        if self._saved_fd is not None:
            os.close(self._saved_fd)
            self._saved_fd = None

    def _pump(self):
        buf = b""
        while True:
            try:
                chunk = os.read(self._read_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._emit(line.decode("utf-8", errors="replace"))
        if buf:
            self._emit(buf.decode("utf-8", errors="replace"))

    def _emit(self, line: str):
        line = line.strip()
        if not line:
            return
        cfg = self._runner.model_config
        if cfg.verbose:
            _emit_verbose(self._runner, line, "mtmd")
            return
        classified = _classify(line, _MTMD_PATTERNS)
        if classified is None:
            return  # unknown fd noise (e.g. Qt debug) — drop when not verbose
        category, bucket = classified
        if not getattr(cfg, category, False):
            return
        if bucket == "basic":
            _emit_basic(self._runner, line)
        else:
            _emit_verbose(self._runner, line, "mtmd")


# ---- Stdout routing ----

class StdoutRouter(io.StringIO):
    """Captures stdout and forwards each line to output_fn as a basic_log chunk."""

    def __init__(self, output_fn):
        super().__init__()
        self._output_fn = output_fn

    def write(self, s: str) -> int:
        if s and s != '\n':
            self._output_fn(f"<basic_log>{s}\n</basic_log>")
        return len(s)

    def flush(self):
        pass
