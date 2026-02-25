"""Low-level model execution and thread-based service wrapper.

ModelRunner loads a .gguf model via llama-cpp-python and handles inference
(standard generation, beam search, branch-at-step). RunnerService wraps it
in a background thread with a state machine and prompt queue.
"""
import ctypes
import io
import queue
import threading
import time
from contextlib import redirect_stdout as _redirect_stdout, redirect_stderr as _redirect_stderr
from enum import Enum
from typing import Callable

import llama_cpp
import numpy as np
from llama_cpp import Llama

from params import (
    ModelConfig, GenerationConfig, RunnerConfig,
    ParameterVisibility,
)


_runner_counter = 0
_runner_counter_lock = threading.Lock()


def _next_runner_index() -> int:
    global _runner_counter
    with _runner_counter_lock:
        idx = _runner_counter
        _runner_counter += 1
        return idx


_LLAMA_LOG_LEVEL_NAMES = {2: "error", 3: "warn", 4: "info"}

# Thread-keyed log routing: each RunnerService thread registers its ModelRunner here.
# The global callback dispatches to the runner on the calling thread, so logs from
# one agent never appear under another even when multiple agents are loaded.
_thread_log_map: dict[int, "ModelRunner"] = {}
_thread_log_lock = threading.Lock()


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
    if level > 3 and not runner.model_config.verbose:
        return
    level_name = _LLAMA_LOG_LEVEL_NAMES.get(level, str(level))
    runner._output_callback(
        f'\n<llama_cpp model="{runner.name}" index="{runner._runner_index}"'
        f' level="{level_name}">{msg}</llama_cpp>'
    )


llama_cpp.llama_log_set(_global_llama_log_cb, ctypes.c_void_p(0))


# ---- Model runner ----


class _StdoutRouter(io.StringIO):
    """Captures stdout and forwards each line to output_fn as a basic_log chunk."""
    def __init__(self, output_fn):
        super().__init__()
        self._output_fn = output_fn

    def write(self, s: str) -> int:
        if s and s != '\n':
            self._output_fn(f"<basic_log>{s}</basic_log>")
        return len(s)

    def flush(self):
        pass


def _log_softmax(scores: np.ndarray):
    """Numerically stable log-softmax over a score vector."""
    max_s = float(np.max(scores))
    exp_s = np.exp(scores - max_s)
    return scores - max_s - np.log(np.sum(exp_s))


def _parse_stop_tokens(llm, stop: list[str] | None) -> set[int]:
    """Convert stop strings to single-token IDs (multi-token stops are skipped)."""
    tokens = set()
    if stop:
        for s in stop:
            toks = llm.tokenize(s.encode(), add_bos=False)
            if len(toks) == 1:
                tokens.add(toks[0])
    return tokens


class ModelRunner:
    def __init__(self, model_config: ModelConfig, name: str = "model",
                 active_groups: list[str] | None = None,
                 output_callback: Callable[[str], None] | None = None):
        self.name = name
        self.model_config = model_config
        self.active_groups = active_groups or ["essential"]
        self.llm = None
        self.stats: dict = {}
        self._output_callback = output_callback or print
        self._runner_index = _next_runner_index()
        self._visibility = ParameterVisibility(
            active_groups=self.active_groups,
            output_callback=self._output_callback,
        )

    def update_active_groups(self, groups: list[str]):
        """Hot-swap visible parameter groups on a live runner."""
        self.active_groups = list(groups)
        self._visibility.active_groups = list(groups)

    def _log(self, msg: str):
        """Prefixed essential log line (always visible)."""
        self._output_callback(f"[{self.name}] {msg}\n")

    def _basic_log(self, msg: str):
        """Prefixed log line shown in Basic/Verbose modes only."""
        self._output_callback(f"<basic_log>[{self.name}] {msg}\n</basic_log>")

    def _emit(self, text: str):
        """Raw text for streaming tokens (always visible)."""
        self._output_callback(text)

    def _basic_emit(self, text: str):
        """Raw diagnostic text shown in Basic/Verbose modes only."""
        self._output_callback(f"<basic_log>{text}</basic_log>")

    def _require_logits_all(self, feature: str) -> bool:
        """Check logits_all and log an error if missing. Returns True if OK."""
        if self.model_config.logits_all:
            return True
        self._basic_log(f"ERROR: {feature} requires logits_all=True. "
                        "Enable it in the visibility group and reload the model.")
        self._basic_log("Falling back to standard generation.")
        return False

    def load(self):
        """Load model with visibility into the process."""
        with _thread_log_lock:
            _thread_log_map[threading.get_ident()] = self

        self._basic_log(f"Loading from {self.model_config.model_path}")
        self._visibility.log_loading(self.model_config, label=self.name)

        start = time.time()
        kwargs = self.model_config.to_llama_kwargs(self.active_groups)

        _router = _StdoutRouter(self._output_callback)
        if kwargs.get("verbose", False):
            # verbose=True: llama-cpp keeps the global callback installed.
            with _redirect_stdout(_router), _redirect_stderr(_router):
                self.llm = Llama(**kwargs)
        else:
            # verbose=False: Llama() calls llama_log_set(NULL) internally;
            # reinstall the global callback so errors/warnings still surface.
            with _redirect_stdout(_router), _redirect_stderr(_router):
                self.llm = Llama(**kwargs)
            llama_cpp.llama_log_set(_global_llama_log_cb, ctypes.c_void_p(0))

        load_time = time.time() - start
        self._log(f"Loaded in {load_time:.2f}s")

    def generate(self, prompt: str,
                 gen_config: GenerationConfig | None = None):
        """Generate with full visibility based on active parameter groups."""
        if gen_config is None:
            gen_config = GenerationConfig()

        self._visibility.log_generation(gen_config, label=self.name)

        kwargs = gen_config.to_generation_kwargs(self.active_groups)

        # stream/echo control code flow -- read directly, remove from kwargs
        do_stream = kwargs.pop("stream", False)
        do_echo = kwargs.pop("echo", False)

        # Beam search / branching params
        beam_width = kwargs.pop("beam_width", 1)
        beam_depth = kwargs.pop("beam_depth", 0)
        length_penalty = kwargs.pop("length_penalty", 1.0)
        beam_log_tree = kwargs.pop("beam_log_tree", False)
        beam_top_results = kwargs.pop("beam_top_results", 0)
        branch_at = kwargs.pop("branch_at", 0)
        branch_pick = kwargs.pop("branch_pick", 0)

        if beam_width > 1:
            max_tokens = kwargs.pop("max_tokens", 256)
            return self._generate_beam(
                prompt, beam_width, length_penalty,
                beam_log_tree, beam_top_results,
                max_tokens=beam_depth if beam_depth > 0 else max_tokens,
                stop=kwargs.pop("stop", None),
            )
        elif branch_at > 0:
            return self._generate_branch(
                prompt, branch_at, branch_pick,
                max_tokens=kwargs.pop("max_tokens", 256),
                stop=kwargs.pop("stop", None),
            )

        # Tokenize for stats
        tokens = self.llm.tokenize(prompt.encode())
        self._basic_log(f"Input tokens: {len(tokens)}")

        start = time.time()
        generated_tokens = 0
        full_response = ""

        if do_stream:
            for chunk in self.llm(prompt, stream=True, echo=do_echo, **kwargs):
                token_text = chunk["choices"][0]["text"]
                full_response += token_text
                generated_tokens += 1
                self._emit(token_text)
        else:
            result = self.llm(prompt, echo=do_echo, **kwargs)
            full_response = result["choices"][0]["text"]
            generated_tokens = result["usage"]["completion_tokens"]
            self._emit(full_response)

        if not full_response.endswith('\n'):
            self._emit('\n')

        elapsed = time.time() - start
        tps = generated_tokens / elapsed if elapsed > 0 else 0

        self._emit(f"\n{'-'*40}\n")
        self._log("STATS:")
        self._emit(f"  Output tokens: {generated_tokens}\n")
        self._emit(f"  Time: {elapsed:.2f}s\n")
        self._emit(f"  Speed: {tps:.1f} tok/s\n")

        self.stats = {
            "input_tokens": len(tokens),
            "output_tokens": generated_tokens,
            "time": elapsed,
            "tokens_per_second": tps,
        }

        return full_response

    def generate_with_logits(self, prompt: str, max_tokens: int = 50):
        """Generate with visibility into model's token probabilities."""
        self._basic_log("GENERATING WITH LOGIT VISIBILITY:")
        self._basic_emit("-" * 40 + "\n")

        self.llm.reset()
        tokens = self.llm.tokenize(prompt.encode())
        self.llm.eval(tokens)

        generated = []

        for i in range(max_tokens):
            scores = self.llm.scores[len(tokens) + i - 1]
            token = self.llm.sample(top_k=40, top_p=0.95, temp=0.7)
            token_text = self.llm.detokenize([token]).decode(errors="ignore")

            top_indices = np.argsort(scores)[-5:][::-1]
            top_tokens = [
                (self.llm.detokenize([idx]).decode(errors="ignore"), scores[idx])
                for idx in top_indices
            ]

            self._basic_emit(
                f"Token {i+1}: '{token_text}' | "
                f"Top 5: {[(t, f'{s:.2f}') for t, s in top_tokens]}\n"
            )

            if token == self.llm.token_eos():
                break

            generated.append(token)
            self.llm.eval([token])

        return self.llm.detokenize(generated).decode(errors="ignore")

    # ---- Beam search ----

    def _generate_beam(self, prompt: str, beam_width: int,
                       length_penalty: float, log_tree: bool,
                       top_results: int, *, max_tokens: int = 256,
                       stop: list[str] | None = None) -> str:
        """Run beam search over the model, returning the best hypothesis."""
        if not self._require_logits_all("beam search"):
            return self.generate(prompt)

        self._basic_log(f"Beam search: width={beam_width}, max_tokens={max_tokens}, "
                        f"length_penalty={length_penalty}")

        self.llm.reset()
        prompt_tokens = self.llm.tokenize(prompt.encode())
        n_prompt = len(prompt_tokens)
        self._basic_log(f"Input tokens: {n_prompt}")
        self.llm.eval(prompt_tokens)

        eos = self.llm.token_eos()
        stop_tokens = _parse_stop_tokens(self.llm, stop)

        # Each beam: (token_ids, cumulative_log_prob, state)
        initial_state = self.llm.save_state()
        beams = [{"tokens": [], "cum_logp": 0.0, "state": initial_state}]
        completed = []

        for step in range(max_tokens):
            candidates = []

            for bi, beam in enumerate(beams):
                self.llm.load_state(beam["state"])
                logit_idx = n_prompt + len(beam["tokens"]) - 1
                scores = self.llm.scores[logit_idx]
                log_probs = _log_softmax(scores)

                top_k = min(beam_width * 2, len(log_probs))
                top_indices = np.argpartition(log_probs, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(log_probs[top_indices])][::-1]

                if log_tree:
                    parts = []
                    for idx in top_indices:
                        tok_str = self.llm.detokenize([int(idx)]).decode(errors="ignore")
                        parts.append(f'"{tok_str}"({log_probs[idx]:.2f})')
                    self._basic_emit(f"[tree] Step {step + 1}, Beam {bi + 1} -> "
                                     f"{' '.join(parts)}\n")

                for idx in top_indices:
                    idx_int = int(idx)
                    new_tokens = beam["tokens"] + [idx_int]
                    new_logp = beam["cum_logp"] + float(log_probs[idx])
                    seq_len = len(new_tokens)
                    norm_score = new_logp / (seq_len ** length_penalty)
                    candidates.append({
                        "tokens": new_tokens,
                        "cum_logp": new_logp,
                        "norm_score": norm_score,
                        "parent_state": beam["state"],
                        "new_token": idx_int,
                        "is_eos": idx_int == eos or idx_int in stop_tokens,
                        "beam_idx": bi,
                    })

            # Separate EOS candidates
            for c in candidates:
                if c["is_eos"]:
                    completed.append({
                        "tokens": c["tokens"][:-1] if c["is_eos"] else c["tokens"],
                        "cum_logp": c["cum_logp"],
                        "norm_score": c["norm_score"],
                        "truncated": False,
                    })

            # Keep only non-EOS, prune to top beam_width
            active_candidates = [c for c in candidates if not c["is_eos"]]
            active_candidates.sort(key=lambda c: c["norm_score"], reverse=True)
            active_candidates = active_candidates[:beam_width]

            if log_tree:
                for c in active_candidates:
                    tok_str = self.llm.detokenize([c["new_token"]]).decode(errors="ignore")
                    self._basic_emit(f"  [kept] \"{tok_str}\" "
                                     f"(cum: {c['cum_logp']:.2f}, "
                                     f"norm: {c['norm_score']:.2f})\n")

            if not active_candidates:
                break

            # Early stop: all completed beams outscore all active beams
            if completed:
                best_completed = max(c["norm_score"] for c in completed)
                best_active = active_candidates[0]["norm_score"]
                if best_completed >= best_active:
                    break

            # Expand survivors: load parent state, eval new token, save new state
            new_beams = []
            for c in active_candidates:
                self.llm.load_state(c["parent_state"])
                self.llm.eval([c["new_token"]])
                new_state = self.llm.save_state()
                new_beams.append({
                    "tokens": c["tokens"],
                    "cum_logp": c["cum_logp"],
                    "state": new_state,
                })
            beams = new_beams

            if (step + 1) % 5 == 0:
                best_text = self.llm.detokenize(beams[0]["tokens"]).decode(errors="ignore")
                preview = best_text[:60] + ("..." if len(best_text) > 60 else "")
                self._basic_emit(f"[beam] Step {step + 1}/{max_tokens} | "
                                 f"active: {len(beams)}, done: {len(completed)} | "
                                 f"best: \"{preview}\"\n")

        # Add truncated beams (active beams that didn't reach EOS)
        for beam in beams:
            if beam["tokens"]:
                seq_len = len(beam["tokens"])
                norm = beam["cum_logp"] / (seq_len ** length_penalty)
                completed.append({
                    "tokens": beam["tokens"],
                    "cum_logp": beam["cum_logp"],
                    "norm_score": norm,
                    "truncated": True,
                })

        completed.sort(key=lambda c: c["norm_score"], reverse=True)
        n_show = top_results if top_results > 0 else beam_width
        show = completed[:n_show]

        self._emit(f"\n{'=' * 40}\n")
        self._basic_log(f"BEAM SEARCH RESULTS ({len(show)} beams):")

        best_text = ""
        for i, beam in enumerate(show):
            text = self.llm.detokenize(beam["tokens"]).decode(errors="ignore")
            tag = " [TRUNCATED]" if beam["truncated"] else ""
            self._emit(f"  [Beam {i + 1}] score: {beam['norm_score']:.4f}{tag}\n")
            self._emit(f"  {text}\n\n")
            if i == 0:
                best_text = text

        return best_text

    # ---- Branch at step ----

    def _generate_branch(self, prompt: str, branch_at: int, branch_pick: int,
                         *, max_tokens: int = 256,
                         stop: list[str] | None = None) -> str:
        """Re-run generation, forcing an alternate token at a specific step."""
        if not self._require_logits_all("branching"):
            return self.generate(prompt)

        self._basic_log(f"Branching at step {branch_at}, picking alternative #{branch_pick}")

        self.llm.reset()
        prompt_tokens = self.llm.tokenize(prompt.encode())
        n_prompt = len(prompt_tokens)
        self._basic_log(f"Input tokens: {n_prompt}")
        self.llm.eval(prompt_tokens)

        eos = self.llm.token_eos()
        stop_tokens = _parse_stop_tokens(self.llm, stop)

        generated = []

        for step in range(max_tokens):
            logit_idx = n_prompt + step - 1 if step > 0 else n_prompt - 1
            scores = self.llm.scores[logit_idx]
            log_probs = _log_softmax(scores)

            current_step = step + 1  # 1-indexed for user display

            if current_step == branch_at:
                top5 = np.argsort(log_probs)[-5:][::-1]
                self._basic_emit(f"[branch] Step {current_step}: top candidates:\n")
                for rank, idx in enumerate(top5):
                    tok_str = self.llm.detokenize([int(idx)]).decode(errors="ignore")
                    marker = ""
                    if rank == 0:
                        marker = "  <- would have been chosen"
                    elif rank == branch_pick + 1:
                        marker = "  <- PICKED"
                    self._basic_emit(f"  #{rank} \"{tok_str}\" "
                                     f"(log_prob: {log_probs[idx]:.2f}){marker}\n")

                # Pick the alternate token (skip rank 0 = greedy best)
                pick_rank = branch_pick + 1
                if pick_rank >= len(top5):
                    pick_rank = len(top5) - 1
                    self._basic_log(f"branch_pick {branch_pick} out of range, "
                                   f"using rank {pick_rank}")
                token = int(top5[pick_rank])
            else:
                token = self.llm.sample(top_k=40, top_p=0.95, temp=0.7)

            if token == eos or token in stop_tokens:
                break

            generated.append(token)
            token_text = self.llm.detokenize([token]).decode(errors="ignore")
            self._emit(token_text)
            self.llm.eval([token])

        full_text = self.llm.detokenize(generated).decode(errors="ignore")
        self._emit(f"\n{'-' * 40}\n")
        self._basic_log(f"Branch result ({len(generated)} tokens):")
        self._emit(f"  {full_text}\n")
        return full_text

    def unload(self):
        """Free memory."""
        if self.llm:
            del self.llm
            self.llm = None
            self._basic_log("Unloaded")
        with _thread_log_lock:
            _thread_log_map.pop(threading.get_ident(), None)


# ---- Service layer ----

class ServiceState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    GENERATING = "generating"
    STOPPING = "stopping"
    ERROR = "error"


_SHUTDOWN_SENTINEL = object()


class RunnerService:
    def __init__(self, config: RunnerConfig,
                 output_callback: Callable[[str], None] | None = None):
        self._config = config
        self._output_callback = output_callback or print
        self._queue: queue.Queue = queue.Queue()
        self._state = ServiceState.IDLE
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._runner: ModelRunner | None = None

    @property
    def state(self) -> ServiceState:
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, value: ServiceState):
        with self._state_lock:
            self._state = value

    @property
    def config(self) -> RunnerConfig:
        return self._config

    @config.setter
    def config(self, value: RunnerConfig):
        self._config = value

    def update_active_groups(self, groups: list[str]):
        """Propagate group changes to the live runner (if loaded)."""
        self._config.active_groups = list(groups)
        if self._runner is not None:
            self._runner.update_active_groups(groups)

    def start(self):
        if self.state not in (ServiceState.IDLE, ServiceState.ERROR):
            self._output_callback(
                f"<basic_log>[service] Cannot start: state is {self.state.value}\n</basic_log>"
            )
            return
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self.state in (ServiceState.IDLE, ServiceState.STOPPING):
            return
        self.state = ServiceState.STOPPING
        self._queue.put(_SHUTDOWN_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None

    def submit_prompt(self, prompt: str,
                      gen_config: GenerationConfig | None = None):
        if self.state != ServiceState.READY:
            self._output_callback(
                f"<basic_log>[service] Cannot submit: state is {self.state.value}\n</basic_log>"
            )
            return
        self._queue.put({
            "prompt": prompt,
            "gen_config": gen_config or self._config.generation_config,
        })

    def _run_loop(self):
        try:
            self.state = ServiceState.LOADING
            self._runner = ModelRunner(
                model_config=self._config.model_config,
                name=self._config.model_name,
                active_groups=self._config.active_groups,
                output_callback=self._output_callback,
            )
            self._runner.load()
            self.state = ServiceState.READY

            while True:
                item = self._queue.get()
                if item is _SHUTDOWN_SENTINEL:
                    break
                try:
                    self.state = ServiceState.GENERATING
                    self._runner.generate(
                        prompt=item["prompt"],
                        gen_config=item["gen_config"],
                    )
                except Exception as e:
                    self._output_callback(
                        f"\n[service] Generation error: {e}\n"
                    )
                finally:
                    self.state = ServiceState.READY

        except Exception as e:
            self._output_callback(f"\n[service] Fatal error: {e}\n")
            self.state = ServiceState.ERROR
            return
        finally:
            if self._runner is not None:
                self._runner.unload()
                self._runner = None

        self.state = ServiceState.IDLE
