"""Low-level model execution and thread-based service wrapper.

ModelRunner loads a .gguf model via llama-cpp-python and handles inference
(standard generation, beam search, branch-at-step). RunnerService wraps it
in a background thread with a state machine and prompt queue.
"""
import os
import queue
import threading
import time
from contextlib import redirect_stdout as _redirect_stdout, redirect_stderr as _redirect_stderr
from enum import Enum
from typing import Callable

import numpy as np
from llama_cpp import Llama

from params import (
    ModelConfig, GenerationConfig, RunnerConfig,
    ParameterVisibility,
)
from chat_format import (
    apply_template, apply_multiturn_template, trim_history,
    get_stop_tokens, load_template,
)
from llama_logs import (
    FdStderrCapture, StdoutRouter, LOG_CATEGORY_FIELDS,
    register_thread_runner, unregister_thread_runner,
    reinstall_log_callback,
)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(_SRC_DIR)), "templates")


_runner_counter = 0
_runner_counter_lock = threading.Lock()


def _next_runner_index() -> int:
    global _runner_counter
    with _runner_counter_lock:
        idx = _runner_counter
        _runner_counter += 1
        return idx


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
        self._has_vision = False
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

    def _emit_stats(self, generated_tokens: int, elapsed: float,
                    show_stats: str) -> None:
        """Emit generation stats based on the show_stats setting."""
        tps = generated_tokens / elapsed if elapsed > 0 else 0
        if show_stats == "always":
            self._emit(f"{'-'*40}\n")
            self._log("STATS:")
            self._emit(f"  Output tokens: {generated_tokens}\n")
            self._emit(f"  Time: {elapsed:.2f}s\n")
            self._emit(f"  Speed: {tps:.1f} tok/s\n")
        elif show_stats == "basic":
            self._basic_emit(f"{'-'*40}\n")
            self._basic_log("STATS:")
            self._basic_emit(f"  Output tokens: {generated_tokens}\n")
            self._basic_emit(f"  Time: {elapsed:.2f}s\n")
            self._basic_emit(f"  Speed: {tps:.1f} tok/s\n")

    def _emit_logprobs(self, logprobs: dict | None) -> None:
        """Render top-N token distribution from a choice['logprobs'] payload
        into the basic log sink. No-op if logprobs is missing or empty."""
        if not logprobs:
            return
        import math
        tokens = logprobs.get("tokens") or []
        token_logprobs = logprobs.get("token_logprobs") or []
        top_list = logprobs.get("top_logprobs") or []
        for i, chosen in enumerate(tokens):
            top = top_list[i] if i < len(top_list) else None
            chosen_lp = token_logprobs[i] if i < len(token_logprobs) else None
            if chosen_lp is None:
                # prompt-echo tokens have no sampled logprob; skip
                continue
            self._basic_log(f"step: chose {chosen!r} (logprob {chosen_lp:.2f})")
            if not top:
                continue
            for tok, lp_val in sorted(top.items(), key=lambda x: -x[1]):
                prob = math.exp(lp_val) * 100
                marker = "*" if tok == chosen else " "
                self._basic_emit(f"  {marker} {tok!r:<20} {lp_val:>7.2f}  {prob:>5.1f}%\n")

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
        register_thread_runner(self)

        self._basic_log(f"Loading from {self.model_config.model_path}")
        self._visibility.log_loading(self.model_config, label=self.name)

        start = time.time()
        kwargs = self.model_config.to_llama_kwargs(self.active_groups)

        # Log category toggles are consumed by the log callback / fd capture;
        # they are not arguments to Llama().
        for _k in LOG_CATEGORY_FIELDS:
            kwargs.pop(_k, None)

        # --- Multimodal vision projector ---
        mmproj_path = kwargs.pop("chat_handler_path", None)
        self._has_vision = False
        if mmproj_path and os.path.isfile(mmproj_path):
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            self._basic_log(f"Loading vision projector: {mmproj_path}")
            handler = Llava15ChatHandler(
                clip_model_path=mmproj_path,
                verbose=kwargs.get("verbose", False),
            )
            kwargs["chat_handler"] = handler
            self._has_vision = True
        elif mmproj_path:
            self._log(f"WARNING: mmproj not found: {mmproj_path}")

        _router = StdoutRouter(self._output_callback)
        with FdStderrCapture(self), _redirect_stdout(_router), _redirect_stderr(_router):
            self.llm = Llama(**kwargs)
        # Llama() calls llama_log_set(NULL) when verbose=False; reinstall our
        # global callback so errors/warnings and category-filtered logs keep flowing.
        if not kwargs.get("verbose", False):
            reinstall_log_callback()

        load_time = time.time() - start
        self._emit('\n')
        self._log(f"Loaded in {load_time:.2f}s")

    def generate(self, prompt: str,
                 gen_config: GenerationConfig | None = None,
                 history: list[dict] | None = None,
                 images: list[str] | None = None):
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

        # Stats display
        show_stats = kwargs.pop("show_stats", "always")

        # Chat template params -- not passed to llama-cpp
        chat_template_name = kwargs.pop("chat_template", "none")
        system_prompt = kwargs.pop("system_prompt", "")

        # Multi-turn history params -- not passed to llama-cpp
        use_history = kwargs.pop("use_history", False)
        max_history_turns = kwargs.pop("max_history_turns", 10)

        # Session params -- used at load time in main_window; pop to keep out of llama kwargs
        kwargs.pop("session_mode", None)
        kwargs.pop("session_file", None)

        if chat_template_name != "none":
            template = load_template(_TEMPLATES_DIR, chat_template_name)
            if use_history and history:
                trimmed = trim_history(history, max_history_turns)
                prompt = apply_multiturn_template(template, system_prompt, trimmed, prompt)
                self._basic_log(f"Chat template (multi-turn, {len(trimmed)} turns): {chat_template_name}")
            else:
                prompt = apply_template(template, system_prompt, prompt)
                self._basic_log(f"Chat template: {chat_template_name}")
            if not kwargs.get("stop"):
                auto_stop = get_stop_tokens(template)
                if auto_stop:
                    kwargs["stop"] = auto_stop
                    self._basic_log(f"Auto-stop tokens: {auto_stop}")

        # Context window warning for multi-turn prompts
        if use_history and history and self.llm is not None:
            n_ctx = self.model_config.n_ctx
            prompt_tokens = self.llm.tokenize(prompt.encode())
            usage = len(prompt_tokens) / n_ctx if n_ctx > 0 else 0
            if usage > 0.9:
                self._log(f"WARNING: prompt uses {usage:.0%} of context window "
                          f"({len(prompt_tokens)}/{n_ctx} tokens). "
                          f"Consider reducing max_history_turns.")

        # Filter valid images
        valid_images = []
        if images and self._has_vision:
            for img_path in images:
                if os.path.isfile(img_path):
                    valid_images.append(img_path)
                else:
                    self._basic_log(f"WARNING: image not found, skipping: {img_path}")

        if beam_width > 1:
            if valid_images:
                self._log("WARNING: beam search does not support images, ignoring attachments.")
            max_tokens = kwargs.pop("max_tokens", 256)
            return self._generate_beam(
                prompt, beam_width, length_penalty,
                beam_log_tree, beam_top_results,
                max_tokens=beam_depth if beam_depth > 0 else max_tokens,
                stop=kwargs.pop("stop", None),
            )
        elif branch_at > 0:
            if valid_images:
                self._log("WARNING: branch-at-step does not support images, ignoring attachments.")
            return self._generate_branch(
                prompt, branch_at, branch_pick,
                max_tokens=kwargs.pop("max_tokens", 256),
                stop=kwargs.pop("stop", None),
            )

        # --- Vision path: use create_chat_completion with image content ---
        if valid_images:
            return self._generate_vision(
                prompt, valid_images, kwargs,
                system_prompt=system_prompt,
                history=history if use_history else None,
                max_history_turns=max_history_turns,
                do_stream=do_stream, show_stats=show_stats,
            )

        # Tokenize for stats
        tokens = self.llm.tokenize(prompt.encode())
        self._basic_log(f"Input tokens: {len(tokens)}")

        start = time.time()
        generated_tokens = 0
        full_response = ""

        _router = StdoutRouter(self._output_callback)
        if do_stream:
            with FdStderrCapture(self), _redirect_stdout(_router), _redirect_stderr(_router):
                for chunk in self.llm(prompt, stream=True, echo=do_echo, **kwargs):
                    choice = chunk["choices"][0]
                    token_text = choice["text"]
                    full_response += token_text
                    generated_tokens += 1
                    self._emit(token_text)
                    self._emit_logprobs(choice.get("logprobs"))
        else:
            with FdStderrCapture(self), _redirect_stdout(_router), _redirect_stderr(_router):
                result = self.llm(prompt, echo=do_echo, **kwargs)
            full_response = result["choices"][0]["text"]
            generated_tokens = result["usage"]["completion_tokens"]
            self._emit(full_response)
            self._emit_logprobs(result["choices"][0].get("logprobs"))

        if not full_response.endswith('\n'):
            self._emit('\n')

        elapsed = time.time() - start
        tps = generated_tokens / elapsed if elapsed > 0 else 0

        self._emit_stats(generated_tokens, elapsed, show_stats)

        self.stats = {
            "input_tokens": len(tokens),
            "output_tokens": generated_tokens,
            "time": elapsed,
            "tokens_per_second": tps,
        }

        return full_response

    # ---- Vision / multimodal generation ----

    _MIME_MAP = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    }

    def _generate_vision(self, prompt: str, image_paths: list[str],
                         gen_kwargs: dict, *, system_prompt: str = "",
                         history: list[dict] | None = None,
                         max_history_turns: int = 10,
                         do_stream: bool = False,
                         show_stats: str = "always") -> str:
        """Generate using create_chat_completion with image content blocks."""
        import base64
        from chat_format import _extract_text

        self._basic_log(f"Vision generation: {len(image_paths)} image(s)")

        # Build current user content with images
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img_path in image_paths:
            ext = os.path.splitext(img_path)[1].lower()
            mime = self._MIME_MAP.get(ext, "image/png")
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

        # Build messages list
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            trimmed = trim_history(history, max_history_turns)
            for turn in trimmed:
                turn_content = turn["content"]
                if isinstance(turn_content, list):
                    # Re-encode images from path references in history
                    rebuilt: list[dict] = []
                    for block in turn_content:
                        if block.get("type") == "image_path":
                            path = block["path"]
                            if not os.path.isfile(path):
                                self._basic_log(f"WARNING: history image missing: {path}")
                                continue
                            ext = os.path.splitext(path)[1].lower()
                            mime = self._MIME_MAP.get(ext, "image/png")
                            with open(path, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode()
                            rebuilt.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            })
                        else:
                            rebuilt.append(block)
                    messages.append({"role": turn["role"], "content": rebuilt})
                else:
                    messages.append({"role": turn["role"], "content": turn_content})
        messages.append({"role": "user", "content": content})

        start = time.time()
        generated_tokens = 0
        full_response = ""

        _router = StdoutRouter(self._output_callback)
        if do_stream:
            with FdStderrCapture(self), _redirect_stdout(_router), _redirect_stderr(_router):
                for chunk in self.llm.create_chat_completion(
                    messages=messages, stream=True, **gen_kwargs,
                ):
                    delta = chunk["choices"][0].get("delta", {})
                    token_text = delta.get("content", "")
                    if token_text:
                        full_response += token_text
                        generated_tokens += 1
                        self._emit(token_text)
        else:
            with FdStderrCapture(self), _redirect_stdout(_router), _redirect_stderr(_router):
                result = self.llm.create_chat_completion(
                    messages=messages, **gen_kwargs,
                )
            full_response = result["choices"][0]["message"]["content"] or ""
            generated_tokens = result.get("usage", {}).get("completion_tokens", 0)
            self._emit(full_response)

        if not full_response.endswith('\n'):
            self._emit('\n')

        elapsed = time.time() - start
        tps = generated_tokens / elapsed if elapsed > 0 else 0

        self._emit_stats(generated_tokens, elapsed, show_stats)

        self.stats = {
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
        unregister_thread_runner()


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
                 output_callback: Callable[[str], None] | None = None,
                 completion_callback: Callable[[str], None] | None = None):
        self._config = config
        self._output_callback = output_callback or print
        self._completion_callback = completion_callback
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
                      gen_config: GenerationConfig | None = None,
                      history: list[dict] | None = None,
                      images: list[str] | None = None):
        if self.state != ServiceState.READY:
            self._output_callback(
                f"<basic_log>[service] Cannot submit: state is {self.state.value}\n</basic_log>"
            )
            return
        self._queue.put({
            "prompt": prompt,
            "gen_config": gen_config or self._config.generation_config,
            "history": history or [],
            "images": images or [],
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
                    response = self._runner.generate(
                        prompt=item["prompt"],
                        gen_config=item["gen_config"],
                        history=item.get("history", []),
                        images=item.get("images", []),
                    )
                    if self._completion_callback and response is not None:
                        self._completion_callback(response)
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
