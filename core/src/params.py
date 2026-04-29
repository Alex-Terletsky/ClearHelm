"""Configuration data structures for the ClearHelm runner.

Defines PARAMETER_GROUPS (visibility groups), ModelConfig and GenerationConfig
(llama-cpp-python settings), RunnerConfig (serializable composite), and
ParameterVisibility (active-parameter logging).
"""

import json
from dataclasses import dataclass, field, fields
from typing import Callable, NamedTuple


# ---- Module parameter descriptor ----

class ModuleParam(NamedTuple):
    """Describes a single configurable parameter exposed by a module."""
    name: str
    type: type
    default: object
    description: str = ""


class InterceptableCommand(NamedTuple):
    """Describes a command a module can gate for user approval."""
    name: str
    description: str = ""
    intercept: bool = False  # module author sets the safe default


# ---- Parameter group definitions ----

PARAMETER_GROUPS = {
    "essential": {
        "description": "Core parameters always shown",
        "loading": ["model_path", "n_ctx", "n_gpu_layers"],
        "generation": ["max_tokens", "temperature", "stop"],
    },
    "performance": {
        "description": "Hardware and speed tuning",
        "loading": [
            "n_threads", "n_threads_batch", "n_batch", "n_ubatch",
            "use_mmap", "use_mlock", "flash_attn", "offload_kqv",
        ],
        "generation": [],
    },
    "sampling_basic": {
        "description": "Common sampling parameters",
        "loading": [],
        "generation": ["top_p", "top_k", "repeat_penalty"],
    },
    "sampling_advanced": {
        "description": "Advanced sampling methods",
        "loading": [],
        "generation": [
            "min_p", "typical_p", "tfs_z", "frequency_penalty",
            "presence_penalty", "mirostat_mode", "mirostat_tau",
            "mirostat_eta",
        ],
    },
    "constraints": {
        "description": "Output constraints and biasing",
        "loading": [],
        "generation": ["grammar", "logit_bias"],
    },
    "context_extension": {
        "description": "Extended context / RoPE parameters",
        "loading": [
            "rope_freq_base", "rope_freq_scale", "rope_scaling_type",
            "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_fast",
            "yarn_beta_slow", "yarn_orig_ctx",
        ],
        "generation": [],
    },
    "adapters": {
        "description": "LoRA and fine-tune adapters",
        "loading": ["lora_path", "lora_base", "lora_scale"],
        "generation": [],
    },
    "visibility": {
        "description": "Debug and transparency options",
        "loading": ["verbose", "logits_all", "seed",
                    "log_image_progress", "log_prompt_debug", "log_tensor_debug",
                    "log_backend", "log_model_load", "log_perf_stats", "log_decode"],
        "generation": ["logprobs", "stream", "echo", "show_stats"],
    },
    "multi_gpu": {
        "description": "Multi-GPU configuration",
        "loading": ["tensor_split", "main_gpu", "split_mode"],
        "generation": [],
    },
    "speculative": {
        "description": "Speculative decoding",
        "loading": ["draft_model"],
        "generation": [],
    },
    "beam_search": {
        "description": "Beam search and branching",
        "loading": [],
        "generation": [
            "beam_width", "beam_depth", "length_penalty",
            "beam_log_tree", "beam_top_results",
            "branch_at", "branch_pick",
        ],
    },
    "chat": {
        "description": "Chat template and system prompt",
        "loading": [],
        "generation": ["chat_template", "system_prompt", "use_history", "max_history_turns"],
    },
    "session": {
        "description": "Chat session loading behavior",
        "loading": [],
        "generation": ["session_mode", "session_file"],
    },
    "multimodal": {
        "description": "Vision / multimodal projector",
        "loading": ["chat_handler_path"],
        "generation": [],
    },
}

# ---- Per-parameter metadata (tooltips, widget constraints) ----
# Only min/max/step/decimals where bounds are truly fixed regardless of model.

PARAM_META: dict[str, dict] = {
    # ModelConfig
    "model_path":       {"description": "Path to the .gguf model file"},
    "n_ctx":            {"description": "Context window size in tokens", "min": 1},
    "n_gpu_layers":     {"description": "Layers to offload to GPU (-1 = all)"},
    "n_threads":        {"description": "CPU threads for generation (None = auto)"},
    "n_threads_batch":  {"description": "CPU threads for prompt eval (None = auto)"},
    "n_batch":          {"description": "Prompt processing batch size", "min": 1},
    "n_ubatch":         {"description": "Physical batch size for computation", "min": 1},
    "use_mmap":         {"description": "Memory-map model file (faster load, less RAM)"},
    "use_mlock":        {"description": "Lock model in RAM (prevents swapping)"},
    "offload_kqv":      {"description": "Offload KV cache to GPU"},
    "flash_attn":       {"description": "Use flash attention (faster, less VRAM)"},
    "logits_all":       {"description": "Compute logits for all tokens (required for beam search)"},
    "rope_freq_base":   {"description": "RoPE base frequency (0 = model default)"},
    "rope_freq_scale":  {"description": "RoPE frequency scaling factor (0 = model default)"},
    "rope_scaling_type": {"description": "RoPE scaling type (-1 = model default)"},
    "yarn_ext_factor":  {"description": "YaRN extrapolation factor (-1 = model default)"},
    "yarn_attn_factor": {"description": "YaRN attention scaling factor"},
    "yarn_beta_fast":   {"description": "YaRN beta fast"},
    "yarn_beta_slow":   {"description": "YaRN beta slow"},
    "yarn_orig_ctx":    {"description": "Original context size the model was trained with", "min": 0},
    "lora_path":        {"description": "Path to LoRA adapter file"},
    "lora_base":        {"description": "Path to base model for LoRA scaling"},
    "lora_scale":       {"description": "LoRA adapter strength", "min": 0.0, "step": 0.1, "decimals": 2},
    "verbose":          {"description": "Print llama.cpp loading/inference logs"},
    "seed":             {"description": "RNG seed (-1 = random)"},
    "log_image_progress": {"description": "Show image encode/decode progress and timing in basic logs"},
    "log_prompt_debug":   {"description": "Show prompt template dumps during vision generation (verbose logs)"},
    "log_tensor_debug":   {"description": "Show internal tensor/batch state during image tokenization (verbose logs)"},
    "log_backend":        {"description": "Show GPU/backend device init (ggml/cuda/vulkan) in basic logs"},
    "log_model_load":     {"description": "Show model loader, metadata, and context init (verbose logs)"},
    "log_perf_stats":     {"description": "Show end-of-generation llama_perf timing in basic logs"},
    "log_decode":         {"description": "Show per-token decode/eval/sampler logs (verbose, very noisy)"},
    "tensor_split":     {"description": "Fraction of model to put on each GPU"},
    "main_gpu":         {"description": "GPU used for scratch and small tensors", "min": 0},
    "split_mode":       {"description": "How to split across GPUs (1 = layer, 2 = row)", "min": 0, "max": 2},
    "draft_model":      {"description": "Smaller draft model for speculative decoding"},
    "chat_handler_path": {"description": "Path to multimodal vision projector (mmproj) GGUF file"},
    # GenerationConfig
    "max_tokens":       {"description": "Maximum tokens to generate", "min": 1},
    "temperature":      {"description": "Randomness (0 = deterministic, higher = more random)", "min": 0.0, "step": 0.05, "decimals": 2},
    "stop":             {"description": "Stop generation when any of these strings appear"},
    "top_p":            {"description": "Nucleus sampling: consider tokens covering top p% probability", "min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2},
    "top_k":            {"description": "Only sample from the top k tokens", "min": 0},
    "repeat_penalty":   {"description": "Penalise recently used tokens (1.0 = off)", "min": 0.0, "step": 0.05, "decimals": 2},
    "min_p":            {"description": "Minimum probability relative to the top token", "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 2},
    "typical_p":        {"description": "Locally typical sampling threshold (1.0 = off)", "min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2},
    "tfs_z":            {"description": "Tail free sampling z-value (1.0 = off)", "min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2},
    "frequency_penalty": {"description": "Penalise tokens by how often they've appeared", "min": 0.0, "step": 0.05, "decimals": 2},
    "presence_penalty": {"description": "Penalise tokens that have appeared at all", "min": 0.0, "step": 0.05, "decimals": 2},
    "mirostat_mode":    {"description": "Mirostat sampling version (0 = off, 1 or 2)", "min": 0, "max": 2},
    "mirostat_tau":     {"description": "Mirostat target entropy", "min": 0.0, "step": 0.1, "decimals": 1},
    "mirostat_eta":     {"description": "Mirostat learning rate", "min": 0.0, "step": 0.01, "decimals": 2},
    "grammar":          {"description": "GBNF grammar object to constrain output format"},
    "logit_bias":       {"description": "Manually adjust token probabilities {token_id: bias}"},
    "logprobs":         {"description": "Return log probabilities for top N tokens", "min": 0},
    "stream":           {"description": "Yield tokens as they are generated"},
    "echo":             {"description": "Include the prompt in the output"},
    "show_stats":       {"description": "Generation stats display: \"always\", \"basic\" (log mode only), or \"off\""},
    "beam_width":       {"description": "Number of beams (1 = standard greedy/sampling)", "min": 1},
    "beam_depth":       {"description": "Max steps for beam search (0 = use max_tokens)", "min": 0},
    "length_penalty":   {"description": "Normalise beam scores by length (1.0 = off)", "min": 0.0, "step": 0.1, "decimals": 2},
    "beam_log_tree":    {"description": "Print the full beam expansion tree"},
    "beam_top_results": {"description": "How many beams to display (0 = beam_width)", "min": 0},
    "branch_at":        {"description": "Force an alternate token at this generation step", "min": 0},
    "branch_pick":      {"description": "Which rank alternative to pick at the branch point", "min": 0},
    "chat_template":    {"description": "Template name (matches JSON filename stem, or \"none\")"},
    "system_prompt":    {"description": "System message content; empty = skip system block"},
    "use_history":      {"description": "Send prior conversation turns as context to the model"},
    "max_history_turns": {"description": "Maximum user+assistant turn pairs to include", "min": 1, "max": 100},
    "session_mode":     {"description": "Session init: \"new\" (empty), \"recent\" (latest file), \"file\" (specific path)"},
    "session_file":     {"description": "Active session file path (auto-set, or manual for mode=file)"},
}

# Reverse lookups: param_name -> group_name
_LOADING_PARAM_GROUP: dict[str, str] = {}
_GENERATION_PARAM_GROUP: dict[str, str] = {}
for _gname, _gdef in PARAMETER_GROUPS.items():
    for _p in _gdef["loading"]:
        _LOADING_PARAM_GROUP[_p] = _gname
    for _p in _gdef["generation"]:
        _GENERATION_PARAM_GROUP[_p] = _gname

# Params that bypass group toggling (always passed to llama-cpp)
_ALWAYS_PASS_LOADING = frozenset({"verbose"})
_ALWAYS_PASS_GENERATION = frozenset({"stream", "echo"})


def _filter_params(obj, param_group: dict[str, str],
                   always_pass: frozenset, active_groups: list[str]) -> dict:
    """Return kwargs dict for obj, filtered by active_groups."""
    active = set(active_groups) | {"essential"}
    result: dict = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        group = param_group.get(f.name)
        if group is None:
            continue
        if f.name not in always_pass and group not in active:
            continue
        if value is None:
            continue
        result[f.name] = value
    return result


def _collect_visible(obj, active_groups: list[str], key: str) -> dict[str, dict]:
    """Return {group_name: {param: value}} for all active groups."""
    active = set(active_groups) | {"essential"}
    visible: dict[str, dict] = {}
    for gname in active:
        gdef = PARAMETER_GROUPS.get(gname)
        if gdef is None:
            continue
        params = {p: getattr(obj, p) for p in gdef[key] if hasattr(obj, p)}
        if params:
            visible[gname] = params
    return visible


# ---- Configuration dataclasses ----

@dataclass
class ModelConfig:
    """All llama-cpp-python Llama() constructor parameters.

    Defaults allow partial construction (e.g. ModelConfig(model_path=...))
    and act as fallbacks when loading a config file that omits some fields.
    """

    # --- Essential ---
    model_path: str = ""
    n_ctx: int = 4096
    n_gpu_layers: int = -1

    # --- Performance ---
    n_threads: int | None = None
    n_threads_batch: int | None = None
    n_batch: int = 512
    n_ubatch: int = 512
    use_mmap: bool = True
    use_mlock: bool = False
    offload_kqv: bool = True
    flash_attn: bool = False

    # --- Context / Memory ---
    logits_all: bool = False

    # --- RoPE / Extended Context ---
    rope_freq_base: float = 0.0
    rope_freq_scale: float = 0.0
    rope_scaling_type: int = -1
    yarn_ext_factor: float = -1.0
    yarn_attn_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_orig_ctx: int = 0

    # --- LoRA / Adapters ---
    lora_path: str | None = None
    lora_base: str | None = None
    lora_scale: float = 1.0

    # --- Debug / Verbose ---
    verbose: bool = False
    seed: int = -1

    # --- Log category filters (runtime; popped from Llama() kwargs) ---
    log_image_progress: bool = True
    log_prompt_debug:   bool = False
    log_tensor_debug:   bool = False
    log_backend:        bool = True
    log_model_load:     bool = False
    log_perf_stats:     bool = True
    log_decode:         bool = False

    # --- Multi-GPU ---
    tensor_split: list | None = None
    main_gpu: int = 0
    split_mode: int = 1

    # --- Speculative ---
    draft_model: object | None = None

    # --- Multimodal ---
    chat_handler_path: str | None = None

    def to_llama_kwargs(self, active_groups: list[str]) -> dict:
        """Build kwargs dict filtered by *active_groups*.

        Params whose group is toggled OFF are omitted so llama-cpp uses its
        own defaults.  Params in ``_ALWAYS_PASS_LOADING`` are never filtered
        out (e.g. ``verbose``).
        """
        return _filter_params(self, _LOADING_PARAM_GROUP, _ALWAYS_PASS_LOADING, active_groups)

    def get_visible_params(self, active_groups: list[str]) -> dict[str, dict]:
        """Return ``{group_name: {param: value}}`` for display."""
        return _collect_visible(self, active_groups, "loading")


@dataclass
class GenerationConfig:
    """All llama-cpp-python generation / sampling parameters.

    Defaults serve the same purpose as ModelConfig: allow partial construction
    and fill in anything missing from a loaded config file.
    """

    # --- Essential ---
    max_tokens: int = 256
    temperature: float = 0.7
    stop: list[str] | None = None

    # --- Sampling - basic ---
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1

    # --- Sampling - advanced ---
    min_p: float = 0.05
    typical_p: float = 1.0
    tfs_z: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1

    # --- Constraints ---
    grammar: object | None = None
    logit_bias: dict | None = None

    # --- Visibility / Debug ---
    logprobs: int | None = None
    stream: bool = True
    echo: bool = False

    # --- Beam search ---
    beam_width: int = 1
    beam_depth: int = 0
    length_penalty: float = 1.0
    beam_log_tree: bool = False
    beam_top_results: int = 0

    # --- Branching ---
    branch_at: int = 0
    branch_pick: int = 0

    # --- Stats ---
    show_stats: str = "always"    # "always" | "basic" | "off"

    # --- Chat template ---
    chat_template: str = "none"
    system_prompt: str = ""

    # --- Multi-turn history ---
    use_history: bool = False
    max_history_turns: int = 10

    # --- Session ---
    session_mode: str = "new"       # "new" | "recent" | "file"
    session_file: str = ""

    def to_generation_kwargs(self, active_groups: list[str]) -> dict:
        """Build kwargs dict filtered by *active_groups*.

        ``stream`` and ``echo`` are always included (they control code flow).
        """
        kwargs = _filter_params(self, _GENERATION_PARAM_GROUP, _ALWAYS_PASS_GENERATION, active_groups)
        # logprobs=0 is meaningless and requires logits_all=True; treat it as unset.
        if not kwargs.get('logprobs'):
            kwargs.pop('logprobs', None)
        # GBNF grammars are authored as strings in JSON configs; llama-cpp-python
        # expects a LlamaGrammar object at call time.
        if isinstance(kwargs.get('grammar'), str):
            from llama_cpp import LlamaGrammar
            kwargs['grammar'] = LlamaGrammar.from_string(kwargs['grammar'])
        return kwargs

    def get_visible_params(self, active_groups: list[str]) -> dict[str, dict]:
        """Return ``{group_name: {param: value}}`` for display."""
        return _collect_visible(self, active_groups, "generation")


# ---- Visibility logger ----

class ParameterVisibility:
    """Logs parameter state based on active visibility groups."""

    def __init__(self, active_groups: list[str],
                 output_callback: Callable[[str], None] | None = None):
        self.active_groups = list(active_groups)
        self._output = output_callback or print

    def _log_params(self, heading: str, visible: dict[str, dict],
                    label: str):
        if not visible:
            return
        prefix = f"[{label}] " if label else ""
        lines = f"{prefix}{heading} parameters:\n"
        for gname, params in visible.items():
            desc = PARAMETER_GROUPS[gname]["description"]
            lines += f"  [{gname}] {desc}\n"
            for k, v in params.items():
                lines += f"    {k}: {v}\n"
        self._output(f"<basic_log>{lines}</basic_log>")

    def log_loading(self, config: ModelConfig, label: str = ""):
        self._log_params("Loading",
                         config.get_visible_params(self.active_groups), label)

    def log_generation(self, config: GenerationConfig, label: str = ""):
        self._log_params("Generation",
                         config.get_visible_params(self.active_groups), label)

    def log_live_stats(self, tokens_generated: int, elapsed: float,
                       label: str = ""):
        prefix = f"[{label}] " if label else ""
        tps = tokens_generated / elapsed if elapsed > 0 else 0
        self._output(f"{prefix}Live: {tokens_generated} tokens, "
                     f"{elapsed:.2f}s, {tps:.1f} tok/s\n")

    def log_active_groups(self):
        lines = "Parameter groups:\n"
        for gname, gdef in PARAMETER_GROUPS.items():
            on = gname in self.active_groups or gname == "essential"
            tag = "ON" if on else "OFF"
            lines += f"  [{tag:>3}] {gname}: {gdef['description']}\n"
        self._output(f"<basic_log>{lines}</basic_log>")


# ---- Runner config (serialization / deserialization) ----

def _serializable_fields(obj) -> dict:
    """Extract JSON-safe fields from a dataclass instance."""
    result: dict = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        try:
            json.dumps(v)
            result[f.name] = v
        except (TypeError, ValueError):
            pass
    return result

@dataclass
class RunnerConfig:
    model_config: ModelConfig
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    model_name: str = "model"
    active_groups: list[str] = field(
        default_factory=lambda: ["essential", "visibility"]
    )
    module_config: dict = field(default_factory=dict)
    module_access: dict = field(default_factory=dict)
    module_intercept: dict = field(default_factory=dict)
    # {"module_name": {"command_name": True/False}}

    @classmethod
    def from_file(cls, path: str, model_path: str = "") -> "RunnerConfig":
        """Load from a JSON config file.

        *model_path* overrides the value in the file (useful when the user
        picks a model interactively).
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Resolve active groups from the toggle map
        group_toggles = data.get("active_groups", {})
        active = [g for g, on in group_toggles.items() if on]

        # Build ModelConfig (filter to valid fields, skip nulls)
        mc_fields = {f.name for f in fields(ModelConfig)}
        mc_data = {k: v for k, v in data.get("model_config", {}).items()
                   if k in mc_fields and v is not None}
        if model_path:
            mc_data["model_path"] = model_path
        mc = ModelConfig(**mc_data)

        # Build GenerationConfig
        gc_fields = {f.name for f in fields(GenerationConfig)}
        gc_data = {k: v for k, v in data.get("generation_config", {}).items()
                   if k in gc_fields and v is not None}
        gc = GenerationConfig(**gc_data)

        return cls(
            model_config=mc,
            generation_config=gc,
            model_name=data.get("model_name", "model"),
            active_groups=active,
            module_config=data.get("module_config", {}),
            module_access=data.get("module_access", {}),
            module_intercept=data.get("module_intercept", {}),
        )

    def to_file(self, path: str, module_schemas: dict | None = None,
                intercept_schemas: dict | None = None):
        """Persist current config to a JSON file.

        *module_schemas* is an optional ``{name: [ModuleParam, ...]}`` dict.
        When provided, enabled installed modules get their config filled with
        schema defaults (self-documenting).  Disabled modules are omitted.
        Uninstalled-but-enabled module configs are preserved as-is.

        *intercept_schemas* is an optional ``{name: [InterceptableCommand, ...]}``
        dict.  When provided, enabled modules get their intercept entries filled
        with schema defaults (``intercept``).  When ``None``, the
        existing ``module_intercept`` data is preserved as-is.
        """
        group_toggles = {
            g: (g in self.active_groups or g == "essential")
            for g in PARAMETER_GROUPS
        }

        mc_dict = _serializable_fields(self.model_config)
        gc_dict = _serializable_fields(self.generation_config)

        # Build module_config / module_access for output.
        # A module is "enabled" only if module_access says True (opt-in).
        # Disabled modules are omitted from module_config.
        out_module_config: dict = {}
        out_module_access: dict = {}
        out_module_intercept: dict = {}
        schemas = module_schemas or {}
        i_schemas = intercept_schemas or {}

        # Collect every module name the user has explicitly touched
        all_mod_names = set(self.module_access) | set(self.module_config)

        for mod_name in all_mod_names:
            enabled = self.module_access.get(mod_name, False)
            out_module_access[mod_name] = enabled
            if not enabled:
                continue
            cfg = dict(self.module_config.get(mod_name, {}))
            if mod_name in schemas:
                for p in schemas[mod_name]:
                    cfg.setdefault(p.name, p.default)
            out_module_config[mod_name] = cfg

            # Build intercept settings for this module
            existing = self.module_intercept.get(mod_name, {})
            if i_schemas and mod_name in i_schemas:
                intercept_cfg = dict(existing)
                for cmd in i_schemas[mod_name]:
                    intercept_cfg.setdefault(cmd.name, cmd.intercept)
                out_module_intercept[mod_name] = intercept_cfg
            elif existing:
                out_module_intercept[mod_name] = existing

        data = {
            "model_name": self.model_name,
            "active_groups": group_toggles,
            "model_config": mc_dict,
            "generation_config": gc_dict,
            "module_config": out_module_config,
            "module_access": out_module_access,
            "module_intercept": out_module_intercept,
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
