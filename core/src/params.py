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
        "loading": ["verbose", "logits_all", "seed"],
        "generation": ["logprobs", "stream", "echo"],
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
        "generation": ["chat_template", "system_prompt"],
    },
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
    model_path: str = ""                   # path to the .gguf model file
    n_ctx: int = 4096                      # context window size in tokens
    n_gpu_layers: int = -1                 # layers to offload to GPU (-1 = all)

    # --- Performance ---
    n_threads: int | None = None           # CPU threads for generation (None = auto)
    n_threads_batch: int | None = None     # CPU threads for prompt eval (None = auto)
    n_batch: int = 512                     # prompt processing batch size
    n_ubatch: int = 512                    # physical batch size for computation
    use_mmap: bool = True                  # memory-map model file (faster load, less RAM)
    use_mlock: bool = False                # lock model in RAM (prevents swapping)
    offload_kqv: bool = True               # offload KV cache to GPU
    flash_attn: bool = False               # use flash attention (faster, less VRAM)

    # --- Context / Memory ---
    logits_all: bool = False               # compute logits for all tokens (required for beam search)

    # --- RoPE / Extended Context ---
    rope_freq_base: float = 0.0            # RoPE base frequency (0 = model default)
    rope_freq_scale: float = 0.0           # RoPE frequency scaling factor (0 = model default)
    rope_scaling_type: int = -1            # RoPE scaling type (-1 = model default)
    yarn_ext_factor: float = -1.0          # YaRN extrapolation factor (-1 = model default)
    yarn_attn_factor: float = 1.0          # YaRN attention scaling factor
    yarn_beta_fast: float = 32.0           # YaRN beta fast
    yarn_beta_slow: float = 1.0            # YaRN beta slow
    yarn_orig_ctx: int = 0                 # original context size the model was trained with

    # --- LoRA / Adapters ---
    lora_path: str | None = None           # path to LoRA adapter file
    lora_base: str | None = None           # path to base model for LoRA scaling
    lora_scale: float = 1.0                # LoRA adapter strength

    # --- Debug / Verbose ---
    verbose: bool = False                  # print llama.cpp loading/inference logs
    seed: int = -1                         # RNG seed (-1 = random)

    # --- Multi-GPU ---
    tensor_split: list | None = None       # fraction of model to put on each GPU
    main_gpu: int = 0                      # GPU used for scratch and small tensors
    split_mode: int = 1                    # how to split across GPUs (1 = layer, 2 = row)

    # --- Speculative ---
    draft_model: object | None = None      # smaller draft model for speculative decoding

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
    max_tokens: int = 256              # maximum tokens to generate
    temperature: float = 0.7           # randomness (0 = deterministic, higher = more random)
    stop: list[str] | None = None      # stop generation when any of these strings appear

    # --- Sampling - basic ---
    top_p: float = 0.95                # nucleus sampling: consider tokens covering top p% probability
    top_k: int = 40                    # only sample from the top k tokens
    repeat_penalty: float = 1.1        # penalise recently used tokens (1.0 = off)

    # --- Sampling - advanced ---
    min_p: float = 0.05                # minimum probability relative to the top token
    typical_p: float = 1.0             # locally typical sampling threshold (1.0 = off)
    tfs_z: float = 1.0                 # tail free sampling z-value (1.0 = off)
    frequency_penalty: float = 0.0     # penalise tokens by how often they've appeared
    presence_penalty: float = 0.0      # penalise tokens that have appeared at all
    mirostat_mode: int = 0             # mirostat sampling version (0 = off, 1 or 2)
    mirostat_tau: float = 5.0          # mirostat target entropy
    mirostat_eta: float = 0.1          # mirostat learning rate

    # --- Constraints ---
    grammar: object | None = None      # GBNF grammar object to constrain output format
    logit_bias: dict | None = None     # manually adjust token probabilities {token_id: bias}

    # --- Visibility / Debug ---
    logprobs: int | None = None        # return log probabilities for top N tokens
    stream: bool = True                # yield tokens as they are generated
    echo: bool = False                 # include the prompt in the output

    # --- Beam search ---
    beam_width: int = 1                # number of beams (1 = standard greedy/sampling)
    beam_depth: int = 0                # max steps for beam search (0 = use max_tokens)
    length_penalty: float = 1.0        # normalise beam scores by length (1.0 = off)
    beam_log_tree: bool = False        # print the full beam expansion tree
    beam_top_results: int = 0          # how many beams to display (0 = beam_width)

    # --- Branching ---
    branch_at: int = 0                 # force an alternate token at this generation step
    branch_pick: int = 0               # which rank alternative to pick at the branch point

    # --- Chat template ---
    chat_template: str = "none"        # template name (matches JSON filename stem, or "none")
    system_prompt: str = ""            # system message content; empty = skip system block

    def to_generation_kwargs(self, active_groups: list[str]) -> dict:
        """Build kwargs dict filtered by *active_groups*.

        ``stream`` and ``echo`` are always included (they control code flow).
        """
        kwargs = _filter_params(self, _GENERATION_PARAM_GROUP, _ALWAYS_PASS_GENERATION, active_groups)
        # logprobs=0 is meaningless and requires logits_all=True; treat it as unset.
        if not kwargs.get('logprobs'):
            kwargs.pop('logprobs', None)
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
