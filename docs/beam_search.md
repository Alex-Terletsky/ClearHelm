# Beam Search

Beam search is a heuristic search algorithm that finds high-quality sequences without exhaustively exploring all possibilities. Instead of committing to one token at a time (greedy decoding), it maintains **k parallel candidates** (the beam width) at every step, keeping only the most promising ones before expanding further.

---

## How It Works

Standard greedy decoding picks the single highest-probability token at each step. This is fast but can get stuck — a slightly lower-probability early choice might lead to a much better overall sequence. Beam search hedges against this by tracking multiple candidates simultaneously.

```
Step 0:  [prompt]
          ↓
Step 1:  [beam 1: " The"]  [beam 2: " A"]  [beam 3: " Once"]
          ↓                  ↓                ↓
Step 2:  [beam 1: " The cat"] [beam 2: " A large"] [beam 3: " Once upon"]
          ...
```

At each step:
1. Every active beam is expanded — the top `beam_width * 2` next tokens are generated from each.
2. All candidates across all beams are scored together.
3. Only the top `beam_width` non-finished candidates survive into the next step.
4. Candidates that produce an EOS token (or a stop string) are moved to the **completed** pool.
5. Search ends when `max_tokens` is reached, all beams complete, or every completed sequence already outscores every active one.

The highest-scoring completed sequence is returned.

---

## Scoring

Each candidate accumulates a **cumulative log-probability** as tokens are added:

```
cum_logp += log P(token_i | prompt + all previous tokens)
```

Log-softmax is applied to the raw model logits to convert them to proper log-probabilities:

```
log_probs = scores - max(scores) - log(sum(exp(scores - max(scores))))
```

### Length Normalization

Raw cumulative log-probabilities are negative and grow more negative with length, which would unfairly favour short sequences. Scores are normalized before ranking:

```
norm_score = cum_logp / (sequence_length ^ length_penalty)
```

With `length_penalty = 1.0` (the default) this is a simple average log-probability per token. Setting it lower (e.g. `0.6`) reduces how much length is penalised, favouring longer outputs.

---

## Prerequisites

Beam search requires the model to expose logits for every token position, not just the last one. This is controlled by a **model loading parameter**:

```
logits_all = true
```

This is a **loading parameter** — it is set when the model is loaded and cannot be changed without reloading. If you attempt beam search with `logits_all = false`, generation will fall back to standard sampling and log an error:

```
ERROR: beam search requires logits_all=True.
Enable it in the visibility group and reload the model.
```

To enable it, turn on the `beam_search` visibility group in the UI (which surfaces the relevant parameters) and ensure `logits_all` is checked before loading your model. The `fullvis.json` config has this pre-configured.

---

## Parameters

All beam search parameters live in the **Beam search and branching** parameter group. The group must be toggled on in the parameter panel for these controls to appear and take effect.

### `beam_width` — number of beams
**Type:** `int` | **Default:** `1`

The number of candidate sequences maintained in parallel. Set to `1` to disable beam search entirely (falls back to standard sampling). Higher values explore more of the probability space but multiply compute and memory cost.

| Value | Behaviour |
|---|---|
| `1` | Standard greedy/sampling — beam search inactive |
| `2–4` | Light beam search — modest quality improvement |
| `4–8` | Typical sweet spot for text generation |
| `>10` | Diminishing returns; significant slowdown |

Beam search only activates when `beam_width > 1`.

### `beam_depth` — maximum steps
**Type:** `int` | **Default:** `0`

Caps how many tokens the beam search will generate. When `0`, the global `max_tokens` setting is used instead. Set this to a smaller value if you want beam search to run for a fixed number of steps regardless of the main token budget.

### `length_penalty` — score normalization exponent
**Type:** `float` | **Default:** `1.0`

Controls how much sequences are penalised for being long when scores are compared. Computed as:

```
norm_score = cum_logp / (length ^ length_penalty)
```

| Value | Effect |
|---|---|
| `0.0` | No normalization — raw cumulative score (strongly prefers short output) |
| `0.6–0.7` | Common in NLP literature — mild length tolerance |
| `1.0` | Per-token average log-probability — neutral |
| `>1.0` | Rewards longer sequences |

Start with the default `1.0` and lower it slightly if outputs are being cut short unexpectedly.

### `beam_log_tree` — expansion tree logging
**Type:** `bool` | **Default:** `false`

When enabled, prints a detailed trace of every beam expansion step to the output panel. For each step and each active beam, it shows:

- The top candidate tokens with their log-probabilities
- Which tokens survived the pruning step

Example output:
```
[tree] Step 1, Beam 1 -> " The"(-0.45) " A"(-1.12) " Once"(-2.30) " It"(-2.88)
  [kept] " The" (cum: -0.45, norm: -0.45)
  [kept] " A"   (cum: -1.12, norm: -1.12)
```

This is useful for understanding why beam search chose a particular output, but produces a lot of output — leave it off for normal use.

### `beam_top_results` — results to display
**Type:** `int` | **Default:** `0`

How many completed beams to show in the output after search ends. When `0`, defaults to showing all `beam_width` results. Set to `1` to suppress the ranked list and only show the winning sequence.

The output always looks like:

```
========================================
BEAM SEARCH RESULTS (3 beams):
  [Beam 1] score: -0.3821
  The cat sat on the mat and looked outside.

  [Beam 2] score: -0.4102
  The cat sat on the mat quietly.

  [Beam 3] score: -0.4389 [TRUNCATED]
  The cat sat near the window
```

`[TRUNCATED]` means the beam hit the token limit rather than producing an EOS token.

The **highest-scoring beam is always returned** as the final response, regardless of how many are displayed.

---

## Branch-at-Step

Branch-at-step is a separate but related mode that lets you explore what would have happened if the model had chosen differently at a specific token position. It is activated when `branch_at > 0` and `beam_width` is left at `1`.

### `branch_at` — step to branch at
**Type:** `int` | **Default:** `0`

The generation step (1-indexed) at which to force an alternative token. All steps before and after this point use normal sampling. Set to `0` to disable branching.

### `branch_pick` — which alternative to choose
**Type:** `int` | **Default:** `0`

At the branch point, the top 5 token candidates are ranked by probability. `branch_pick = 0` always picks **rank #1** (the second-best token, since rank #0 is greedy and is skipped). `branch_pick = 1` picks rank #2, and so on.

The output at the branch point shows the full candidate list so you can see what was chosen:

```
[branch] Step 4: top candidates:
  #0 " sat"  (log_prob: -0.31)  <- would have been chosen
  #1 " lay"  (log_prob: -1.45)  <- PICKED
  #2 " slept"(log_prob: -2.10)
  #3 " ran"  (log_prob: -2.78)
  #4 " hid"  (log_prob: -3.12)
```

This is useful for creative exploration — run the same prompt multiple times with different `branch_at` and `branch_pick` values to see how small divergences cascade into different continuations.

---

## Progress Output

During a beam search run, a progress line is printed every 5 steps:

```
[beam] Step 10/256 | active: 4, done: 2 | best: "The cat sat on the mat and lo..."
```

This shows the current step, how many beams are still active, how many have finished, and a 60-character preview of the current best beam.

---

## Common Configurations

**Default (beam search off)**
```json
"beam_width": 1
```

**Light beam search — fast, modest quality gain**
```json
"beam_width": 4,
"beam_depth": 0,
"length_penalty": 1.0
```

**Thorough search with debug output**
```json
"beam_width": 8,
"length_penalty": 0.7,
"beam_log_tree": true,
"beam_top_results": 4
```

**Show only the top result**
```json
"beam_width": 4,
"beam_top_results": 1
```

**Explore a narrative branch**
```json
"beam_width": 1,
"branch_at": 5,
"branch_pick": 0
```

---

## Performance Notes

- Each beam requires its own forward pass and saved KV-cache state. **Memory scales linearly with `beam_width`**.
- Generation is roughly `beam_width` times slower than standard sampling.
- `beam_depth` can be used to limit the search to the most important early tokens, after which the winning beam continues with normal sampling.
- Beam search does not stream tokens as they are generated — output only appears once the search is complete.
- Branch-at-step does stream tokens normally after the forced branch point.
