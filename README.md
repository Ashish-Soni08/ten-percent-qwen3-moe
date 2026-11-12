# Qwen3.5-35B-A3B at 9.72% of fp16 for safety moderation, no training

This is an entry for the [10% challenge](https://github.com/DeusExperiments/ten-percent-challenge): safety moderation. **It does not pass rule 2.** On the locked exam the compressed model keeps **95.04%** of the items the base model gets right (the rule needs 98%, and 95% in every category; 3 of 6 categories reach 95%). We publish it as a complete, honest measurement.

## Result at a glance

| Rule | Requirement | Result | Status |
|---|---|---|---|
| 1. Size | ≤ 10% of fp16 (71,903,655,008 B) | Artifact (weights, tokenizer, config) 6,985,902,688 B, plus about 0.2 MB of code and the feedback state (a JSON list of integers): **9.716%** | Pass |
| 2. Faithful | Keep ≥ 98% of base-correct items, ≥ 95% in every category | **95.04%** overall (3524 / 3708); lowest category or source aegis_response/unsafe 85.47% | **Fail** |
| 3. Exam | ≥ 2,000 locked items, ≥ 300 per category, ≥ 500 unseen | 4,395 locked items in 6 categories; 1,680 from unseen sources | Pass |
| 4. No new trained parts | No heads, adapters, memory, or prompt feedback | Experts are exact slices of the base, quantized post-training with no data-fitted values. Feedback state is a list of integers. | Pass (see the rule-5 note) |
| 5. Feedback teaches | ≤ 25 real mistakes; locked probes improve; nearby kept | Locked probes 1 / 229 (0.44%) → **64 / 229 (27.95%)**. Nearby items kept 292 / 300 (97.33%). | Pass |
| 6. Controls | Shuffled and empty clearly worse | Shuffled 0 / 229 (0.00%), empty 1 / 229 (0.44%), real 64 / 229 (27.95%) | Pass |
| 7. Real job | Real output, no multiple choice | The model writes the label itself by free greedy generation; compared as an exact string | Pass |
| 8. Something to keep | Well above chance, below ceiling | Base 84.4% on the exam | Pass |

## Model, task, data

- **Base:** `Qwen/Qwen3.5-35B-A3B` (Apache-2.0). Hybrid MoE: 40 layers (30 linear-attention, 10 full-attention), 256 routed experts plus a shared expert per layer, top-8 routing. fp16 size 71,903,655,008 B (from its `model.safetensors.index.json`), including the vision tower and MTP layer.
- **Task:** Content-safety classification. A policy is given in the system prompt and the model answers `safe` or `unsafe` for a user prompt or a prompt-response pair, by free greedy generation.
- **In-distribution source:** Aegis 2.0 (NVIDIA, CC-BY-4.0): calibration 2,500 train items, feedback pool 1,688 validation items, locked exam 2,715 test items.
- **Unseen source:** OpenAI moderation evaluation set (MIT): 1,680 locked exam items, never used to build, tune or calibrate.

## Compression (no gradients anywhere, no data-fitted values)

1. **Text model only.** The vision tower and the MTP layer are dropped; the shipped config is the text-only `Qwen3_5MoeForCausalLM`.
2. **Expert selection.** Keep the 108 of 256 routed experts per layer with the most routing mass on the task's calibration prompts (`data/saf_calib.jsonl`), each followed by the base model's own answer (`fc/q35_route_stats.py`). Router rows are sliced to match. K = 108 is the largest uniform count that fits the size budget; it was fixed before any evaluation.
3. **Quantization** (`fc/q35_build.py`, `fc/qpack.py`): round-to-nearest, groups of 128, fp16 scale and minimum, with the clipping ratio picked per group from a fixed grid by weight reconstruction error. Routed experts 3-bit; attention, linear-attention projections, shared expert and embeddings 4-bit; LM head 6-bit. Routers, norms, convolutions and other small tensors stay bf16.
4. **Runtime.** `fc/q35_unpack.py` expands the artifact to a bf16 checkpoint once (a cache, not shipped); SGLang serves it with deterministic inference.

## Locked exam

`data/saf_exam.jsonl` (shipped gzipped as `data/locked/exam.jsonl.gz`) has sha256 `a4b3c7cfffab542b1d2a33b5148a98913d7ad9d5b6d6a02584c6a0ca614948c1`. It was opened once, after the build was fixed.

| Category | n | Base acc | Build acc | Kept of base-correct |
|---|---|---|---|---|
| **ALL** | 4,395 | 84.37% | 83.48% | **95.04% (3524 / 3708)** |
| source: aegis_prompt | 1,902 | 81.91% | 80.39% | 94.87% (1478 / 1558) |
| source: aegis_response | 813 | 81.06% | 79.46% | 91.50% (603 / 659) |
| source: openai_mod | 1,680 | 88.75% | 88.93% | 96.78% (1443 / 1491) |
| aegis_prompt/safe | 878 | 86.79% | 90.66% | 98.69% (752 / 762) |
| aegis_prompt/unsafe | 1,024 | 77.73% | 71.58% | 91.21% (726 / 796) |
| aegis_response/safe | 419 | 86.63% | 89.74% | 96.42% (350 / 363) |
| aegis_response/unsafe | 394 | 75.13% | 68.53% | 85.47% (253 / 296) |
| openai_mod/safe | 1,158 | 89.29% | 90.67% | 97.58% (1009 / 1034) |
| openai_mod/unsafe | 522 | 87.55% | 85.06% | 94.97% (434 / 457) |

After the real feedback (5 experts switched off) the same exam keeps 95.50% (3541 / 3708).

## Feedback: switching existing experts off (no weight changes)

`python fc/q35_cli.py learn --artifact ART --in feedback.jsonl --guard guard.jsonl --out state.json` takes at most 25 annotated mistakes and writes only a list of `(layer, expert)` integers. `python fc/q35_cli.py answer --artifact ART --state state.json --in q.jsonl --out a.jsonl` runs the model with those experts switched off: the router never selects them (a fixed -1e4 router offset; every other offset is 0).

- Every weight tensor is byte-identical before and after feedback. No head, adapter, lookup memory, or prompt change.
- Experts are ranked by first-order attribution toward the annotated answers, switched off a few at a time with each step verified by real forward passes, and a switch is refused if it changes any of the model's own answers on a 160-item guard set.

**Feedback class.** Real mistakes of the compressed build on feedback-pool items it never calibrated on, for the gold label(s) where it errs most. 25 become feedback; the rest are locked probes (229), written to disk before any feedback ran. Nearby items are 300 pool items of the same and the confused labels that the build answers right. Files and sha256 are in `data/feedback/`.

| Arm | Locked probes right | Feedback items right | Nearby kept | Experts off |
|---|---|---|---|---|
| Empty | 1 / 229 (0.44%) | 0 / 25 (0.00%) | 300 / 300 (100.00%) | 0 |
| Shuffled labels | 0 / 229 (0.00%) | 0 / 25 (0.00%) | 297 / 300 (99.00%) | 1 |
| **Real** | **64 / 229 (27.95%)** | 7 / 25 (28.00%) | 292 / 300 (97.33%) | 5 |
| (Base model, for reference) | 63 / 229 (27.51%) | | 291 / 300 (97.00%) | |

**Rule-5 note.** The organizer rejected #3 and #4 because their feedback wrote values fitted to the annotations into weights or heads. This method writes no values: it chooses which of the model's own experts stay on. We think that is on the allowed side, but the choice is driven by the feedback, so we flag it as a gray zone.

## Disclosures

- This task was built on Qwen3.5 with no prior dev test on this model; the recipe (108 experts per layer, the same quantization) is the one fixed for all our Qwen3.5 entries.
- The safety exam items were run through the base model and an older Qwen3-30B build on 11 Nov when we first tried this task on that model; they were never used to build, tune or calibrate this Qwen3.5 build.
- Weights are not hosted yet; the artifact is rebuilt from the base checkpoint by the scripts below.

## Reproduce

Shipped data: `data/dev/*.jsonl.gz` (calibration, dev and test-probe sets), `data/locked/exam.jsonl.gz` (the locked exam), `data/feedback/` (feedback pool, feedback items, shuffled feedback, locked probes, nearby items, guard set, with `SHA256SUMS`). Results: `results/` (feedback states, exam reports, and the full output of the final run in `results/run_output.txt`).

All GPU work ran on one H100 (Modal). Scripts run with the repository root as working directory and `/data` mounted.

1. Data: the shipped files in `data/` (Aegis 2.0 train/validation/test and the OpenAI moderation eval, rendered with the policy prompt shown in each item).
2. Base answers on the calibration prompts: `python3 fc/gen_sglang.py --model Qwen/Qwen3.5-35B-A3B --out base --max-new 4 --split data/saf_calib.jsonl`; routing mass: `python3 fc/q35_route_stats.py --model ... --items data/saf_calib.jsonl --answers base/... --out rs.pt`.
3. Build, exam, feedback and controls: `bash runs/q35_task.sh saf 108 4 data/saf_exam.jsonl data/saf_pool.jsonl`.
4. Commands: `fc/q35_cli.py answer` and `fc/q35_cli.py learn` (docstring at the top of the file).

## Licences

- Code: Apache-2.0 (see `LICENSE`). Base model: Apache-2.0.
- Aegis 2.0 CC-BY-4.0. OpenAI moderation eval MIT. Exam and feedback files are derived from them with attribution.
