# Qwen3.5-35B-A3B at 9.72% of fp16 for prompt injection, no training

This is an entry for the [10% challenge](https://github.com/DeusExperiments/ten-percent-challenge): prompt-injection detection. **It does not pass rule 2.** On the locked exam the compressed model keeps **88.24%** of the items the base model gets right (the rule needs 98%, and 95% in every category; 3 of 6 categories reach 95%). We publish it as a complete, honest measurement.

## Result at a glance

| Rule | Requirement | Result | Status |
|---|---|---|---|
| 1. Size | ≤ 10% of fp16 (71,903,655,008 B) | Artifact (weights, tokenizer, config) 6,985,902,737 B, plus about 0.2 MB of code and the feedback state (a JSON list of integers): **9.716%** | Pass |
| 2. Faithful | Keep ≥ 98% of base-correct items, ≥ 95% in every category | **88.24%** overall (3683 / 4174); lowest category or source piarena/yes 74.13% | **Fail** |
| 3. Exam | ≥ 2,000 locked items, ≥ 300 per category, ≥ 500 unseen | 4,900 locked items in 6 categories; 2,500 from unseen sources | Pass |
| 4. No new trained parts | No heads, adapters, memory, or prompt feedback | Experts are exact slices of the base, quantized post-training with no data-fitted values. Feedback state is a list of integers. | Pass (see the rule-5 note) |
| 5. Feedback teaches | ≤ 25 real mistakes; locked probes improve; nearby kept | Locked probes 4 / 336 (1.19%) → **243 / 336 (72.32%)**. Nearby items kept 284 / 298 (95.30%). | Pass |
| 6. Controls | Shuffled and empty clearly worse | Shuffled 15 / 336 (4.46%), empty 4 / 336 (1.19%), real 243 / 336 (72.32%) | Pass |
| 7. Real job | Real output, no multiple choice | The model writes the label itself by free greedy generation; compared as an exact string | Pass |
| 8. Something to keep | Well above chance, below ceiling | Base 85.2% on the exam | Pass |

## Model, task, data

- **Base:** `Qwen/Qwen3.5-35B-A3B` (Apache-2.0). Hybrid MoE: 40 layers (30 linear-attention, 10 full-attention), 256 routed experts plus a shared expert per layer, top-8 routing. fp16 size 71,903,655,008 B (from its `model.safetensors.index.json`), including the vision tower and MTP layer.
- **Task:** Indirect prompt-injection detection for AI agents. The user message is content a tool returned (email, document, web page, code); the model answers `yes` if it contains instructions aimed at the agent, else `no`, by free greedy generation.
- **In-distribution source:** Horizon-Labs/prompt-injection-eval-suite configs `bipia` and `synthetic_docs_test`: calibration 1,000, dev 1,000, locked exam 2,400, feedback pool 2,132 (disjoint).
- **Unseen source:** configs `piarena` and `llmail_phase2` of the same suite (different sources and attack styles): 2,500 locked exam items, never used to build, tune or calibrate.

## Compression (no gradients anywhere, no data-fitted values)

1. **Text model only.** The vision tower and the MTP layer are dropped; the shipped config is the text-only `Qwen3_5MoeForCausalLM`.
2. **Expert selection.** Keep the 108 of 256 routed experts per layer with the most routing mass on the task's calibration prompts (`data/pi_calib.jsonl`), each followed by the base model's own answer (`fc/q35_route_stats.py`). Router rows are sliced to match. K = 108 is the largest uniform count that fits the size budget; it was fixed before any evaluation.
3. **Quantization** (`fc/q35_build.py`, `fc/qpack.py`): round-to-nearest, groups of 128, fp16 scale and minimum, with the clipping ratio picked per group from a fixed grid by weight reconstruction error. Routed experts 3-bit; attention, linear-attention projections, shared expert and embeddings 4-bit; LM head 6-bit. Routers, norms, convolutions and other small tensors stay bf16.
4. **Runtime.** `fc/q35_unpack.py` expands the artifact to a bf16 checkpoint once (a cache, not shipped); SGLang serves it with deterministic inference.

## Locked exam

`data/pi_exam.jsonl` (shipped gzipped as `data/locked/exam.jsonl.gz`) has sha256 `6eb50b833e040d6bc46b388795c89145c1b1b0b75c7414a13bedcc146bd0ce02`. It was opened once, after the build was fixed.

| Category | n | Base acc | Build acc | Kept of base-correct |
|---|---|---|---|---|
| **ALL** | 4,900 | 85.18% | 76.45% | **88.24% (3683 / 4174)** |
| source: llmail | 1,454 | 66.99% | 57.98% | 83.78% (816 / 974) |
| source: piarena | 1,046 | 97.80% | 86.14% | 87.59% (896 / 1023) |
| source: sdocs | 1,246 | 97.11% | 97.43% | 98.76% (1195 / 1210) |
| bipia | 1,154 | 83.80% | 68.28% | 80.25% (776 / 967) |
| llmail/yes | 1,454 | 66.99% | 57.98% | 83.78% (816 / 974) |
| piarena/no | 536 | 99.25% | 100.00% | 100.00% (532 / 532) |
| piarena/yes | 510 | 96.27% | 71.57% | 74.13% (364 / 491) |
| sdocs/no | 840 | 96.19% | 97.50% | 99.01% (800 / 808) |
| sdocs/yes | 406 | 99.01% | 97.29% | 98.26% (395 / 402) |

After the real feedback (18 experts switched off) the same exam keeps 95.78% (3998 / 4174).

## Feedback: switching existing experts off (no weight changes)

`python fc/q35_cli.py learn --artifact ART --in feedback.jsonl --guard guard.jsonl --out state.json` takes at most 25 annotated mistakes and writes only a list of `(layer, expert)` integers. `python fc/q35_cli.py answer --artifact ART --state state.json --in q.jsonl --out a.jsonl` runs the model with those experts switched off: the router never selects them (a fixed -1e4 router offset; every other offset is 0).

- Every weight tensor is byte-identical before and after feedback. No head, adapter, lookup memory, or prompt change.
- Experts are ranked by first-order attribution toward the annotated answers, switched off a few at a time with each step verified by real forward passes, and a switch is refused if it changes any of the model's own answers on a 160-item guard set.

**Feedback class.** Real mistakes of the compressed build on feedback-pool items it never calibrated on, for the gold label(s) where it errs most. 25 become feedback; the rest are locked probes (336), written to disk before any feedback ran. Nearby items are 300 pool items of the same and the confused labels that the build answers right. Files and sha256 are in `data/feedback/`.

| Arm | Locked probes right | Feedback items right | Nearby kept | Experts off |
|---|---|---|---|---|
| Empty | 4 / 336 (1.19%) | 0 / 25 (0.00%) | 298 / 300 (99.33%) | 0 |
| Shuffled labels | 15 / 336 (4.46%) | 0 / 25 (0.00%) | 293 / 300 (97.67%) | 3 |
| **Real** | **243 / 336 (72.32%)** | 22 / 25 (88.00%) | 285 / 300 (95.00%) | 18 |
| (Base model, for reference) | 199 / 336 (59.23%) | | 296 / 300 (98.67%) | |

Here the feedback also repairs much of what compression broke: the real feedback state raises the locked exam from 88.24% to 95.78% kept, and the probes go above the uncompressed base (243 vs 199 of 336). The compressed model's main failure is missing injections (answering `no` on bipia and piarena attacks); our pre-build dev check (104 experts, 97.3% kept on 1,000 dev items) did not show this gap, so the locked exam is much harsher than dev.

**Rule-5 note.** The organizer rejected #3 and #4 because their feedback wrote values fitted to the annotations into weights or heads. This method writes no values: it chooses which of the model's own experts stay on. We think that is on the allowed side, but the choice is driven by the feedback, so we flag it as a gray zone.

## Disclosures

- Before the build we ran a small test of several tasks on this model (base accuracy, and kept share with 104 experts per layer and plain 3-bit rounding) on dev items and on a separate probe sample of the unseen source, and chose which tasks to build from those numbers. The locked exam items are disjoint from every item used there.
- Weights are not hosted yet; the artifact is rebuilt from the base checkpoint by the scripts below.

## Reproduce

Shipped data: `data/dev/*.jsonl.gz` (calibration, dev and test-probe sets), `data/locked/exam.jsonl.gz` (the locked exam), `data/feedback/` (feedback pool, feedback items, shuffled feedback, locked probes, nearby items, guard set, with `SHA256SUMS`). Results: `results/` (feedback states, exam reports, and the full output of the final run in `results/run_output.txt`).

All GPU work ran on one H100 (Modal). Scripts run with the repository root as working directory and `/data` mounted.

1. Data: `python3 fc/pi_prep.py` (raw files under `data_raw/`). Shipped copies are in `data/`.
2. Base answers on the calibration prompts: `python3 fc/gen_sglang.py --model Qwen/Qwen3.5-35B-A3B --out base --max-new 4 --split data/pi_calib.jsonl`; routing mass: `python3 fc/q35_route_stats.py --model ... --items data/pi_calib.jsonl --answers base/... --out rs.pt`.
3. Build, exam, feedback and controls: `bash runs/q35_task.sh pi 108 4 data/pi_exam.jsonl data/pi_pool.jsonl`.
4. Commands: `fc/q35_cli.py answer` and `fc/q35_cli.py learn` (docstring at the top of the file).

## Licences

- Code: Apache-2.0 (see `LICENSE`). Base model: Apache-2.0.
- Horizon-Labs/prompt-injection-eval-suite: per-config licences (CC-BY-SA-4.0 / Apache-2.0 / MIT); see the dataset card. Exam and feedback files are derived from them with attribution.
