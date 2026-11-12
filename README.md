# Qwen3.5-35B-A3B at 9.72% of fp16 for language ID, no training

This is an entry for the [10% challenge](https://github.com/DeusExperiments/ten-percent-challenge): language identification. **It does not pass rule 2.** On the locked exam the compressed model keeps **98.03%** of the items the base model gets right, which meets the 98% overall bar, but 5 of 32 categories fall below 95% (lowest: massive/Malay 84.83%). We publish it as a complete, honest measurement.

## Result at a glance

| Rule | Requirement | Result | Status |
|---|---|---|---|
| 1. Size | ≤ 10% of fp16 (71,903,655,008 B) | Artifact (weights, tokenizer, config) 6,985,902,729 B, plus about 0.2 MB of code and the feedback state (a JSON list of integers): **9.716%** | Pass |
| 2. Faithful | Keep ≥ 98% of base-correct items, ≥ 95% in every category | **98.03%** overall (9002 / 9183); lowest category or source massive/Malay 84.83% | **Fail** |
| 3. Exam | ≥ 2,000 locked items, ≥ 300 per category, ≥ 500 unseen | 9,600 locked items in 32 categories; 4,800 from unseen sources | Pass |
| 4. No new trained parts | No heads, adapters, memory, or prompt feedback | Experts are exact slices of the base, quantized post-training with no data-fitted values. Feedback state is a list of integers. | Pass (see the rule-5 note) |
| 5. Feedback teaches | ≤ 25 real mistakes; locked probes improve; nearby kept | Locked probes 0 / 87 (0.00%) → **58 / 87 (66.67%)**. Nearby items kept 298 / 300 (99.33%). | Pass |
| 6. Controls | Shuffled and empty clearly worse | Shuffled 0 / 87 (0.00%), empty 0 / 87 (0.00%), real 58 / 87 (66.67%) | Pass |
| 7. Real job | Real output, no multiple choice | The model writes the label itself by free greedy generation; compared as an exact string | Pass |
| 8. Something to keep | Well above chance, below ceiling | Base 95.7% on the exam | Pass |

## Model, task, data

- **Base:** `Qwen/Qwen3.5-35B-A3B` (Apache-2.0). Hybrid MoE: 40 layers (30 linear-attention, 10 full-attention), 256 routed experts plus a shared expert per layer, top-8 routing. fp16 size 71,903,655,008 B (from its `model.safetensors.index.json`), including the vision tower and MTP layer.
- **Task:** Language identification over 16 Latin-script languages that include look-alike pairs (Danish/Norwegian/Swedish, Indonesian/Malay, Spanish/Portuguese, Dutch/Afrikaans). The model writes the language name by free greedy generation.
- **In-distribution source:** FLORES-200 dev + devtest (CC-BY-SA-4.0, via mteb/flores): calibration 1,000, dev 1,600, locked exam 4,800 (300 per language), feedback pool 8,000 (disjoint).
- **Unseen source:** MASSIVE utterances (CC-BY-4.0): 4,800 locked exam items (300 per language), never used to build, tune or calibrate.

## Compression (no gradients anywhere, no data-fitted values)

1. **Text model only.** The vision tower and the MTP layer are dropped; the shipped config is the text-only `Qwen3_5MoeForCausalLM`.
2. **Expert selection.** Keep the 108 of 256 routed experts per layer with the most routing mass on the task's calibration prompts (`data/lid_calib.jsonl`), each followed by the base model's own answer (`fc/q35_route_stats.py`). Router rows are sliced to match. K = 108 is the largest uniform count that fits the size budget; it was fixed before any evaluation.
3. **Quantization** (`fc/q35_build.py`, `fc/qpack.py`): round-to-nearest, groups of 128, fp16 scale and minimum, with the clipping ratio picked per group from a fixed grid by weight reconstruction error. Routed experts 3-bit; attention, linear-attention projections, shared expert and embeddings 4-bit; LM head 6-bit. Routers, norms, convolutions and other small tensors stay bf16.
4. **Runtime.** `fc/q35_unpack.py` expands the artifact to a bf16 checkpoint once (a cache, not shipped); SGLang serves it with deterministic inference.

## Locked exam

`data/lid_exam.jsonl` (shipped gzipped as `data/locked/exam.jsonl.gz`) has sha256 `220e0e7a8b61356f319a0b587dcce31b306735ad0b278a95f47bc1a577b0d838`. It was opened once, after the build was fixed.

| Category | n | Base acc | Build acc | Kept of base-correct |
|---|---|---|---|---|
| **ALL** | 9,600 | 95.66% | 95.14% | **98.03% (9002 / 9183)** |
| source: flores | 4,800 | 97.94% | 98.27% | 99.26% (4666 / 4701) |
| source: massive | 4,800 | 93.37% | 92.00% | 96.74% (4336 / 4482) |
| flores/Afrikaans | 300 | 99.67% | 99.33% | 99.33% (297 / 299) |
| flores/Danish | 300 | 99.33% | 99.67% | 99.66% (297 / 298) |
| flores/Dutch | 300 | 100.00% | 100.00% | 100.00% (300 / 300) |
| flores/French | 300 | 100.00% | 100.00% | 100.00% (300 / 300) |
| flores/German | 300 | 100.00% | 100.00% | 100.00% (300 / 300) |
| flores/Icelandic | 300 | 99.67% | 100.00% | 100.00% (299 / 299) |
| flores/Indonesian | 300 | 98.67% | 96.33% | 97.64% (289 / 296) |
| flores/Italian | 300 | 100.00% | 100.00% | 100.00% (300 / 300) |
| flores/Malay | 300 | 71.33% | 81.67% | 93.46% (200 / 214) |
| flores/Norwegian | 300 | 99.67% | 99.33% | 99.33% (297 / 299) |
| flores/Polish | 300 | 99.67% | 100.00% | 100.00% (299 / 299) |
| flores/Portuguese | 300 | 99.33% | 97.00% | 97.65% (291 / 298) |
| flores/Romanian | 300 | 100.00% | 100.00% | 100.00% (300 / 300) |
| flores/Slovenian | 300 | 100.00% | 100.00% | 100.00% (300 / 300) |
| flores/Spanish | 300 | 99.67% | 99.00% | 99.33% (297 / 299) |
| flores/Swedish | 300 | 100.00% | 100.00% | 100.00% (300 / 300) |
| massive/Afrikaans | 300 | 82.00% | 88.67% | 98.78% (243 / 246) |
| massive/Danish | 300 | 96.00% | 93.33% | 96.53% (278 / 288) |
| massive/Dutch | 300 | 99.67% | 95.67% | 95.99% (287 / 299) |
| massive/French | 300 | 99.00% | 98.33% | 99.33% (295 / 297) |
| massive/German | 300 | 97.67% | 97.33% | 99.32% (291 / 293) |
| massive/Icelandic | 300 | 98.00% | 98.67% | 100.00% (294 / 294) |
| massive/Indonesian | 300 | 99.67% | 94.33% | 94.65% (283 / 299) |
| massive/Italian | 300 | 99.67% | 97.67% | 97.99% (293 / 299) |
| massive/Malay | 300 | 48.33% | 52.67% | 84.83% (123 / 145) |
| massive/Norwegian | 300 | 86.33% | 80.33% | 91.51% (237 / 259) |
| massive/Polish | 300 | 99.33% | 99.67% | 100.00% (298 / 298) |
| massive/Portuguese | 300 | 96.33% | 90.67% | 93.43% (270 / 289) |
| massive/Romanian | 300 | 98.00% | 94.33% | 95.92% (282 / 294) |
| massive/Slovenian | 300 | 96.67% | 97.67% | 98.28% (285 / 290) |
| massive/Spanish | 300 | 99.67% | 99.33% | 99.67% (298 / 299) |
| massive/Swedish | 300 | 97.67% | 93.33% | 95.22% (279 / 293) |

After the real feedback (24 experts switched off) the same exam keeps 97.44% (8948 / 9183).

## Feedback: switching existing experts off (no weight changes)

`python fc/q35_cli.py learn --artifact ART --in feedback.jsonl --guard guard.jsonl --out state.json` takes at most 25 annotated mistakes and writes only a list of `(layer, expert)` integers. `python fc/q35_cli.py answer --artifact ART --state state.json --in q.jsonl --out a.jsonl` runs the model with those experts switched off: the router never selects them (a fixed -1e4 router offset; every other offset is 0).

- Every weight tensor is byte-identical before and after feedback. No head, adapter, lookup memory, or prompt change.
- Experts are ranked by first-order attribution toward the annotated answers, switched off a few at a time with each step verified by real forward passes, and a switch is refused if it changes any of the model's own answers on a 160-item guard set.

**Feedback class.** Real mistakes of the compressed build on feedback-pool items it never calibrated on, for the gold label(s) where it errs most. 25 become feedback; the rest are locked probes (87), written to disk before any feedback ran. Nearby items are 300 pool items of the same and the confused labels that the build answers right. Files and sha256 are in `data/feedback/`.

| Arm | Locked probes right | Feedback items right | Nearby kept | Experts off |
|---|---|---|---|---|
| Empty | 0 / 87 (0.00%) | 0 / 25 (0.00%) | 300 / 300 (100.00%) | 0 |
| Shuffled labels | 0 / 87 (0.00%) | 0 / 25 (0.00%) | 229 / 300 (76.33%) | 24 |
| **Real** | **58 / 87 (66.67%)** | 19 / 25 (76.00%) | 298 / 300 (99.33%) | 24 |
| (Base model, for reference) | 25 / 87 (28.74%) | | 292 / 300 (97.33%) | |

**Rule-5 note.** The organizer rejected #3 and #4 because their feedback wrote values fitted to the annotations into weights or heads. This method writes no values: it chooses which of the model's own experts stay on. We think that is on the allowed side, but the choice is driven by the feedback, so we flag it as a gray zone.

## Disclosures

- Before the build we ran a small test of several tasks on this model (base accuracy, and kept share with 104 experts per layer and plain 3-bit rounding) on dev items and on a separate probe sample of the unseen source, and chose which tasks to build from those numbers. The locked exam items are disjoint from every item used there.
- Weights are not hosted yet; the artifact is rebuilt from the base checkpoint by the scripts below.

## Reproduce

Shipped data: `data/dev/*.jsonl.gz` (calibration, dev and test-probe sets), `data/locked/exam.jsonl.gz` (the locked exam), `data/feedback/` (feedback pool, feedback items, shuffled feedback, locked probes, nearby items, guard set, with `SHA256SUMS`). Results: `results/` (feedback states, exam reports, and the full output of the final run in `results/run_output.txt`).

All GPU work ran on one H100 (Modal). Scripts run with the repository root as working directory and `/data` mounted.

1. Data: `python3 fc/lid_sent_prep.py` (raw files under `data_raw/`). Shipped copies are in `data/`.
2. Base answers on the calibration prompts: `python3 fc/gen_sglang.py --model Qwen/Qwen3.5-35B-A3B --out base --max-new 6 --split data/lid_calib.jsonl`; routing mass: `python3 fc/q35_route_stats.py --model ... --items data/lid_calib.jsonl --answers base/... --out rs.pt`.
3. Build, exam, feedback and controls: `bash runs/q35_task.sh lid 108 6 data/lid_exam.jsonl data/lid_pool.jsonl`.
4. Commands: `fc/q35_cli.py answer` and `fc/q35_cli.py learn` (docstring at the top of the file).

## Licences

- Code: Apache-2.0 (see `LICENSE`). Base model: Apache-2.0.
- FLORES-200 CC-BY-SA-4.0. MASSIVE CC-BY-4.0. Exam and feedback files are derived from them with attribution.
