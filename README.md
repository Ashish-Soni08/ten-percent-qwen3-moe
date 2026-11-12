# Qwen3.5-35B-A3B at 9.72% of fp16 for sentiment, no training

This is an entry for the [10% challenge](https://github.com/DeusExperiments/ten-percent-challenge): review sentiment. **It passes the faithfulness rule (rule 2).** On the locked exam the compressed model keeps **98.96%** of the items the base model gets right, and at least 98.28% in every category and source.

## Result at a glance

| Rule | Requirement | Result | Status |
|---|---|---|---|
| 1. Size | ≤ 10% of fp16 (71,903,655,008 B) | Artifact (weights, tokenizer, config) 6,985,902,689 B, plus about 0.2 MB of code and the feedback state (a JSON list of integers): **9.716%** | Pass |
| 2. Faithful | Keep ≥ 98% of base-correct items, ≥ 95% in every category | **98.96%** overall (3228 / 3262); lowest category or source sst2 98.28% | **Pass** |
| 3. Exam | ≥ 2,000 locked items, ≥ 300 per category, ≥ 500 unseen | 3,372 locked items in 4 categories; 1,372 from unseen sources | Pass |
| 4. No new trained parts | No heads, adapters, memory, or prompt feedback | Experts are exact slices of the base, quantized post-training with no data-fitted values. Feedback state is a list of integers. | Pass (see the rule-5 note) |
| 5. Feedback teaches | ≤ 25 real mistakes; locked probes improve; nearby kept | Locked probes 0 / 86 (0.00%) → **12 / 86 (13.95%)**. Nearby items kept 300 / 300 (100.00%). | Pass |
| 6. Controls | Shuffled and empty clearly worse | Shuffled 0 / 86 (0.00%), empty 0 / 86 (0.00%), real 12 / 86 (13.95%) | Pass |
| 7. Real job | Real output, no multiple choice | The model writes the label itself by free greedy generation; compared as an exact string | Pass |
| 8. Something to keep | Well above chance, below ceiling | Base 96.7% on the exam | Pass |

## Model, task, data

- **Base:** `Qwen/Qwen3.5-35B-A3B` (Apache-2.0). Hybrid MoE: 40 layers (30 linear-attention, 10 full-attention), 256 routed experts plus a shared expert per layer, top-8 routing. fp16 size 71,903,655,008 B (from its `model.safetensors.index.json`), including the vision tower and MTP layer.
- **Task:** Binary review sentiment. The system prompt asks for `positive` or `negative`, and the model writes the label by free greedy generation, compared as an exact string after trimming and lowercasing.
- **In-distribution source:** amazon_polarity test split (Apache-2.0): calibration 1,000, dev 1,000, locked exam 2,000, feedback pool 6,000 (all disjoint).
- **Unseen source:** IMDB test (Maas et al., 2011) and SST-2 validation (GLUE): 1,372 locked exam items, never used to build, tune or calibrate.

## Compression (no gradients anywhere, no data-fitted values)

1. **Text model only.** The vision tower and the MTP layer are dropped; the shipped config is the text-only `Qwen3_5MoeForCausalLM`.
2. **Expert selection.** Keep the 108 of 256 routed experts per layer with the most routing mass on the task's calibration prompts (`data/sent_calib.jsonl`), each followed by the base model's own answer (`fc/q35_route_stats.py`). Router rows are sliced to match. K = 108 is the largest uniform count that fits the size budget; it was fixed before any evaluation.
3. **Quantization** (`fc/q35_build.py`, `fc/qpack.py`): round-to-nearest, groups of 128, fp16 scale and minimum, with the clipping ratio picked per group from a fixed grid by weight reconstruction error. Routed experts 3-bit; attention, linear-attention projections, shared expert and embeddings 4-bit; LM head 6-bit. Routers, norms, convolutions and other small tensors stay bf16.
4. **Runtime.** `fc/q35_unpack.py` expands the artifact to a bf16 checkpoint once (a cache, not shipped); SGLang serves it with deterministic inference.

## Locked exam

`data/sent_exam.jsonl` (shipped gzipped as `data/locked/exam.jsonl.gz`) has sha256 `729928cbb01ab8a6ad6dc06634b53692ed0c7d91c5dc185d42e4495ec6804fdd`. It was opened once, after the build was fixed.

| Category | n | Base acc | Build acc | Kept of base-correct |
|---|---|---|---|---|
| **ALL** | 3,372 | 96.74% | 96.12% | **98.96% (3228 / 3262)** |
| source: amazon | 2,000 | 97.20% | 96.80% | 99.33% (1931 / 1944) |
| amazon/negative | 978 | 98.06% | 97.65% | 99.27% (952 / 959) |
| amazon/positive | 1,022 | 96.38% | 95.99% | 99.39% (979 / 985) |
| imdb | 887 | 96.05% | 94.93% | 98.47% (839 / 852) |
| sst2 | 485 | 96.08% | 95.46% | 98.28% (458 / 466) |

After the real feedback (9 experts switched off) the same exam keeps 99.05% (3231 / 3262).

## Feedback: switching existing experts off (no weight changes)

`python fc/q35_cli.py learn --artifact ART --in feedback.jsonl --guard guard.jsonl --out state.json` takes at most 25 annotated mistakes and writes only a list of `(layer, expert)` integers. `python fc/q35_cli.py answer --artifact ART --state state.json --in q.jsonl --out a.jsonl` runs the model with those experts switched off: the router never selects them (a fixed -1e4 router offset; every other offset is 0).

- Every weight tensor is byte-identical before and after feedback. No head, adapter, lookup memory, or prompt change.
- Experts are ranked by first-order attribution toward the annotated answers, switched off a few at a time with each step verified by real forward passes, and a switch is refused if it changes any of the model's own answers on a 160-item guard set.

**Feedback class.** Real mistakes of the compressed build on feedback-pool items it never calibrated on, for the gold label(s) where it errs most. 25 become feedback; the rest are locked probes (86), written to disk before any feedback ran. Nearby items are 300 pool items of the same and the confused labels that the build answers right. Files and sha256 are in `data/feedback/`.

| Arm | Locked probes right | Feedback items right | Nearby kept | Experts off |
|---|---|---|---|---|
| Empty | 0 / 86 (0.00%) | 0 / 25 (0.00%) | 300 / 300 (100.00%) | 0 |
| Shuffled labels | 0 / 86 (0.00%) | 0 / 25 (0.00%) | 299 / 300 (99.67%) | 24 |
| **Real** | **12 / 86 (13.95%)** | 6 / 25 (24.00%) | 300 / 300 (100.00%) | 9 |
| (Base model, for reference) | 17 / 86 (19.77%) | | 299 / 300 (99.67%) | |

The probes are hard items: the uncompressed base model itself gets only 17 of 86 right, so the real feedback recovers most of the base model's level (12 of 86) while every nearby item and the locked exam stay intact (99.05% kept after feedback). As a sanity check of the mechanism, switching off 100 of the 108 experts in every layer makes the model emit garbage (`results/run_output.txt`, "switch check"). The whole pipeline was run twice from the same inputs and gave identical numbers and identical feedback-file hashes.

**Rule-5 note.** The organizer rejected #3 and #4 because their feedback wrote values fitted to the annotations into weights or heads. This method writes no values: it chooses which of the model's own experts stay on. We think that is on the allowed side, but the choice is driven by the feedback, so we flag it as a gray zone.

## Disclosures

- Before the build we ran a small test of several tasks on this model (base accuracy, and kept share with 104 experts per layer and plain 3-bit rounding) on dev items and on a separate probe sample of the unseen source, and chose which tasks to build from those numbers. The locked exam items are disjoint from every item used there.
- The packed artifact (6,985,902,689 B, byte-identical to the one evaluated) is hosted at https://huggingface.co/ashish-soni08/qwen35-35b-a3b-sentiment-10pct. Pass that folder as `--artifact`; it can also be rebuilt from the base checkpoint by the scripts below.

## Reproduce

Shipped data: `data/dev/*.jsonl.gz` (calibration, dev and test-probe sets), `data/locked/exam.jsonl.gz` (the locked exam), `data/feedback/` (feedback pool, feedback items, shuffled feedback, locked probes, nearby items, guard set, with `SHA256SUMS`). Results: `results/` (feedback states, exam reports, and the full output of the final run in `results/run_output.txt`).

All GPU work ran on one H100 (Modal). Scripts run with the repository root as working directory and `/data` mounted.

1. Data: `python3 fc/lid_sent_prep.py` (raw files under `data_raw/`). Shipped copies are in `data/`.
2. Base answers on the calibration prompts: `python3 fc/gen_sglang.py --model Qwen/Qwen3.5-35B-A3B --out base --max-new 4 --split data/sent_calib.jsonl`; routing mass: `python3 fc/q35_route_stats.py --model ... --items data/sent_calib.jsonl --answers base/... --out rs.pt`.
3. Build, exam, feedback and controls: `bash runs/q35_task.sh sent 108 4 data/sent_exam.jsonl data/sent_pool.jsonl`.
4. Commands: `fc/q35_cli.py answer` and `fc/q35_cli.py learn` (docstring at the top of the file).

## Licences

- Code: Apache-2.0 (see `LICENSE`). Base model: Apache-2.0.
- amazon_polarity: Apache-2.0. IMDB: released by Maas et al. (2011) for research use. SST-2: from GLUE (Socher et al., 2013). Exam and feedback files are derived from them with attribution.
