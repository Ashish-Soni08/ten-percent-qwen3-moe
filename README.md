# Qwen3-30B-A3B at 9.93% of fp16, no training: an honest non-qualifying entry to the 10% challenge

This is our entry for the [10% challenge](https://github.com/DeusExperiments/ten-percent-challenge). **It does not pass rule 2.** On the locked exam the compressed model keeps **94.54%** of the items the base model gets right. The rule needs 98%.

We publish it anyway because it is a complete, honest measurement:
- Three tasks.
- One locked exam, opened once.
- Feedback that changes no weights, with shuffled and empty controls.
- A record of what did and did not move the faithfulness number.

## Result at a glance

| Rule | Requirement | Result | Status |
|---|---|---|---|
| 1. Size | ≤ 10% of fp16 (61,064,245,248 B) | Artifact (weights, tokenizer, config) 6,065,976,857 B, plus code 0.4 MB: **9.934%** | Pass |
| 2. Faithful | Keep ≥ 98% of base-correct items, ≥ 95% in every category | **94.54%** overall (5,883 / 6,223). 7 of 11 categories at or above 95%. | **Fail** |
| 3. Exam | ≥ 2,000 locked items, ≥ 300 per category, ≥ 500 from an unseen source | 7,580 items. CLINC150 test (4,500, 10 domains × 450). Banking77 test (3,080), never used to build, tune or calibrate. | Pass |
| 4. No new trained parts | No heads, adapters, memory, or prompt feedback | Experts are exact slices of the base, quantized post-training. Feedback state is a list of integers. | Pass (see the note on rule 5) |
| 5. Feedback teaches | ≤ 25 annotated real mistakes; locked probes improve; nearby items kept | Locked probes **0% → 44.3%** (101 / 228). Nearby items kept **89.3%**. | Teaches, but breaks too much nearby |
| 6. Controls | Shuffled and empty feedback clearly worse | Shuffled **4.0%**, empty **0%**, against 44.3% for real | Pass |
| 7. Real job | Real output, no multiple choice | The model writes the intent name by free greedy generation, compared as an exact string | Pass |
| 8. Something to keep | Well above chance, below ceiling | Base 82.1% on the exam (chance about 1%) | Pass |

## Model, task, compression

- **Base:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE: 48 layers, 128 experts, top-8 routing). Apache-2.0.
- **Task:** intent routing for a virtual assistant.
  - The system prompt lists the allowed intents, and the model writes one intent name.
  - In-distribution source: CLINC150 (CC-BY-3.0), 150 intents. Unseen source: Banking77 (CC-BY-4.0), 77 intents with its own label list.
- **Compression (no gradients anywhere):**
  1. **Expert selection.** We kept 3,300 of 6,144 routed experts using the REAP saliency score (router weight × expert output norm). It was measured on 1,500 CLINC train prompts, using the base model's own answers, with a global budget and at least 36 experts per layer. The router is padded with a fixed -1e4 bias on dropped slots.
  2. **Mixed-precision GPTQ.** The 1,300 most-salient kept experts get 3-bit, the other 2,000 get 2-bit. Attention and embeddings are 4-bit, the LM head 6-bit. Groups of 128 with fp16 scale and min. Hessians come from the pruned bf16 model on the same calibration prompts.

## Locked exam (opened once, 11 Nov 2026 ~21:30 UTC)

`data/locked/int_exam.jsonl.gz` decompresses to a file with sha256 `739f3fd89e0ff94a2f60352150dd96064aa3d3d6530c5fe5b09032d94074dc67`.

| Category | n | Base acc | Build acc | Kept of base-correct |
|---|---|---|---|---|
| **ALL** | 7,580 | 82.10% | 79.22% | **94.54%** (5,883 / 6,223) |
| CLINC, all domains | 4,500 | 89.04% | 86.42% | 95.68% |
| Banking77 (unseen) | 3,080 | 71.95% | 68.70% | 92.46% |
| clinc / auto_and_commute | 450 | 85.33% | 80.22% | 92.97% |
| clinc / banking | 450 | 92.22% | 90.67% | 97.11% |
| clinc / credit_cards | 450 | 89.56% | 88.67% | 97.52% |
| clinc / home | 450 | 77.56% | 75.11% | 94.56% |
| clinc / kitchen_and_dining | 450 | 90.44% | 86.89% | 95.33% |
| clinc / meta | 450 | 88.44% | 85.11% | 94.22% |
| clinc / small_talk | 450 | 87.56% | 87.78% | 97.97% |
| clinc / travel | 450 | 94.44% | 87.33% | 90.82% |
| clinc / utility | 450 | 96.22% | 94.67% | 98.15% |
| clinc / work | 450 | 88.67% | 87.78% | 97.99% |

After real feedback, the same exam keeps 93.30%. Feedback costs another 1.2 points, mostly in `home` and `travel`. Full numbers are in `runs/int8_report.txt`.

## Feedback: switching existing experts off (no weight changes)

`python fc/cli.py learn --artifact ART --in feedback.jsonl --out state.json` takes at most 25 annotated mistakes and writes only a list of `(layer, expert)` integers. `python fc/cli.py answer --artifact ART --state state.json --in q.jsonl --out a.jsonl` runs the model with those experts switched off. The router never selects them; this uses a fixed -1e4 bias and no fitted value.

- Every weight tensor is byte-identical before and after feedback.
- There is no head, adapter, lookup memory, or prompt change.
- How experts are chosen:
  - Rank experts by first-order attribution toward the annotated answers.
  - Switch off a few at a time, verifying each step with real forward passes.
  - Refuse any switch that changes the model's own answers on a 160-item guard set.

**Feedback class.** These are real mistakes of the compressed build on CLINC train items it never calibrated on, for the three intents where it errs most: `reminder_update`, `distance` and `travel_notification`.
- 25 feedback items.
- 228 locked probes of the same kind.
- 300 nearby items (same and confused intents, which the build answers right).
- Files and sha256 are in `data/feedback/`.

| Arm | Locked probes right | Feedback items right | Nearby kept | Experts off |
|---|---|---|---|---|
| Empty | 0 / 228 (0%) | 0 / 25 | 300 / 300 | 0 |
| Shuffled labels | 9 / 228 (4.0%) | 0 / 25 | 283 / 300 (94.3%) | 4 |
| **Real** | **101 / 228 (44.3%)** | 14 / 25 | 268 / 300 (89.3%) | 25 |
| (Base model, for reference) | 55 / 228 (24.1%) | | 294 / 300 | |

The real feedback teaches well beyond the base (44% vs 24%), and both controls do clearly worse. But 25 switched-off experts break 10.7% of nearby items. The guard set protects far-away intents, not neighbours. We report this as measured and did not re-tune after opening the exam.

**Rule-5 note.** The organizer rejected #3 and #4 because their feedback wrote values fitted to the annotations into weights or heads. Our method writes no values: it chooses which of the model's own experts are on. We think that is on the allowed side of the line, but the choice is still driven by the feedback, so we flag it as a gray zone.

## What we tried, and why we stopped short of 98%

All GPU work ran on Modal H100s. Total spend was about $55; see `runs/SPEND.md`.

| Attempt | Size | Kept of base-correct |
|---|---|---|
| **Function calling** (xLAM + BFCL, locked 2,798-item exam) | | |
| All 128 experts, 4-bit RTN | 16 GB | 97.1% / 98.8% (dev: parallel / single) |
| 96 experts / layer, bf16 | 47 GB | 96.7% / 98.3% (dev) |
| rp56: reconstruction-based expert selection + GPTQ 3-bit | 6.07 GB | 91.9% / 95.4% (dev); **locked exam 92.3%** |
| **Safety moderation** (Aegis 2.0) | | |
| Recon prune + GPTQ | 6.07 GB | 87.4% (dev; base only 76% right) |
| **Intent routing** (CLINC150 dev, 1,500 items; base 90.3%) | | |
| Recon selection 2,690 experts, bf16 / + GPTQ 3-bit | 28 GB / 6.07 GB | 95.5% / 94.8% |
| REAP-saliency selection 2,690 / 3,000 / 3,300 / 3,600 experts, bf16 | 25-34 GB | 97.3% / 98.45% / **98.67%** / 98.45% |
| 3,300 experts, 1,300 at 3-bit + 2,000 at 2-bit, GPTQ | 6.07 GB | **96.45%** → final build |
| 3,000 experts, 2,000 at 3-bit + 1,000 at 2-bit, GPTQ | 6.08 GB | 95.35% |
| CLINC 10 coarse domains, same build | 6.07 GB | 91.8% (base only 90.7% right) |
| **Qwen3.5-35B-A3B** (256 finer experts): 108 per layer by routing mass, RTN 3/4/6-bit, same locked exam | 6.99 GB (9.72%) | **locked exam 93.45%** (Banking77 91.5%); feedback probes 0 → 44/70 but nearby kept only 54% |

**What we learned:**
- With 128 coarse experts, about 3,300 experts are needed just to reach 98.5% before quantization. Fitting that into 6.1 GB forces many 2-bit experts, which costs about 2 points.
- The faithfulness loss is concentrated where the base is unsure: near-synonym intents, and the unseen Banking77 labels.
- Distillation, LoRA, or QAT recovery would likely close the gap, but rule 2 forbids training.
- Another entrant's backup reports 99.5% on Qwen3.5-35B-A3B, which has 256 finer experts. We tried it on 12 Nov with the same locked exam: it kept 93.45%, slightly worse than this Qwen3-30B build, so this remains our intent-routing entry. (On review sentiment the same Qwen3.5 recipe keeps 98.96%; see the other entry PRs in this repo.)

## Reproduce

Weights are not hosted. The artifact lives on our Modal volume. Rebuild it from the base checkpoint as follows (scripts run inside the SGLang image with `/data` mounted; `tools/modal_judge.sh` submits them):

1. **Data:** `python3 fc/intent_prep.py`, `fc/intent_exam.py`, `fc/select_int_fb.py`. Raw CLINC150 and Banking77 files go in `data_raw/intent/`.
2. **Base answers and saliency:** `runs/int2.sh` (first lines) and `runs/int3.sh`.
3. **Build:** `runs/int5.sh` produces `/data/fc/art/int3300` (6,065,976,857 B) and the dev check.
4. **Feedback, controls and the locked exam:** `runs/int8.sh`. Output is `runs/int8_report.txt`.
5. **Commands:** `fc/cli.py answer` and `fc/cli.py learn` (docstring at the top of the file).

The function-calling and safety pipelines are in `runs/stage*.sh` and `runs/saf*.sh`, with code in `fc/`. One item of the function-calling guard set (`fc/guard.jsonl`) was removed before publishing because its dataset text contains a token-shaped string; the published guard has 153 items, and the runs used 154.

## Licences

- Code: Apache-2.0 (see `LICENSE`).
- Base model: Apache-2.0.
- CLINC150 is CC-BY-3.0 (Larson et al., 2019) and Banking77 is CC-BY-4.0 (Casanueva et al., 2020). Exam and feedback files are derived from them with attribution.
