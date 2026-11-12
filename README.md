# 10% challenge attempt: Qwen3-30B-A3B for function calling

An attempt at the [10% challenge](https://github.com/DeusExperiments/ten-percent-challenge). Start from
Qwen3-30B-A3B-Instruct-2507 (fp16 61.1 GB), ship 6.07 GB (9.94%), keep its function-calling behaviour, and learn from
up to 25 annotated items without training any weights.

**Result: it does not pass rule 2 (faithfulness).** On the locked exam, the compressed model keeps 92.3% of the items
the base got right; the rule needs 98% overall and 95% in every category. Everything else is in place: size, locked
exam with an unseen source, real outputs, and feedback that teaches with working shuffled and empty controls.

## Task and exam

The model receives a user request and tool definitions (OpenAI-style JSON schema, rendered by Qwen's chat template) and
must emit the right `<tool_call>` blocks: function names and argument values, one call or several in parallel.

| split | source | items | use |
|---|---|---|---|
| calibration | xLAM-60k (CC-BY-4.0) + ToolACE (Apache-2.0) prompts | 1,500 + 1,000 | expert selection, GPTQ |
| dev | xLAM + ToolACE | 800 + 600 | choosing the build |
| **locked exam** | xLAM (600 single, 600 parallel) + **BFCL v3 (unseen source)**: 658 simple, 500 multiple, 440 parallel | **2,798** | run once at the end |

`data/exam.jsonl` sha256 `0f80fd8952ffb680bc22f0e157607a1b4a43cf6479cb7e2dd12729f1669df428`. BFCL was never used to
build, tune or calibrate. Scoring (`fc/score.py`): xLAM compares calls one to one (arguments that only one side has
must equal the tool default); BFCL uses an AST-style check against its possible answers.

## Compression (no weight training)

1. **Expert selection by reconstruction** (`fc/recon_prune.py`): per layer, greedily drop the expert whose removal
   (router re-picking top-8 among the rest) least increases the MoE output error on calibration tokens; split a global
   budget of 2,690 of 6,144 experts across layers by marginal error (24 to 72 experts per layer).
2. **GPTQ** (`fc/gptq_compress.py`): experts 3-bit, attention 4-bit (group 128, fp16 scale and min), with Hessians from
   the pruned model. Embeddings 4-bit and LM head 6-bit, round-to-nearest with MSE clipping.
3. Packed artifact (`fc/compress.py` format, `fc/unpack.py` loader): 6,072,953,023 bytes plus code and the feedback state.
   At run time it unpacks to bf16 and is served by SGLang. A small patch (`fc/sglpatch`) gives the router a per-expert
   offset, used for padding slots and for feedback.

## Feedback: expert switching (no weight change)

`python fc/cli.py learn --artifact ART --in feedback.jsonl --out state.json` finds experts to switch off so the
annotated answers become the model's own greedy answers. Candidates are ranked by first-order attribution, every switch
is checked with a real forward pass, and a switch is refused if it changes any answer on `guard.jsonl` (154 questions
the model already gets right). The stored state is a list of (layer, slot) pairs. The answer command applies it as a
router offset of -1e4.

The feedback taught: *don't add optional arguments I didn't ask for*, the shipped model's most common systematic
mistake. Annotations are the model's own answers with the unrequested arguments removed. Tools are disjoint between
feedback items and probes.

A second method was built and dropped: closed-form MEMIT/AlphaEdit-style edits to attention output projections. It
fixed only 3 of 50 probes and hurt multi-call answers (`fc/feedback.py: learn_edit`).

## Results

Locked exam, run once (`runs/stage7.sh`, `fc/final_report.py`):

| category | base acc | compressed acc | kept of base-correct |
|---|---|---|---|
| xlam_single | 87.8% | 86.0% | 95.3% |
| xlam_parallel | 78.7% | 77.3% | 94.1% |
| bfcl_simple (unseen) | 90.6% | 85.9% | 90.9% |
| bfcl_multiple (unseen) | 83.0% | 77.8% | 90.1% |
| bfcl_parallel (unseen) | 88.9% | 83.2% | 90.8% |
| **all** | **85.8%** | **82.2%** | **92.3%** (rule: 98%) |

Feedback, 25 items, 50 locked probes picked from the final model's own mistakes (`data/fb2`, `runs/stage8b.sh`):

| feedback | experts switched off | locked probes right | nearby items kept (293) |
|---|---|---|---|
| empty | 0 | 0/50 | 293 |
| shuffled labels | 0 | 0/50 | 293 |
| **real** | 12 | **13/50** | 286 |

## How far compression got (xLAM dev, share of base-correct kept)

| build | size | parallel | single |
|---|---|---|---|
| all experts, 4-bit RTN | 16.3 GB | 97.1% | 98.8% |
| 96 experts/layer, bf16 | 46.6 GB | 96.7% | 98.3% |
| 64 experts/layer, bf16 | 32.1 GB | 95.1% | 96.5% |
| 56 experts, saliency, 3-bit RTN | 6.07 GB | 86.6% | 91.6% |
| same + GPTQ | 6.07 GB | 89.9% | 93.1% |
| **reconstruction selection + GPTQ (final)** | 6.07 GB | **91.9%** | **95.4%** |

Even pruning a quarter of the experts in bf16 misses 98%. For this model and task, the faithfulness bar does not seem
reachable at 10% without training.

## Reproduce

`runs/*.sh` are the exact scripts, run on one H100 through `tools/modal_judge.sh`. `runs/SPEND.md` logs GPU cost.

## Notes on the published files

- Data files are gzipped (`data/exam.jsonl.gz` decompresses to the file whose sha256 is given above).
- Some xLAM items contain a string shaped like a GitHub token (dataset text, not a real credential). GitHub push
  protection blocks it, so it is replaced with `ghp_REDACTED` in the published copies of `data/calib`, `data/dev`,
  `data/fb/*` and `fc/guard.jsonl`. The locked exam is unaffected. The runs used the original text.
- Licences: code Apache-2.0; xLAM-60k CC-BY-4.0; ToolACE Apache-2.0; BFCL v3 Apache-2.0.
