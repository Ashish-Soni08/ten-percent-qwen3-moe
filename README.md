# ten-percent-qwen3-moe

Our entries to the [10% challenge](https://github.com/DeusExperiments/ten-percent-challenge): ship a model at no more than 10% of its fp16 size without training its weights, match the base on a locked exam, and let it learn from up to 25 annotated mistakes. Every entry here uses the same idea. Keep the routed experts a task needs, quantize with round-to-nearest or GPTQ, and learn from feedback by switching existing experts off with a fixed router offset. No weight value ever changes.

## Outcome (challenge closed 17 Nov 2026)

**No entry won.** The maintainer reviewed all eleven submissions and found that none passed every rule and the maintainer's check, so the prize went unclaimed ([challenge README](https://github.com/DeusExperiments/ten-percent-challenge#outcome)).

**Our sentiment entry came closest.** The maintainer said it "came closer than any other submission here." They downloaded the hosted weights and reran the full pipeline:

- **Locked exam confirmed.** The rerun kept 99.02% of base-correct items before feedback and 99.17% after. Rules 1 to 4 pass.
- **Failed the held-out feedback check (rule 6).**
  - The maintainer used 20,000 held-out Yelp reviews, where the compressed model made 431 mistakes.
  - Feedback used 25 of those mistakes as they came (18 positive, 7 negative), leaving 406 unseen ones to test on.
  - Real labels fixed 77 of the 406, but flipped labels on the same 25 items still fixed 52.
  - So the expert switch mostly tilts answers toward one label rather than learning from the true labels. The wrong-label control was not clearly worse, as rule 6 requires.

The other four posted entries were closed as non-qualifying under rule 2 (faithfulness), as each one stated up front.

| Task | Base model | Entry PR | Challenge issue | Locked exam kept | Result |
|---|---|---|---|---|---|
| Review sentiment | Qwen3.5-35B-A3B | [#5](https://github.com/Ashish-Soni08/ten-percent-qwen3-moe/pull/5) | [#7](https://github.com/DeusExperiments/ten-percent-challenge/issues/7) | 98.96% (maintainer rerun 99.02%) | Passed rules 1 to 4, failed rule 6 on held-out Yelp |
| Intent routing (CLINC150 + Banking77) | Qwen3-30B-A3B | [#1](https://github.com/Ashish-Soni08/ten-percent-qwen3-moe/pull/1) | [#8](https://github.com/DeusExperiments/ten-percent-challenge/issues/8) | 94.54% | Non-qualifying, rule 2 |
| Safety moderation | Qwen3.5-35B-A3B | [#4](https://github.com/Ashish-Soni08/ten-percent-qwen3-moe/pull/4) | [#9](https://github.com/DeusExperiments/ten-percent-challenge/issues/9) | 95.04% | Non-qualifying, rule 2 |
| Language ID | Qwen3.5-35B-A3B | [#3](https://github.com/Ashish-Soni08/ten-percent-qwen3-moe/pull/3) | [#10](https://github.com/DeusExperiments/ten-percent-challenge/issues/10) | 98.03% (5 of 32 categories under 95%) | Non-qualifying, rule 2 |
| Function calling | Qwen3-30B-A3B | [#2](https://github.com/Ashish-Soni08/ten-percent-qwen3-moe/pull/2) | [#11](https://github.com/DeusExperiments/ten-percent-challenge/issues/11) | 92.3% | Non-qualifying, rule 2 |
| Prompt injection | Qwen3.5-35B-A3B | [#6](https://github.com/Ashish-Soni08/ten-percent-qwen3-moe/pull/6) | not submitted | 88.24% (95.78% after feedback) | Not submitted |

Hosted weights (sentiment only): https://huggingface.co/ashish-soni08/qwen35-35b-a3b-sentiment-10pct

## What we learned

- **Compression is solved for short-label tasks.** Keeping 108 of 256 experts per layer, with 3-bit round-to-nearest experts, reaches about 99% faithfulness on sentiment.
- **Feedback is the open problem.** Switching experts off lifted our own locked probes from 0/86 to 12/86, against 0/86 for shuffled labels. On the maintainer's held-out data, it mostly shifted the label prior. It did not learn from the true labels over wrong ones.
- **The controls need to be stronger.** A future attempt should test the feedback against flipped labels on fresh held-out mistakes, as the maintainer did, not only against shuffled labels. It should also balance the 25 feedback items across labels.

Each entry's code, locked files and full write-up live on its PR branch.
