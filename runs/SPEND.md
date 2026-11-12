# Modal spend log (1.4 x seconds x hourly rate)
| when (UTC) | run | runner | seconds | est USD |
|---|---|---|---|---|
| 11-10 16:15 | smoke | cpu | 5 | 0.00 |
| 11-10 16:16 | model download | cpu | 195 | 0.02 |
| 11-10 16:19 | data prep | cpu | 25 | 0.00 |
| 11-10 16:20 | stage1 attempt (crashed) | h100 | 88 | 0.14 |
| 11-10 16:23 | sglang debug x2 | h100 | 35+~260 | 0.45 |
| 11-10 16:29 | stage1 base gen (ok) + REAP stats (timed out, too slow) | h100 | 3302 | 5.07 |
| 11-10 17:26 | REAP stats batched | h100 | ~330 | 0.51 |
| 11-10 17:31 | stage2 sweep (2 parallel calls) | h100 | 1018+823 | 2.83 |
| 11-10 17:52 | stage2 sweep 2 (2 parallel calls) | h100 | 863+933 | 2.76 |
| 11-10 18:06 | stage3a build g56q3 + mine 18k | h100 | 452 | 0.69 |
| 11-10 18:15 | stage3b feedback tune (bias bug) | h100 | 303 | 0.47 |
| 11-10 18:22 | stage3b feedback tune | h100 | 628 | 0.96 |
| 11-10 18:33 | stage3c feedback tune 2 | h100 | 897 | 1.38 |
| 11-10 18:49 | stage3d locked feedback comparison | h100 | 1515 | 2.33 |
| 11-10 19:23 | stage4 GPTQ g56 | h100 | 950 | 1.46 |
| 11-10 19:15 | stage3e switching rerun + sglang confirm | h100 | 2284 | 3.51 |
| 11-10 19:41 | stage5 recon-prune + GPTQ (rp56) | h100 | 1844 | 2.83 |
| 11-10 20:11 | stage6 recon-prune 16k tok, minper 12 | h100 | 2008 | 3.08 |
| 11-10 20:45 | stage6b GPTQ rp56b (worse, dropped) | h100 | 1020 | 1.57 |
| 11-10 21:03 | stage7 final locked exam + feedback (rp56) | h100 | 1769 | 2.72 |
| 11-10 21:38 | stage8a mine rp56 | h100 | 400 | 0.61 |
| 11-10 21:42 | stage8b fresh-probe feedback + controls | h100 | 2324 | 3.57 |
| 11-11 14:19 | safety probe base + rp56 | h100 | 429 | 0.66 |
| 11-11 16:00 | safety base gen + recon prune | h100 | 1457 | 2.24 |
| 11-11 16:30 | safety GPTQ rebuild + eval | h100 | 932.4 | 1.43 |
