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
| 11-11 17:47 | intent base check (4 sets) + rp56 | h100 | 433 | 0.67 |
| 11-11 18:32 | CLINC base + recon prune + GPTQ int56 + dev | h100 | 2148 | 3.30 |
| 11-11 18:57 | CLINC reap stats + 3 prune-only selections | h100 | 878.5 | 1.35 |
| 11-11 19:43 | step1 prune-only 3000/3300/3600 | h100 | 821 | 1.26 |
| 11-11 20:01 | step2 int3300 mixed 2/3-bit GPTQ + dev | h100 | 886 | 1.36 |
| 11-11 20:33 | CLINC 10-domain test on int3300 | h100 | 424 | 0.65 |
| 11-11 20:52 | int3000 mixed build + dev | h100 | 825 | 1.27 |
| 11-11 20:57 | pool gen base + int3300 | h100 | 256 | 0.39 |
| 11-11 21:33 | final: learn real/shuf, probes, locked exam base/empty/real | h100 | 2088 | 3.21 |
| 11-11 22:56 | q35a download + base gen CLINC/FC | h100 | 1654.5 | 2.54 |
| 11-11 23:10 | q35b PI base + route stats + k104 RTN3 slice + dev clinc/pi/fc | h100 | 921.4 | 1.42 |
| 11-11 23:30 | q35c LID + sentiment base + k104 dev/unseen | h100 | 945.2 | 1.45 |
| 11-12 06:04 | q35p code probe | h100 | 45.8 | 0.07 |
| 11-12 06:14 | q35s smoke: sent K108 build + load + FB test | h100 | 521.1 | 0.80 |
| 11-12 07:15 | f_saf try1 (timed out at 3600 s; partial outputs kept) | h100 | 3600 | 5.53 |
| 11-12 08:15 | f_sent try1 (timed out at 3600 s; partial outputs kept) | h100 | 3600 | 5.53 |
| 11-12 08:17 | f_pi try1 (timed out at 3600 s; partial outputs kept) | h100 | 3600 | 5.53 |
| 11-12 08:33 | f_saf resume (safety Qwen3.5 complete) | h100 | 388.3 | 0.60 |
| 11-12 08:26 | f_lid try1 (base OOM: shared GPU with another job) | h100 | 685.1 | 1.05 |
| 11-12 08:41 | f_sent resume (sentiment complete) | h100 | 481.6 | 0.74 |
| 11-12 08:54 | f_clinc full (Qwen3.5 intent routing) | h100 | 2247.0 | 3.45 |
| 11-12 09:24 | f_lid full (language ID) | h100 | 1791.0 | 2.75 |
| 11-12 09:41 | f_pi resume try2 (timed out; shared GPU, no lock) | h100 | 3600 | 5.53 |
| 11-12 10:12 | f_sent rerun (reproduced identical numbers, switch check) | h100 | 2680.6 | 4.12 |
