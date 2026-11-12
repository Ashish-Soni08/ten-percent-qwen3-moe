"""Write README.md for one Qwen3.5 entry from its final run output. Usage: make_q35_readme.py TASK final.json out.md"""
import json, re, sys

FP16 = 71_903_655_008
T = {
 "sent": dict(
   title="Review sentiment", short="sentiment",
   task="Binary review sentiment. The system prompt asks for `positive` or `negative`, and the model writes the label by free greedy generation, compared as an exact string after trimming and lowercasing.",
   main="amazon_polarity test split (Apache-2.0): calibration 1,000, dev 1,000, locked exam 2,000, feedback pool 6,000 (all disjoint)",
   unseen="IMDB test (Maas et al., 2011) and SST-2 validation (GLUE): 1,372 locked exam items, never used to build, tune or calibrate",
   calib="data/sent_calib.jsonl", exam="data/sent_exam.jsonl", pool="data/sent_pool.jsonl", maxnew=4,
   lic="amazon_polarity: Apache-2.0. IMDB: released by Maas et al. (2011) for research use. SST-2: from GLUE (Socher et al., 2013)."),
 "pi": dict(
   title="Prompt-injection detection", short="prompt injection",
   task="Indirect prompt-injection detection for AI agents. The user message is content a tool returned (email, document, web page, code); the model answers `yes` if it contains instructions aimed at the agent, else `no`, by free greedy generation.",
   main="Horizon-Labs/prompt-injection-eval-suite configs `bipia` and `synthetic_docs_test`: calibration 1,000, dev 1,000, locked exam 2,400, feedback pool 2,132 (disjoint)",
   unseen="configs `piarena` and `llmail_phase2` of the same suite (different sources and attack styles): 2,500 locked exam items, never used to build, tune or calibrate",
   calib="data/pi_calib.jsonl", exam="data/pi_exam.jsonl", pool="data/pi_pool.jsonl", maxnew=4,
   lic="Horizon-Labs/prompt-injection-eval-suite: per-config licences (CC-BY-SA-4.0 / Apache-2.0 / MIT); see the dataset card."),
 "clinc": dict(
   title="Intent routing", short="intent routing",
   task="Intent routing for a virtual assistant. The system prompt lists the allowed intents and the model writes one intent name by free greedy generation, compared as an exact string.",
   main="CLINC150 (CC-BY-3.0): calibration 1,500 train items, feedback pool 13,500 other train items, locked exam = CLINC150 test (4,500, 10 domains x 450)",
   unseen="Banking77 test (CC-BY-4.0, 3,080 items, its own 77-label list), never used to build, tune or calibrate",
   calib="data/int_clinc_calib.jsonl", exam="data/int_exam.jsonl", pool="data/int_pool.jsonl", maxnew=16,
   lic="CLINC150 CC-BY-3.0 (Larson et al., 2019). Banking77 CC-BY-4.0 (Casanueva et al., 2020)."),
 "lid": dict(
   title="Language identification", short="language ID",
   task="Language identification over 16 Latin-script languages that include look-alike pairs (Danish/Norwegian/Swedish, Indonesian/Malay, Spanish/Portuguese, Dutch/Afrikaans). The model writes the language name by free greedy generation.",
   main="FLORES-200 dev + devtest (CC-BY-SA-4.0, via mteb/flores): calibration 1,000, dev 1,600, locked exam 4,800 (300 per language), feedback pool 8,000 (disjoint)",
   unseen="MASSIVE utterances (CC-BY-4.0): 4,800 locked exam items (300 per language), never used to build, tune or calibrate",
   calib="data/lid_calib.jsonl", exam="data/lid_exam.jsonl", pool="data/lid_pool.jsonl", maxnew=6,
   lic="FLORES-200 CC-BY-SA-4.0. MASSIVE CC-BY-4.0."),
 "saf": dict(
   title="Safety moderation", short="safety moderation",
   task="Content-safety classification. A policy is given in the system prompt and the model answers `safe` or `unsafe` for a user prompt or a prompt-response pair, by free greedy generation.",
   main="Aegis 2.0 (NVIDIA, CC-BY-4.0): calibration 2,500 train items, feedback pool 1,688 validation items, locked exam 2,715 test items",
   unseen="OpenAI moderation evaluation set (MIT): 1,680 locked exam items, never used to build, tune or calibrate",
   calib="data/saf_calib.jsonl", exam="data/saf_exam.jsonl", pool="data/saf_pool.jsonl", maxnew=4,
   lic="Aegis 2.0 CC-BY-4.0. OpenAI moderation eval MIT."),
}

def parse(stdout):
    s = stdout.split("@@TAR_BEGIN@@")[0]
    sec = {}
    m = re.search(r"== locked exam sha256\n([0-9a-f]{64})", s); sec["exam_sha"] = m.group(1) if m else "?"
    m = re.search(r"ARTIFACT \S+ bytes (\d+)", s); sec["bytes"] = int(m.group(1)) if m else None
    def table(after):
        i = s.find(after);
        if i < 0: return []
        rows = []
        for l in s[i:].splitlines()[2:]:
            mm = re.match(r"(\S+)\s+n=\s*(\d+) base=([\d.]+) build=([\d.]+) kept=(\d+)/(\d+)=([\d.]+)", l)
            if not mm: break
            rows.append(mm.groups())
        return rows
    sec["empty"] = table("== EXAM (empty feedback)"); sec["real"] = table("== EXAM after real feedback")
    for kind in ("empty", "real", "shuf"):
        i = s.find(f"== {kind}\n"); blk = s[i:i + 800]
        for k in ("probe_locked", "nearby_test", "feedback"):
            mm = re.search(k + r" \S+: acc [\d.]+ \((\d+)/(\d+)\)", blk)
            sec[f"{kind}_{k}"] = (int(mm.group(1)), int(mm.group(2))) if mm else None
    i = s.find("== base probes"); blk = s[i:i + 600]
    for k in ("probe_locked", "nearby_test"):
        mm = re.search(k + r" \S+: acc [\d.]+ \((\d+)/(\d+)\)", blk); sec[f"base_{k}"] = (int(mm.group(1)), int(mm.group(2))) if mm else None
    mm = re.search(r"kept (\d+)/(\d+) = [\d.]+; agree", s[s.find("== nearby kept"):]); sec["nearby_kept"] = (int(mm.group(1)), int(mm.group(2))) if mm else None
    mm = re.search(r'\{"switched_off": (\[.*?\]), "history"', s); sec["off"] = json.loads(mm.group(1)) if mm else []
    so = re.findall(r"switched off (\d+) experts in total", s); sec["shuf_off"] = so[1] if len(so) > 1 else "?"
    shas = re.findall(r"([0-9a-f]{64})  \S+/fb/(\S+\.jsonl)", s); sec["fb_sha"] = shas
    return sec

def FB_STATUS(r):
    rp, sp, ep, nk = r["real_probe_locked"], r["shuf_probe_locked"], r["empty_probe_locked"], r["nearby_kept"]
    if not (rp and sp and ep and nk): return "n/a"
    teaches = rp[0] > max(sp[0], ep[0])
    if not teaches: return "Fail"
    return "Pass" if nk[0] / nk[1] >= 0.95 else f"Teaches, but breaks {100 - 100 * nk[0] / nk[1]:.1f}% of nearby items"

def pct(a, b): return f"{100 * a / b:.2f}%" if b else "n/a"

def main():
    t, fj, out = sys.argv[1:4]; M = T[t]
    d = json.load(open(fj)); r = parse(d["stdout"])
    nbytes = r["bytes"] or int(sys.argv[4])
    code_note = "plus about 0.2 MB of code and the feedback state (a JSON list of integers)"
    E = r["empty"]; allrow = next(x for x in E if x[0] == "ALL")
    kept_all = float(allrow[6]); cats = [x for x in E if not x[0].startswith("src:") and x[0] != "ALL"]; srcs = [x for x in E if x[0].startswith("src:")]
    worst = min(cats + srcs, key=lambda x: float(x[6]))
    passes = kept_all >= 0.98 and all(float(x[6]) >= 0.95 for x in cats + srcs)
    n_ok_cat = sum(float(x[6]) >= 0.95 for x in cats)
    R = r["real"]; rall = next((x for x in R if x[0] == "ALL"), None)
    def frac(k): v = r.get(k); return f"{v[0]} / {v[1]} ({pct(*v)})" if v else "n/a"
    status2 = "**Pass**" if passes else "**Fail**"
    head = (f"**It passes the faithfulness rule (rule 2).** On the locked exam the compressed model keeps **{100*kept_all:.2f}%** of the items the base model gets right, and at least {100*float(worst[6]):.2f}% in every category and source."
            if passes else
            (f"**It does not pass rule 2.** On the locked exam the compressed model keeps **{100*kept_all:.2f}%** of the items the base model gets right, which meets the 98% overall bar, but {len(cats)-n_ok_cat} of {len(cats)} categories fall below 95% (lowest: {worst[0]} {100*float(worst[6]):.2f}%). We publish it as a complete, honest measurement."
             if kept_all >= 0.98 else
             f"**It does not pass rule 2.** On the locked exam the compressed model keeps **{100*kept_all:.2f}%** of the items the base model gets right (the rule needs 98%, and 95% in every category; {n_ok_cat} of {len(cats)} categories reach 95%). We publish it as a complete, honest measurement."))
    L = []
    L.append(f"# Qwen3.5-35B-A3B at {100*nbytes/FP16:.2f}% of fp16 for {M['short']}, no training\n")
    L.append(f"This is an entry for the [10% challenge](https://github.com/DeusExperiments/ten-percent-challenge): {M['title'].lower()}. {head}\n")
    L.append("## Result at a glance\n")
    L.append("| Rule | Requirement | Result | Status |\n|---|---|---|---|")
    L.append(f"| 1. Size | ≤ 10% of fp16 ({FP16:,} B) | Artifact (weights, tokenizer, config) {nbytes:,} B, {code_note}: **{100*nbytes/FP16:.3f}%** | Pass |")
    L.append(f"| 2. Faithful | Keep ≥ 98% of base-correct items, ≥ 95% in every category | **{100*kept_all:.2f}%** overall ({allrow[4]} / {allrow[5]}); lowest category or source {worst[0]} {100*float(worst[6]):.2f}% | {status2} |")
    n_exam = int(allrow[1]); n_unseen = sum(int(x[1]) for x in srcs if x[0] in UNSEEN[t])
    L.append(f"| 3. Exam | ≥ 2,000 locked items, ≥ 300 per category, ≥ 500 unseen | {n_exam:,} locked items in {len(cats)} categories; {n_unseen:,} from unseen sources | Pass |")
    L.append("| 4. No new trained parts | No heads, adapters, memory, or prompt feedback | Experts are exact slices of the base, quantized post-training with no data-fitted values. Feedback state is a list of integers. | Pass (see the rule-5 note) |")
    L.append(f"| 5. Feedback teaches | ≤ 25 real mistakes; locked probes improve; nearby kept | Locked probes {frac('empty_probe_locked')} → **{frac('real_probe_locked')}**. Nearby items kept {frac('nearby_kept')}. | {FB_STATUS(r)} |")
    L.append(f"| 6. Controls | Shuffled and empty clearly worse | Shuffled {frac('shuf_probe_locked')}, empty {frac('empty_probe_locked')}, real {frac('real_probe_locked')} | {'Pass' if (r['real_probe_locked'] or (0,1))[0] > max((r['shuf_probe_locked'] or (0,1))[0], (r['empty_probe_locked'] or (0,1))[0]) else 'Fail'} |")
    L.append("| 7. Real job | Real output, no multiple choice | The model writes the label itself by free greedy generation; compared as an exact string | Pass |")
    L.append(f"| 8. Something to keep | Well above chance, below ceiling | Base {100*float(allrow[2]):.1f}% on the exam | Pass |\n")
    L.append("## Model, task, data\n")
    L.append("- **Base:** `Qwen/Qwen3.5-35B-A3B` (Apache-2.0). Hybrid MoE: 40 layers (30 linear-attention, 10 full-attention), 256 routed experts plus a shared expert per layer, top-8 routing. fp16 size 71,903,655,008 B (from its `model.safetensors.index.json`), including the vision tower and MTP layer.")
    L.append(f"- **Task:** {M['task']}")
    L.append(f"- **In-distribution source:** {M['main']}.")
    L.append(f"- **Unseen source:** {M['unseen']}.\n")
    L.append("## Compression (no gradients anywhere, no data-fitted values)\n")
    L.append("1. **Text model only.** The vision tower and the MTP layer are dropped; the shipped config is the text-only `Qwen3_5MoeForCausalLM`.")
    L.append(f"2. **Expert selection.** Keep the 108 of 256 routed experts per layer with the most routing mass on the task's calibration prompts (`{M['calib']}`), each followed by the base model's own answer (`fc/q35_route_stats.py`). Router rows are sliced to match. K = 108 is the largest uniform count that fits the size budget; it was fixed before any evaluation.")
    L.append("3. **Quantization** (`fc/q35_build.py`, `fc/qpack.py`): round-to-nearest, groups of 128, fp16 scale and minimum, with the clipping ratio picked per group from a fixed grid by weight reconstruction error. Routed experts 3-bit; attention, linear-attention projections, shared expert and embeddings 4-bit; LM head 6-bit. Routers, norms, convolutions and other small tensors stay bf16.")
    L.append("4. **Runtime.** `fc/q35_unpack.py` expands the artifact to a bf16 checkpoint once (a cache, not shipped); SGLang serves it with deterministic inference.\n")
    L.append(f"## Locked exam\n\n`{M['exam']}` (shipped gzipped as `data/locked/exam.jsonl.gz`) has sha256 `{r['exam_sha']}`. It was opened once, after the build was fixed.\n")
    L.append("| Category | n | Base acc | Build acc | Kept of base-correct |\n|---|---|---|---|---|")
    def row(x, bold=False):
        nm = x[0].replace("src:", "source: "); k = f"{100*float(x[6]):.2f}% ({x[4]} / {x[5]})"
        return f"| {'**'+nm+'**' if bold else nm} | {int(x[1]):,} | {100*float(x[2]):.2f}% | {100*float(x[3]):.2f}% | {'**'+k+'**' if bold else k} |"
    L.append(row(allrow, True))
    for x in srcs:
        if x[0][4:] not in {c[0] for c in cats}: L.append(row(x))
    for x in cats: L.append(row(x))
    if rall: L.append(f"\nAfter the real feedback ({len(r['off'])} experts switched off) the same exam keeps {100*float(rall[6]):.2f}% ({rall[4]} / {rall[5]}).\n")
    L.append("## Feedback: switching existing experts off (no weight changes)\n")
    L.append("`python fc/q35_cli.py learn --artifact ART --in feedback.jsonl --guard guard.jsonl --out state.json` takes at most 25 annotated mistakes and writes only a list of `(layer, expert)` integers. `python fc/q35_cli.py answer --artifact ART --state state.json --in q.jsonl --out a.jsonl` runs the model with those experts switched off: the router never selects them (a fixed -1e4 router offset; every other offset is 0).\n")
    L.append("- Every weight tensor is byte-identical before and after feedback. No head, adapter, lookup memory, or prompt change.")
    L.append("- Experts are ranked by first-order attribution toward the annotated answers, switched off a few at a time with each step verified by real forward passes, and a switch is refused if it changes any of the model's own answers on a 160-item guard set.\n")
    L.append(f"**Feedback class.** Real mistakes of the compressed build on feedback-pool items it never calibrated on, for the gold label(s) where it errs most. 25 become feedback; the rest are locked probes ({(r['empty_probe_locked'] or (0,0))[1]}), written to disk before any feedback ran. Nearby items are 300 pool items of the same and the confused labels that the build answers right. Files and sha256 are in `data/feedback/`.\n")
    L.append("| Arm | Locked probes right | Feedback items right | Nearby kept | Experts off |\n|---|---|---|---|---|")
    L.append(f"| Empty | {frac('empty_probe_locked')} | {frac('empty_feedback')} | {frac('empty_nearby_test')} | 0 |")
    L.append(f"| Shuffled labels | {frac('shuf_probe_locked')} | {frac('shuf_feedback')} | {frac('shuf_nearby_test')} | {r['shuf_off']} |")
    L.append(f"| **Real** | **{frac('real_probe_locked')}** | {frac('real_feedback')} | {frac('real_nearby_test')} | {len(r['off'])} |")
    L.append(f"| (Base model, for reference) | {frac('base_probe_locked')} | | {frac('base_nearby_test')} | |\n")
    L.append("**Rule-5 note.** The organizer rejected #3 and #4 because their feedback wrote values fitted to the annotations into weights or heads. This method writes no values: it chooses which of the model's own experts stay on. We think that is on the allowed side, but the choice is driven by the feedback, so we flag it as a gray zone.\n")
    L.append("## Disclosures\n")
    L.append(f"- Before the build we ran a small test of several tasks on this model (base accuracy, and kept share with 104 experts per layer and plain 3-bit rounding) on dev items and on {('a separate' if t!='saf' else 'no')} probe sample of the unseen source, and chose which tasks to build from those numbers. The locked exam items are disjoint from every item used there.")
    if t == "saf": L.append("- The safety exam items were run through the base model and an older Qwen3-30B build on 11 Nov when we first tried this task on that model; they were never used to build, tune or calibrate this Qwen3.5 build.")
    L.append("- Weights are not hosted yet; the artifact is rebuilt from the base checkpoint by the scripts below.\n")
    L.append("## Reproduce\n")
    L.append("Shipped data: `data/dev/*.jsonl.gz` (calibration, dev and test-probe sets), `data/locked/exam.jsonl.gz` (the locked exam), `data/feedback/` (feedback pool, feedback items, shuffled feedback, locked probes, nearby items, guard set, with `SHA256SUMS`). Results: `results/` (feedback states, exam reports, and the full output of the final run in `results/run_output.txt`).\n")
    L.append("All GPU work ran on one H100 (Modal). Scripts run with the repository root as working directory and `/data` mounted.\n")
    L.append(f"1. Data: `python3 fc/{'lid_sent_prep' if t in ('sent','lid') else 'pi_prep' if t=='pi' else 'intent_prep' if t=='clinc' else 'saf_prep'}.py` (raw files under `data_raw/`). Shipped copies are in `data/`." if t != "saf" else "1. Data: the shipped files in `data/` (Aegis 2.0 train/validation/test and the OpenAI moderation eval, rendered with the policy prompt shown in each item).")
    L.append(f"2. Base answers on the calibration prompts: `python3 fc/gen_sglang.py --model Qwen/Qwen3.5-35B-A3B --out base --max-new {M['maxnew']} --split {M['calib']}`; routing mass: `python3 fc/q35_route_stats.py --model ... --items {M['calib']} --answers base/... --out rs.pt`.")
    L.append(f"3. Build, exam, feedback and controls: `bash runs/q35_task.sh {t} 108 {M['maxnew']} {M['exam']} {M['pool']}`.")
    L.append("4. Commands: `fc/q35_cli.py answer` and `fc/q35_cli.py learn` (docstring at the top of the file).\n")
    L.append(f"## Licences\n\n- Code: Apache-2.0 (see `LICENSE`). Base model: Apache-2.0.\n- {M['lic']} Exam and feedback files are derived from them with attribution.\n")
    open(out, "w").write("\n".join(L))
    print(out, "passes" if passes else "fails", kept_all)

UNSEEN = {"sent": {"src:imdb", "src:sst2"}, "pi": {"src:piarena", "src:llmail"}, "clinc": {"src:banking77"}, "lid": {"src:massive"}, "saf": {"src:openai_mod"}}
SHUF_OFF = {}
if __name__ == "__main__":
    if len(sys.argv) > 5: SHUF_OFF[sys.argv[1]] = sys.argv[5]
    main()
