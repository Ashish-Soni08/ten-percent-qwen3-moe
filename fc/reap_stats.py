"""REAP-style expert saliency on calibration data (prompt + the base model's own answer, teacher forced).

For each layer l and expert e: S[l,e] = sum over tokens routed to e of gate_weight * ||expert_e(x)||, plus counts.
Saved separately for all tokens and for answer tokens only.
"""
import argparse, json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--items", required=True); ap.add_argument("--answers", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=100000)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda", experts_implementation="grouped_mm")
model.eval(); cfg = model.config
L, E, K = cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok
stats = {k: torch.zeros(L, E, dtype=torch.float64, device="cuda") for k in ["S_all", "C_all", "W_all", "S_ans", "C_ans", "W_ans"]}
state = {"ans_mask": None}
imp_in = torch.zeros(L, E, cfg.hidden_size, dtype=torch.float32, device="cuda")
imp_dn = torch.zeros(L, E, cfg.moe_intermediate_size, dtype=torch.float32, device="cuda")

def make_hook(l, moe):
    def hook(mod, args):
        x = args[0].reshape(-1, args[0].shape[-1])
        x = x[state["valid"]]
        probs = torch.softmax(x.float() @ moe.gate.weight.float().T, -1)
        tv, ti = probs.topk(K, -1); tv = tv / tv.sum(-1, keepdim=True)
        am = state["ans_mask"]
        ex = moe.experts
        for e in ti.unique().tolist():
            t, p = torch.where(ti == e)
            g, u = torch.nn.functional.linear(x[t], ex.gate_up_proj[e]).chunk(2, -1)
            h = ex.act_fn(g) * u
            y = torch.nn.functional.linear(h, ex.down_proj[e])
            imp_in[l, e] += x[t].float().pow(2).sum(0); imp_dn[l, e] += h.float().pow(2).sum(0)
            w = tv[t, p].double(); nrm = y.float().norm(dim=-1).double()
            stats["S_all"][l, e] += (w * nrm).sum(); stats["C_all"][l, e] += len(t); stats["W_all"][l, e] += w.sum()
            m = am[t]
            stats["S_ans"][l, e] += (w * nrm)[m].sum(); stats["C_ans"][l, e] += m.sum(); stats["W_ans"][l, e] += w[m].sum()
    return hook

for l, layer in enumerate(model.model.layers):
    layer.mlp.register_forward_pre_hook(make_hook(l, layer.mlp))
items = {json.loads(x)["id"]: json.loads(x) for x in open(a.items)}
answers = [json.loads(x) for x in open(a.answers)][: a.n]
seqs = []
for r in answers:
    it = items[r["id"]]
    p = tok.apply_chat_template(it["messages"], tools=it["tools"], add_generation_prompt=True, tokenize=False)
    pi = tok(p, add_special_tokens=False).input_ids; ai = tok(r["text"] + "<|im_end|>", add_special_tokens=False).input_ids
    seqs.append((pi + ai, [False] * len(pi) + [True] * len(ai)))
seqs.sort(key=lambda s: len(s[0]))
t0 = time.time(); ntok = 0; TOK = 24000; i = 0; nb = 0
with torch.no_grad():
    while i < len(seqs):
        j = i
        while j < len(seqs) and (j - i + 1) * len(seqs[j][0]) <= TOK: j += 1
        j = max(j, i + 1); batch = seqs[i:j]; i = j
        T = max(len(s[0]) for s in batch)
        ids = torch.full((len(batch), T), tok.pad_token_id or 0, device="cuda"); att = torch.zeros_like(ids)
        am = torch.zeros(len(batch), T, dtype=torch.bool, device="cuda")
        for b, (s, m) in enumerate(batch):
            ids[b, : len(s)] = torch.tensor(s); att[b, : len(s)] = 1; am[b, : len(s)] = torch.tensor(m)
        state["valid"] = att.reshape(-1).bool(); state["ans_mask"] = am.reshape(-1)[state["valid"]]
        model(input_ids=ids, attention_mask=att); ntok += int(att.sum()); nb += 1
        if nb % 10 == 0: print(nb, i, f"{time.time()-t0:.0f}s", ntok, flush=True)
torch.save({k: v.cpu() for k, v in stats.items()} | {"ntok": ntok}, a.out)
cnt = stats["C_all"].float().clamp(min=1).unsqueeze(-1)
torch.save({"in": (imp_in / cnt).cpu(), "down": (imp_dn / cnt).cpu()}, a.out.replace(".pt", "_imp.pt"))
print("done", time.time() - t0)
