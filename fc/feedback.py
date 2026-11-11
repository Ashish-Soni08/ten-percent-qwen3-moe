"""Learning from annotated feedback without training: two methods on the shipped (compressed) model.

  switch : expert switching. Finds a short list of (layer, expert) pairs to switch off so the corrected answers
           become the model's own greedy answers. Candidates are ranked by first-order attribution and every switch is
           verified by a real forward pass; a guard set of answers the model already gets right must not move.
           Storage: the list. Runtime: router logit offset -1e4 for those experts.
  edit   : closed-form weight edit (MEMIT/AlphaEdit style, no gradient steps on weights). At each token where the
           model's greedy choice differs from the annotation, take the residual direction that raises the annotated
           token over the wrong one (closed form from the output layer) and write it into the attention output
           projections of a few layers with a covariance-regularised least-squares update. Repeated for a few rounds.
           Storage: low-rank factors per edited layer.
"""
import json, math, time, torch, torch.nn.functional as F
import transformers.models.qwen3_moe.modeling_qwen3_moe as qm
from transformers import AutoTokenizer, AutoModelForCausalLM

def _router_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    b = getattr(self, "fc_bias", None)
    if b is not None: router_logits = router_logits + b
    router_probs = torch.softmax(router_logits, dtype=torch.float, dim=-1)
    top_v, top_i = torch.topk(router_probs, self.top_k, dim=-1)
    if self.norm_topk_prob: top_v = top_v / top_v.sum(dim=-1, keepdim=True)
    return router_logits, top_v.to(router_logits.dtype), top_i
qm.Qwen3MoeTopKRouter.forward = _router_forward

def render(calls):
    return "\n".join("<tool_call>\n" + json.dumps({"name": c["name"], "arguments": c["arguments"]}, ensure_ascii=False) + "\n</tool_call>" for c in calls)

class FB:
    def __init__(self, ckpt):
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.m = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda", experts_implementation="grouped_mm")
        self.m.eval(); self.m.requires_grad_(False)
        self.layers = self.m.model.layers; self.E = self.m.config.num_experts
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")
        # router offsets stored in the checkpoint (padding slots are -1e4); transformers does not load them itself
        import os, glob
        from safetensors import safe_open
        self.bias0 = {}
        for fn in glob.glob(os.path.join(ckpt, "*.safetensors")):
            with safe_open(fn, "pt", device="cuda") as f:
                for k in f.keys():
                    if k.endswith("mlp.gate.bias"): self.bias0[int(k.split(".")[2])] = f.get_tensor(k).to(torch.bfloat16)
        self.set_bias([])

    def encode(self, it, answer_text):
        p = self.tok.apply_chat_template(it["messages"], tools=it["tools"], add_generation_prompt=True, tokenize=False)
        pi = self.tok(p, add_special_tokens=False).input_ids
        ai = self.tok(answer_text, add_special_tokens=False).input_ids + [self.end]
        return pi, ai

    # ---------- shared: teacher-forced pass, per-position margins
    def tf(self, pi, ai, grad=False):
        ids = torch.tensor([pi + ai], device="cuda")
        with torch.set_grad_enabled(grad):
            out = self.m(input_ids=ids)
        lg = out.logits[0, len(pi) - 1: len(pi) + len(ai) - 1].float()
        return lg, torch.tensor(ai, device="cuda")

    def divergences(self, pi, ai):
        with torch.no_grad():
            lg, tgt = self.tf(pi, ai)
        am = lg.argmax(-1)
        return [(i, int(tgt[i]), int(am[i])) for i in range(len(ai)) if am[i] != tgt[i]]

    def set_bias(self, off):
        for l, layer in enumerate(self.layers):
            b = self.bias0.get(l, torch.zeros(self.E, device="cuda", dtype=torch.bfloat16)).clone()
            for (ll, e) in off:
                if ll == l: b[e] = -1e4
            layer.mlp.gate.fc_bias = b

    def seq_loss(self, pairs):
        """Sum of per-token CE over annotated answers (teacher forced)."""
        tot = 0.0
        with torch.no_grad():
            for pi, ai in pairs:
                lg, tgt = self.tf(pi, ai); tot += F.cross_entropy(lg, tgt, reduction="sum").item()
        return tot

    def n_wrong(self, pairs):
        return sum(len(self.divergences(pi, ai)) > 0 for pi, ai in pairs)

    # ---------- method A: expert switching
    def attrib(self, pairs, off):
        """First-order effect on the loss of switching each active (layer, expert) off: sum_t dL/dy_t . (-w_te f_e(x_t))."""
        self.set_bias(off)
        score = torch.zeros(len(self.layers), self.E, device="cuda", dtype=torch.float64)
        caps = {}
        def pre(l):
            def h(mod, args): caps[l] = args[0]
            return h
        def post(l):
            def h(mod, args, out):
                out.retain_grad(); caps[("y", l)] = out
            return h
        hs = []
        for l, layer in enumerate(self.layers):
            hs.append(layer.mlp.register_forward_pre_hook(pre(l))); hs.append(layer.mlp.register_forward_hook(post(l)))
        try:
            for pi, ai in pairs:
                caps.clear()
                with torch.enable_grad():
                    ids = torch.tensor([pi + ai], device="cuda")
                    emb = self.m.model.embed_tokens(ids).detach().requires_grad_(True)
                    out = self.m(inputs_embeds=emb)
                    lg = out.logits[0, len(pi) - 1: len(pi) + len(ai) - 1].float()
                    loss = F.cross_entropy(lg, torch.tensor(ai, device="cuda"), reduction="sum")
                    loss.backward()
                with torch.no_grad():
                    for l, layer in enumerate(self.layers):
                        x = caps[l][0].detach(); gy = caps[("y", l)].grad[0]
                        moe = layer.mlp
                        _, tv, ti = moe.gate(x)
                        ex = moe.experts
                        for e in ti.unique().tolist():
                            t, p = torch.where(ti == e)
                            g, u = F.linear(x[t], ex.gate_up_proj[e]).chunk(2, -1)
                            y = F.linear(ex.act_fn(g) * u, ex.down_proj[e]) * tv[t, p, None]
                            score[l, e] += -(gy[t].float() * y.float()).sum().double()
        finally:
            for h in hs: h.remove()
        return score

    def learn_switch(self, pairs, guard, max_off=24, step=4, rounds=8, guard_tol=0, init=None):
        off = list(init or []); self.set_bias(off); base_wrong = self.n_wrong(pairs)
        best = self.seq_loss(pairs); log = [("start", best, base_wrong)]
        guard_ok0 = len(guard) - self.n_wrong(guard)
        for r in range(rounds):
            if len(off) - len(init or []) >= max_off: break
            sc = self.attrib(pairs, off)
            for (l, e) in off: sc[l, e] = float("inf")
            cand = [(int(i // self.E), int(i % self.E)) for i in sc.flatten().argsort()[:24].tolist() if sc.flatten()[i] < 0]
            took = False
            for c in [cand[i:i + step] for i in range(0, len(cand), step)] + [[c] for c in cand]:
                trial = off + c; self.set_bias(trial)
                L = self.seq_loss(pairs)
                if L < best * 0.97:
                    g_ok = len(guard) - self.n_wrong(guard)
                    if g_ok >= guard_ok0 - guard_tol:
                        off, best, took = trial, L, True; log.append((c, L, self.n_wrong(pairs), g_ok)); break
            if not took: break
        self.set_bias(off)
        return off, log

    # ---------- method B: closed-form edit
    def key_cov(self, items_texts, layers, n_tok=200000):
        C = {l: torch.zeros(4096, 4096, device="cuda", dtype=torch.float64) for l in layers}; cnt = 0; caps = {}
        hs = [self.layers[l].self_attn.o_proj.register_forward_pre_hook((lambda l: lambda m, a: caps.__setitem__(l, a[0]))(l)) for l in layers]
        with torch.no_grad():
            for pi, ai in items_texts:
                self.m(input_ids=torch.tensor([pi + ai], device="cuda"))
                for l in layers:
                    k = caps[l][0].double(); C[l] += k.T @ k
                cnt += len(pi) + len(ai)
                if cnt > n_tok: break
        for h in hs: h.remove()
        return {l: (C[l] / cnt).float() for l in layers}

    def learn_edit(self, pairs, layers, C, lam=1e4, tau=2.0, rounds=4):
        W0 = {l: self.layers[l].self_attn.o_proj.weight.data.clone() for l in layers}
        factors = {l: [] for l in layers}; log = []
        for r in range(rounds):
            Ks = {l: [] for l in layers}; Ds = []
            caps = {}
            hs = [self.layers[l].self_attn.o_proj.register_forward_pre_hook((lambda l: lambda m, a: caps.__setitem__(l, a[0]))(l)) for l in layers]
            hn = self.m.model.norm.register_forward_pre_hook(lambda m, a: caps.__setitem__("h", a[0]))
            nd = 0
            for pi, ai in pairs:
                with torch.no_grad():
                    lg, tgt = self.tf(pi, ai)
                hall = caps["h"][0].detach(); kall = {l: caps[l][0].detach() for l in layers}
                am = lg.argmax(-1)
                # positions where the annotated token does not win by tau
                top2 = lg.topk(2, -1)
                rival = torch.where(top2.indices[:, 0] == tgt, top2.indices[:, 1], top2.indices[:, 0])
                marg = lg.gather(1, tgt[:, None])[:, 0] - lg.gather(1, rival[:, None])[:, 0]
                pos = torch.where((am != tgt) | (marg < tau * 0.5))[0].tolist()
                pos = [i for i in pos if am[i] != tgt[i]] + [i for i in pos if am[i] == tgt[i]][:4]
                nd += sum(int(am[i] != tgt[i]) for i in pos)
                for i in pos:
                    t = len(pi) - 1 + i
                    h = hall[t].float().requires_grad_(True)
                    z = self.m.lm_head(self.m.model.norm(h.to(torch.bfloat16)[None]))[0].float()
                    mg = z[tgt[i]] - z[rival[i]]
                    g, = torch.autograd.grad(mg, h)
                    need = max(tau - mg.item(), 0.0)
                    if need <= 0: continue
                    Ds.append(g * need / g.pow(2).sum().clamp(min=1e-6) / len(layers))
                    for l in layers: Ks[l].append(kall[l][t].float())
            for hh in hs + [hn]: hh.remove()
            log.append((r, nd, len(Ds)))
            if not Ds: break
            D = torch.stack(Ds, 1)  # (2048, n)
            for l in layers:
                K = torch.stack(Ks[l], 1)  # (4096, n)
                M = K @ K.T + lam * C[l] + 1e-4 * torch.eye(K.shape[0], device="cuda")
                A = torch.linalg.solve(M, K)  # (4096, n): dW = D A^T
                dW = D @ A.T
                self.layers[l].self_attn.o_proj.weight.data += dW.to(torch.bfloat16)
                factors[l].append((D.half().cpu(), A.half().cpu()))
        return factors, log

    def reset_edit(self, W0):
        for l, w in W0.items(): self.layers[l].self_attn.o_proj.weight.data.copy_(w)
