"""Feedback without training for the Qwen3.5 builds: expert switching.
Finds a short list of (layer, expert) pairs to switch off so the annotated answers become the model's own greedy
answers. Candidates are ranked by first-order attribution and every switch is verified by real forward passes; a
guard set of answers the model already gets right must not move. Storage: the list. Runtime: router logit
offset -1e4 for those experts (a fixed constant, nothing fitted)."""
import glob, os, torch, torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm

ROUTER = getattr(qm, "Qwen3_5MoeTopKRouter", None)
if ROUTER is not None:
    _orig_fwd = ROUTER.forward
    def _router_forward(self, hidden_states):
        b = getattr(self, "fc_bias", None)
        if b is None: return _orig_fwd(self, hidden_states)
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight) + b
        router_probs = torch.softmax(router_logits, dtype=torch.float, dim=-1)
        top_v, top_i = torch.topk(router_probs, self.top_k, dim=-1)
        if getattr(self, "norm_topk_prob", True): top_v = top_v / top_v.sum(dim=-1, keepdim=True)
        return router_logits, top_v.to(router_logits.dtype), top_i
    ROUTER.forward = _router_forward

class FB:
    def __init__(self, ckpt):
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.m = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda")
        self.m.eval(); self.m.requires_grad_(False)
        self.layers = self.m.model.layers; self.E = self.m.config.num_experts
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.set_bias([])

    def encode(self, it, answer_text):
        p = self.tok.apply_chat_template(it["messages"], tools=it.get("tools"), add_generation_prompt=True, tokenize=False, enable_thinking=False)
        pi = self.tok(p, add_special_tokens=False).input_ids[-4000:]
        ai = self.tok(answer_text, add_special_tokens=False).input_ids + [self.end]
        return pi, ai

    def tf(self, pi, ai):
        ids = torch.tensor([pi + ai], device="cuda")
        out = self.m(input_ids=ids)
        return out.logits[0, len(pi) - 1: len(pi) + len(ai) - 1].float(), torch.tensor(ai, device="cuda")

    def wrong(self, pi, ai):
        with torch.no_grad():
            lg, tgt = self.tf(pi, ai)
        return bool((lg.argmax(-1) != tgt).any())

    def set_bias(self, off):
        for l, layer in enumerate(self.layers):
            b = torch.zeros(self.E, device="cuda", dtype=torch.bfloat16)
            for (ll, e) in off:
                if ll == l: b[e] = -1e4
            layer.mlp.gate.fc_bias = b

    def seq_loss(self, pairs):
        tot = 0.0
        with torch.no_grad():
            for pi, ai in pairs:
                lg, tgt = self.tf(pi, ai); tot += F.cross_entropy(lg, tgt, reduction="sum").item()
        return tot

    def n_wrong(self, pairs): return sum(self.wrong(pi, ai) for pi, ai in pairs)

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
                o = out[0] if isinstance(out, tuple) else out
                o.retain_grad(); caps[("y", l)] = o
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
                    F.cross_entropy(lg, torch.tensor(ai, device="cuda"), reduction="sum").backward()
                with torch.no_grad():
                    for l, layer in enumerate(self.layers):
                        x = caps[l][0].detach().reshape(-1, caps[l].shape[-1]); gy = caps[("y", l)].grad[0].reshape(-1, x.shape[-1])
                        moe = layer.mlp
                        r = moe.gate(x); tv, ti = r[1], r[2]
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
        off = list(init or []); self.set_bias(off)
        best = self.seq_loss(pairs); log = [("start", best, self.n_wrong(pairs))]
        guard_ok0 = len(guard) - self.n_wrong(guard)
        for r in range(rounds):
            if len(off) - len(init or []) >= max_off: break
            sc = self.attrib(pairs, off)
            for (l, e) in off: sc[l, e] = float("inf")
            flat = sc.flatten()
            cand = [(int(i // self.E), int(i % self.E)) for i in flat.argsort()[:24].tolist() if flat[i] < 0]
            took = False
            for c in [cand[i:i + step] for i in range(0, len(cand), step)] + [[c] for c in cand]:
                trial = off + c; self.set_bias(trial)
                L = self.seq_loss(pairs)
                if L < best * 0.97:
                    g_ok = len(guard) - self.n_wrong(guard)
                    if g_ok >= guard_ok0 - guard_tol:
                        off, best, took = trial, L, True; log.append((c, L, self.n_wrong(pairs), g_ok)); print(log[-1], flush=True); break
            if not took: break
        self.set_bias(off)
        return off, log
