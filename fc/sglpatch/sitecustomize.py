# Loaded at startup by every Python process when this folder is on PYTHONPATH.
# With FC_ROUTER_BIAS=1, gives SGLang's MoE routers (any ReplicatedLinear whose prefix ends in "mlp.gate") a
# per-expert logit offset read from the checkpoint ("mlp.gate.bias"). The feedback "expert switch" list sets it:
# -1e4 switches an expert off, 0 leaves it alone. Works for Qwen3-MoE and Qwen3.5-MoE.
import os
if os.environ.get("FC_ROUTER_BIAS") == "1":
    try:
        from sglang.srt.layers import linear as _lin
        _orig = _lin.ReplicatedLinear.__init__
        def _init(self, *args, **kw):
            if str(kw.get("prefix", "")).endswith("mlp.gate"):
                if len(args) >= 3: args = args[:2] + (True,) + args[3:]
                else: kw["bias"] = True
            _orig(self, *args, **kw)
        _lin.ReplicatedLinear.__init__ = _init
    except Exception as e:  # pragma: no cover
        print("router-bias patch failed:", repr(e))
