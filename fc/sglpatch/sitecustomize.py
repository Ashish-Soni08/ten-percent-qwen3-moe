# Loaded at startup by every Python process when this folder is on PYTHONPATH.
# With FC_ROUTER_BIAS=1, gives SGLang's Qwen3-MoE router a per-expert logit offset read from the checkpoint
# ("mlp.gate.bias"). The feedback "expert switch" list sets it: -1e4 switches an expert off, 0 leaves it alone.
import os
if os.environ.get("FC_ROUTER_BIAS") == "1":
    try:
        import sglang.srt.models.qwen3_moe as q
        from sglang.srt.layers.linear import ReplicatedLinear
        _orig = q.Qwen3MoeSparseMoeBlock.__init__
        def _init(self, layer_id, config, quant_config=None, prefix=""):
            _orig(self, layer_id, config, quant_config=quant_config, prefix=prefix)
            self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=True, quant_config=None,
                                         prefix=prefix + ".gate" if prefix else "gate")
        q.Qwen3MoeSparseMoeBlock.__init__ = _init
    except Exception as e:  # pragma: no cover
        print("router-bias patch failed:", repr(e))
