"""Unpack the artifact with a learned feedback state and generate with SGLang (as the shipped system would)."""
import sys, json, torch, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unpack import unpack_artifact
art, state, dst = sys.argv[1], sys.argv[2], sys.argv[3]
edits = off = None
if state.endswith(".json"): off = [tuple(x) for x in json.load(open(state))]
elif state.endswith(".pt"):
    f = torch.load(state); edits = {f"model.layers.{l}.self_attn.o_proj.weight": (D.float() @ A.float().T) for l, (D, A) in f.items()}
unpack_artifact(art, dst, edits=edits, router_off=off)
