"""Group-wise asymmetric uniform quantization with MSE-optimal clipping, and dense bit packing (1..8 bits)."""
import torch

def quantize(w, bits, group, grid=(1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76), imp=None):
    """w: (..., out, in) float. imp: optional (..., in) per-input-channel importance (e.g. E[x^2]).
    Returns codes uint8 (..., out, in), scale fp16 (..., out, in/group), mn fp16 (..., out, in/group)."""
    shp = w.shape; n = shp[-1]; assert n % group == 0
    wg = w.float().reshape(*shp[:-1], n // group, group)
    ig = None if imp is None else imp.float().reshape(*imp.shape[:-1], 1, n // group, group).expand_as(wg)
    mx0, mn0 = wg.amax(-1, keepdim=True), wg.amin(-1, keepdim=True)
    qmax = 2 ** bits - 1
    best_err = None
    for s in grid:
        mx, mn = mx0 * s, mn0 * s
        scale = ((mx - mn) / qmax).clamp(min=1e-8)
        scale = scale.half().float(); mn = mn.half().float()
        q = ((wg - mn) / scale).round().clamp(0, qmax)
        err = (q * scale + mn - wg).pow(2)
        if ig is not None: err = err * ig
        err = err.sum(-1, keepdim=True)
        if best_err is None:
            best_err, bq, bs, bm = err, q, scale, mn
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            bq = torch.where(better, q, bq); bs = torch.where(better, scale, bs); bm = torch.where(better, mn, bm)
    return bq.reshape(shp).to(torch.uint8), bs.squeeze(-1).half(), bm.squeeze(-1).half()

def dequantize(codes, scale, mn, group, dtype=torch.bfloat16):
    shp = codes.shape
    q = codes.float().reshape(*shp[:-1], shp[-1] // group, group)
    return (q * scale.float().unsqueeze(-1) + mn.float().unsqueeze(-1)).reshape(shp).to(dtype)

def pack(codes, bits):
    """codes uint8 tensor (any shape, numel % 8 == 0) -> flat uint8 of numel*bits/8 bytes."""
    v = codes.reshape(-1, 8).to(torch.int64)
    acc = torch.zeros(v.shape[0], dtype=torch.int64, device=v.device)
    for i in range(8): acc |= v[:, i] << (bits * i)
    return torch.stack([((acc >> (8 * j)) & 0xFF) for j in range(bits)], 1).to(torch.uint8).reshape(-1)

def unpack(packed, bits, shape):
    b = packed.reshape(-1, bits).to(torch.int64)
    acc = torch.zeros(b.shape[0], dtype=torch.int64, device=b.device)
    for j in range(bits): acc |= b[:, j] << (8 * j)
    m = (1 << bits) - 1
    return torch.stack([(acc >> (bits * i)) & m for i in range(8)], 1).to(torch.uint8).reshape(shape)

if __name__ == "__main__":
    w = torch.randn(4, 64, 256)
    for b in (2, 3, 4, 5, 6, 8):
        c, s, m = quantize(w, b, 64)
        p = pack(c, b); assert torch.equal(unpack(p, b, c.shape), c)
        d = dequantize(c, s, m, 64, torch.float32)
        print(b, p.numel() / c.numel() * 8, ((d - w).pow(2).mean() / w.pow(2).mean()).item())
