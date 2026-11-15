#!/usr/bin/env python
"""ORACLE SELECTION ARM (OSA) — run the sparse model with EXACT attention-mass block selection.

WHY THIS IS NOW THE TOP EXPERIMENT. COBS (2607.09052) Table 1 reports an exact-mass oracle at
32k RULER on their ~1.2B NSA model:

  task    S1   S2   S3   MK1  MK2  MK3  MQ   MV   CWE   FWE  VT
  dense  1.00 1.00 1.00 0.98 0.95 0.92 0.92 0.93 0.44  0.90 0.92
  OSA    1.00 1.00 1.00 0.99 0.94 0.91 0.96 0.96 0.22  0.95 0.98

Their central claim — "if selection retrieves the blocks with the largest attention mass, block
sparse attention can match the quality of dense attention" — holds on 10 of 11 tasks. It fails on
exactly ONE: **CWE, 0.44 -> 0.22, halved**, and they never remark on it. CWE (common-word
extraction) needs evidence AGGREGATED over the whole context rather than retrieved from a span,
so no ranking of blocks by mass can capture it.

That is the oracle contradiction in a single controlled setup:
  **mass is sufficient for RETRIEVAL (concentrated evidence), insufficient for AGGREGATION
  (diffuse evidence).**
Running OSA on OUR substrate (8B InfLLM-V2, 18.8% coverage vs their ~2.3%) turns that from a
citation of someone else's unremarked table into our own measurement, at a different scale, a
different model family, and a different budget regime.

⚠ It also tests whether OUR key negative and COBS's claim are compatible. They are, and the
distinction matters: COBS shows exact mass is a sufficient CRITERION; the project notes§13 show the
QUANTITY of mass a real selector misses does not predict accuracy loss. Both hold, because what
matters is WHICH blocks are missed, not HOW MUCH mass.

HOW. `sparse_forward(q, k, v, ...)` holds the full post-RoPE keys. We stash them, then patch
`compressed_attention` to ignore its pooled estimate and return the top-k blocks by TRUE mass
  mass_b = sum_{j in block b} softmax(q . k_j / sqrt(d))
keeping the selector's own budget, its forced blocks, and its causal masking untouched.

VALIDATION GATE (must run first): with --oracle off this must reproduce the ordinary sparse
number for the cell. With topk >= n_blocks it must reproduce dense. A probe that cannot fail is
not evidence.
"""
import argparse, hashlib, inspect, json, math, os, random, statistics, subprocess, sys, time
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from ckpt_guard import load_final  # noqa: E402  refuses partial training checkpoints

STASH = {}


def load_task(ruler_root, task):
    root = Path(ruler_root)
    cfg = yaml.safe_load((root / "scripts" / "synthetic.yaml").read_text())
    base = cfg[task]["task"]
    import importlib.util
    sp = importlib.util.spec_from_file_location("rec", root/"scripts"/"eval"/"synthetic"/"constants.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    dsp = importlib.util.spec_from_file_location("rdc", root/"scripts"/"data"/"synthetic"/"constants.py")
    dm = importlib.util.module_from_spec(dsp); dsp.loader.exec_module(dm)
    return m.TASKS[base]["metric_fn"], base, int(dm.TASKS[base]["tokens_to_generate"])


def learned_block_scores(qs, kk, Wq_lh, Wk_lh, tau, n_blocks, block):
    """TRAINED indexer block scores (deployment of pheno/indexer_train.py).

    Must match training EXACTLY (the QACP e2e died on a probe/deploy √D mismatch, §50):
    group-mean query, s = (q̄ Wq)·(k Wk)/τ, NO √D factor. Block rank = logsumexp of per-key
    projected logits — per-key structure preserved, which is what every summary lacks (§51).
    qs: [grp, D] this kv-head's queries; kk: [tk, D]; Wq_lh/Wk_lh: [D, r]."""
    qb = qs.mean(0) @ Wq_lh                                    # [r]
    lg = (kk @ Wk_lh) @ qb / tau                               # [tk] per-key indexer logits
    pad = n_blocks * block - kk.shape[0]
    if pad:
        lg = torch.cat([lg, lg.new_full((pad,), -float("inf"))])
    return torch.logsumexp(lg.view(n_blocks, block), dim=-1)   # [n_blocks]


def _parse_layer_budget(spec):
    """'0-15:41,16-23:126,24-31:40' -> [(0, 15, 41), (16, 23, 126), (24, 31, 40)]: FREE slots per inclusive layer range.
    Ranges must not overlap. Layers that no range covers keep every free slot of the row."""
    if spec is None:
        return None
    out = []
    for part in spec.split(","):
        rng, _, n = part.partition(":")
        lo, hi = _parse_layers(rng)
        n = int(n)
        if n < 0:
            raise ValueError(f"negative budget in {part!r}")
        if any(not (hi < a or lo > b) for a, b, _ in out):
            raise ValueError(f"overlapping layer ranges in {spec!r}")
        out.append((lo, hi, n))
    return out


def _layer_free_budget(lb, li):
    """Free slots granted to layer li, or None when no range covers it."""
    for lo, hi, n in lb or ():
        if lo <= li <= hi:
            return n
    return None


def _parse_layers(spec):
    """'16-23' -> (16, 23), '5' -> (5, 5), None -> None. Inclusive layer range for --layers."""
    if spec is None:
        return None
    lo, _, hi = spec.partition("-")
    lo, hi = int(lo), int(hi) if hi else int(lo)
    if lo < 0 or hi < lo:
        raise ValueError(f"bad --layers {spec!r}")
    return lo, hi


def _parse_layer_set(spec):
    """--layers: '16-23' -> (16, 23) exactly as before; a comma list such as '9,16,22' or '9,16-17,22' -> a frozenset
    of layer indices (the layers a calibration probe flags need not be contiguous). None -> None."""
    if spec is None or "," not in spec:
        return _parse_layers(spec)
    out = set()
    for part in spec.split(","):
        lo, hi = _parse_layers(part.strip())
        out.update(range(lo, hi + 1))
    return frozenset(out)


def _parse_kv_heads(spec):
    """--kv_heads '9.0,16.0,22.0' -> frozenset({(9, 0), (16, 0), (22, 0)}) of (layer, KV head). None -> None."""
    if spec is None:
        return None
    out = set()
    for part in spec.split(","):
        l, _, h = part.strip().partition(".")
        if not h or int(l) < 0 or int(h) < 0:
            raise ValueError(f"bad --kv_heads entry {part!r} (want LAYER.HEAD)")
        out.add((int(l), int(h)))
    return frozenset(out)


def _layer_active(layers, li):
    """True when the selection override applies at layer li (always, unless --layers restricts it)."""
    if layers is None:
        return True
    if isinstance(layers, frozenset):
        return li in layers
    return layers[0] <= li <= layers[1]


def _n_active_layers(layers, n_layers, sparse_layers=None):
    """How many of the model's layers the override touches (for the hook-count gates). sparse_layers: the indices of the
    layers that HAVE a sparse-attention path (None = every layer). A hybrid such as MiniCPM-SALA keeps the InfLLM-V2
    selector in 8 of its 32 layers and uses linear attention elsewhere, so only those 8 can call the hook."""
    return sum(1 for li in range(n_layers) if _layer_active(layers, li) and (sparse_layers is None or li in sparse_layers))


def _sparse_layer_ids(model):
    """Indices of the layers whose attention module has the InfLLM-V2 sparse path (every layer for MiniCPM4)."""
    return frozenset(li for li, layer in enumerate(model.model.layers) if hasattr(layer.self_attn, "sparse_forward"))


# optimal step of a uniform mid-rise quantizer for a unit Gaussian (Max 1960; 8 bits: about 4 sigma of range)
_UNIFORM_STEP = {2: 0.996, 3: 0.586, 4: 0.335, 8: 0.0308}


_HADAMARD = {}
_KEY_ROT = {"on": False}          # set from --key_rot by build_model_and_state; every _quantize_keys call honours it
_QUANT_STATS = {"prefix": None}   # set from --quant_stats_prefix: per-channel statistics from the first N cached keys only
# set from --quant_form / --quant_step_scale / --quant_log (the quantizer ablations). The defaults are the repair's quantizer;
# "log" holds running sums of _log_quant's statistics when --quant_log is on.
_QUANT_FORM = {"form": "gauss", "step_scale": 1.0, "log": None}


def _hadamard(D, device):
    """Randomized Hadamard rotation [D, D], orthogonal: Sylvester matrix times a fixed random sign per input channel,
    divided by sqrt(D). Spreads outlier channels over all coordinates before quantization (as in Self-Indexing
    Attention and Adamas, which quantize rotated keys). D must be a power of two. Cached per (D, device)."""
    key = (D, str(device))
    if key not in _HADAMARD:
        if D & (D - 1):
            raise ValueError("Hadamard rotation needs a power-of-two head dimension")
        H = torch.ones(1, 1)
        while H.shape[0] < D:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        sgn = torch.where(torch.rand(D, generator=torch.Generator().manual_seed(1234)) < 0.5, -1.0, 1.0)
        _HADAMARD[key] = ((H * sgn.view(1, D)) / math.sqrt(D)).to(device)
    return _HADAMARD[key]


def _quantize_keys(kk, bits, rot=None):
    """With rot=True (or --key_rot hadamard) the keys are rotated by _hadamard before the per-channel quantizer and
    rotated back afterwards, so callers keep scoring q . k_hat in the original basis (a deployment rotates the query
    once and scans the rotated codes).
    Emulate a low-bit copy of the cached keys that is used ONLY for scoring blocks (attention itself still reads the
    full-precision keys of the selected blocks). kk: [tk, D]. Each channel is standardized with its own mean and
    standard deviation over the cached keys (two numbers per channel, no per-token metadata), so the copy costs
    D * bits / 8 bytes per key (128 dims: 4 bits = 64 B, the rank-32 bf16 indexer's size; 2 bits = 32 B; 1 bit = 16 B).
      bits >= 2: uniform mid-rise quantizer with the step that is optimal for a Gaussian, clipped to 2**bits levels.
      bits == 1: sign about the channel mean, scaled by the channel's mean absolute deviation (the optimal 1-bit code).
    bits None returns kk unchanged. A min-max grid is deliberately not used: outliers stretch it, and at 2 bits it had
    a larger error than the 1-bit code (caught by the unit test).
    Ablations only (--quant_form, --quant_step_scale, bits >= 2): "minmax" is KIVI's asymmetric per-channel grid (Liu et al.,
    2024), 2**bits levels from each channel's minimum to its maximum over the statistics keys, later keys clamped to the end
    levels; "pooled" standardizes the whole head with one mean and one standard deviation; step_scale multiplies the
    Gaussian step, which moves the clamp to +-(2**(bits-1)) * step standard deviations."""
    if bits is None:
        return kk
    if rot is None:
        rot = _KEY_ROT["on"]
    if rot:
        R = _hadamard(kk.shape[-1], kk.device).to(kk.dtype)
        return _quantize_keys(kk @ R.T, bits, rot=False) @ R
    # --quant_stats_prefix N: the channel statistics come from the first N keys only, so the code of a key never depends
    # on later keys (a causal quantizer: needed when rows in the middle of the prefill are scored, and it also means the
    # statistics are frozen once N keys are cached). None = statistics over all the keys passed in.
    ref = kk if _QUANT_STATS["prefix"] is None else kk[:max(int(_QUANT_STATS["prefix"]), 2)]
    form = _QUANT_FORM["form"]
    if _QUANT_FORM["log"] is not None and bits >= 2:
        _log_quant(kk, ref, bits)
    if form == "minmax" and bits >= 2:
        lo, hi = ref.min(0, keepdim=True).values, ref.max(0, keepdim=True).values
        s = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)
        return lo + torch.clamp(torch.round((kk - lo) / s), 0, 2 ** bits - 1) * s
    pooled = form == "pooled" and bits >= 2
    mu = ref.mean().view(1, 1) if pooled else ref.mean(0, keepdim=True)
    dev = kk - mu
    if bits == 1:
        return mu + torch.sign(dev) * (ref - mu).abs().mean(0, keepdim=True)
    sd = ((ref - mu).pow(2).mean().view(1, 1) if pooled else (ref - mu).pow(2).mean(0, keepdim=True)).sqrt().clamp(min=1e-8)
    step, half = _UNIFORM_STEP[bits] * _QUANT_FORM["step_scale"], 2 ** (bits - 1) - 0.5
    z = torch.clamp(torch.floor(dev / (sd * step)) + 0.5, -half, half)
    return mu + z * step * sd


def _log_quant(kk, ref, bits, n_sub=65536):
    """--quant_log: statistics of the keys as the repair's quantizer standardizes them (each channel by its own mean and
    standard deviation over the statistics keys), whatever --quant_form is armed. Per call: the share of entries beyond
    the range of the b-bit Gaussian step (Max 1960, where a unit Gaussian puts 2 * (1 - Phi(half * step)) of its mass),
    the channels' excess kurtosis (median, 90th percentile, max), and the uniform step that minimizes the squared error
    on these keys (grid of 0.5 to 2 times Max's step, on an evenly strided subsample of n_sub entries) together with the
    error at Max's step relative to that minimum. Calls with fewer than 1,024 keys are skipped. Sums in _QUANT_FORM["log"]."""
    if kk.shape[0] < 1024:
        return
    L = _QUANT_FORM["log"]
    with torch.no_grad():
        x, r = kk.float(), ref.float()
        mu = r.mean(0, keepdim=True)
        z = (x - mu) / (r - mu).pow(2).mean(0, keepdim=True).sqrt().clamp(min=1e-8)
        step, half = _UNIFORM_STEP[bits], 2 ** (bits - 1) - 0.5
        clip = (torch.floor(z / step) + 0.5).abs().gt(half).float().mean().item()
        zc = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp(min=1e-8)
        kurt = zc.pow(4).mean(0) - 3.0
        flat = z.flatten()
        flat = flat[::max(1, flat.numel() // n_sub)][:n_sub]
        steps = torch.linspace(0.5 * step, 2.0 * step, 61, device=flat.device)
        q = torch.clamp(torch.floor(flat[None, :] / steps[:, None]) + 0.5, -half, half) * steps[:, None]
        mse = (q - flat[None, :]).pow(2).mean(1)
        qm = torch.clamp(torch.floor(flat / step) + 0.5, -half, half) * step
        mse_max = (qm - flat).pow(2).mean()
        j = int(mse.argmin())
        vals = {"clip": clip, "kurt_median": kurt.median().item(), "kurt_p90": kurt.quantile(0.9).item(),
                "step_opt": steps[j].item(), "mse_ratio_max_step": (mse_max / mse[j]).item()}
    L["calls"] = L.get("calls", 0) + 1
    for k, v in vals.items():
        L[k] = L.get(k, 0.0) + v
    L["kurt_max"] = max(L.get("kurt_max", float("-inf")), kurt.max().item())
    L["bits"] = bits


def arm_quantizer(a):
    """Validate --quant_form / --quant_step_scale / --quant_log against --key_bits and arm _QUANT_FORM (SystemExit on a
    combination that would silently do nothing)."""
    if (a.quant_form != "gauss" or a.quant_step_scale != 1.0 or a.quant_log) and (a.key_bits is None or a.key_bits < 2):
        raise SystemExit("--quant_form / --quant_step_scale / --quant_log need --key_bits 2 or more -- ABORT")
    if a.quant_step_scale <= 0 or (a.quant_form == "minmax" and a.quant_step_scale != 1.0):
        raise SystemExit("--quant_step_scale must be positive, and the min-max grid has no step to scale -- ABORT")
    _QUANT_FORM.update(form=a.quant_form, step_scale=a.quant_step_scale, log=({} if a.quant_log else None))


def quant_log_summary():
    """Means over the logged calls of _log_quant's statistics (None when --quant_log is off or nothing was logged)."""
    L = _QUANT_FORM["log"]
    if not L or not L.get("calls"):
        return None
    n = L["calls"]
    step = _UNIFORM_STEP[L["bits"]]
    edge = 2 ** (L["bits"] - 1) * step                    # the codes cover [-edge, edge) standard deviations
    out = {k: L[k] / n for k in ("clip", "kurt_median", "kurt_p90", "step_opt", "mse_ratio_max_step")}
    out.update({"calls": n, "bits": L["bits"], "kurt_max": L["kurt_max"], "max_step": step, "edge_sd": edge,
                "gauss_clip": math.erfc(edge / math.sqrt(2.0))})
    return out


def _parse_key_summary(spec):
    """'pool:16' -> ('pool', 16); 'pool:32:16' -> ('pool', 32, 16), windows of 32 keys at a stride of 16 (the shipped
    selector's own windows; 2026-09-26, the point R3 asked for). Kinds: pool, channels, pca. None stays None."""
    if spec is None:
        return None
    kind, _, val = str(spec).partition(":")
    parts = val.split(":") if val else []
    if kind not in ("pool", "channels", "pca") or not parts or not all(v.isdigit() and int(v) >= 1 for v in parts) \
            or (kind != "pool" and len(parts) != 1) or len(parts) > 2:
        raise SystemExit(f"--key_summary {spec}: pool:L, pool:L:S, channels:R or pca:R with positive integers -- ABORT")
    if kind == "pool" and len(parts) == 2:
        L, S = int(parts[0]), int(parts[1])
        if S > L:
            raise SystemExit(f"--key_summary {spec}: the stride must not exceed the window -- ABORT")
        return kind, L, S
    return kind, int(parts[0])


def _summarize_keys(kk, spec, qs=None):
    """Full-precision key summaries that spend the selector's side memory on something other than per-token precision:
    the comparison arms of the precision-versus-resolution analysis (papers/THEORY_PRECISION_RESOLUTION.md). kk [tk, D]
    the cached keys of one KV head, qs [grp, D] the decode queries of its group (used by 'channels'), spec from
    _parse_key_summary. Returns [tk, D] keys to score in place of the true ones; at 16 bits per stored number the cost
    per token is D*16/L (pool), R*16 (channels, pca).
      pool:L      every key becomes the mean of its group of L consecutive keys (groups start at multiples of L, a short
                  last group averages what it has). Resolution is given up, precision is kept.
      channels:R  keep the R channels with the largest mean |q_c| * mean |k_c - mu_c| (label channels in the style of
                  Double Sparsity; chosen online for THIS query group from the prompt's own keys, which favours the
                  baseline over its offline calibration). The other channels are set to their mean over the keys,
                  which adds one constant per query to every logit and so changes no softmax over keys.
      pca:R       project the centred keys on their top-R principal directions (an online variant of Loki's offline
                  PCA, which again favours the baseline)."""
    kind, r = spec[0], spec[1]
    tk, D = kk.shape
    if kind == "pool" and len(spec) == 3 and spec[2] != r:
        # pool:L:S -- every key is replaced by the mean of the L-key window that starts at its own S-aligned group, the
        # windows of the shipped selector when L=32 and S=16 (a window that runs past the last key averages what exists)
        L, S = r, spec[2]
        cs = torch.cat([kk.new_zeros(1, D), kk.float().cumsum(0)])           # [tk+1, D]
        starts = torch.arange(0, tk, S, device=kk.device)
        ends = torch.clamp(starts + L, max=tk)
        means = (cs[ends] - cs[starts]) / (ends - starts).unsqueeze(1).to(cs.dtype)
        sizes = torch.clamp(starts + S, max=tk) - starts                       # keys in each S-group
        return means.to(kk.dtype).repeat_interleave(sizes, dim=0)[:tk]
    if kind == "pool":
        if r == 1:
            return kk
        pad = (-tk) % r
        x = torch.cat([kk, kk.new_zeros(pad, D)]) if pad else kk
        cnt = torch.full((x.shape[0] // r, 1), float(r), device=kk.device, dtype=kk.dtype)
        if pad:
            cnt[-1, 0] = r - pad
        return (x.view(-1, r, D).sum(1) / cnt).repeat_interleave(r, dim=0)[:tk]
    mu = kk.mean(0, keepdim=True)
    dev = kk - mu
    if kind == "channels":
        if qs is None:
            raise ValueError("channels:R needs the group's queries")
        if r >= D:
            return kk
        keep = (qs.abs().mean(0) * dev.abs().mean(0)).topk(r).indices
        out = mu.expand(tk, D).clone()
        out[:, keep] = kk[:, keep]
        return out
    if r >= D:
        return kk
    _evals, evecs = torch.linalg.eigh((dev.T @ dev).double())            # ascending eigenvalues
    P = evecs[:, -r:].to(kk.dtype)                                        # [D, r]
    return mu + (dev @ P) @ P.T


def _score_keys(kk, state, qs=None):
    """The keys the armed mass criterion scores: a --key_summary, a --key_bits copy, or the true keys."""
    if state.get("key_summary"):
        return _summarize_keys(kk, state["key_summary"], qs)
    return _quantize_keys(kk, state.get("key_bits"))


def _meanq_mass(qs, kk, D):
    """Per-key attention of the group-MEAN query, scaled to the group size. qs: [grp, D] the queries that share one
    KV head, kk: [tk, D]. This is what indexer_train.py distills from. The exact mass criterion instead sums each
    head's own softmax, so a key that a single head attends sharply survives there and is averaged away here."""
    return torch.softmax((qs.mean(0) @ kk.T) / math.sqrt(D), dim=-1) * qs.shape[0]


def _quest_scores(qs, kk, n_blocks, block, mode):
    """Quest's page criticality (Tang et al., ICML 2024) on the kernel's blocks: the per-channel minimum and maximum of
    a block's keys bound q.k over the block from above, sum_d max(q_d * kmin_d, q_d * kmax_d). Metadata: two vectors
    per block. qs [grp, D] the queries that share the KV head, kk [tk, D]. mode 'mean' bounds the group-mean query,
    'max' / 'sum' take the maximum / the sum over the group's heads of the per-head bounds. Returns [n_blocks]."""
    tk, D = kk.shape
    pad = n_blocks * block - tk
    lo = torch.cat([kk, kk.new_full((pad, D), float("inf"))]) if pad else kk
    hi = torch.cat([kk, kk.new_full((pad, D), -float("inf"))]) if pad else kk
    kmin = lo.view(n_blocks, block, D).min(1).values                    # [n_blocks, D]
    kmax = hi.view(n_blocks, block, D).max(1).values
    Q = qs.mean(0, keepdim=True) if mode == "mean" else qs                 # [g, D]
    b = torch.maximum(Q.unsqueeze(1) * kmin.unsqueeze(0), Q.unsqueeze(1) * kmax.unsqueeze(0)).sum(-1)   # [g, n_blocks]
    return b[0] if mode == "mean" else (b.max(0).values if mode == "max" else b.sum(0))


def _quest_bounds(kk, n_blocks, block):
    """Per-block, per-channel minimum and maximum of the keys (Quest's page metadata). kk [tk, D] -> ([n_blocks, D], ...)."""
    tk, D = kk.shape
    pad = n_blocks * block - tk
    lo = torch.cat([kk, kk.new_full((pad, D), float("inf"))]) if pad else kk
    hi = torch.cat([kk, kk.new_full((pad, D), -float("inf"))]) if pad else kk
    return lo.view(n_blocks, block, D).min(1).values, hi.view(n_blocks, block, D).max(1).values


def _quest_rows(Q, kmin, kmax, mode):
    """Quest's bound for many query positions at once, the same numbers _quest_scores gives one position:
    sum_d max(q_d kmin_d, q_d kmax_d) = relu(q) . kmax - relu(-q) . kmin. Q [k, grp, D] the positions' query heads of one
    KV group; mode 'mean' bounds the group-mean query, 'max' / 'sum' aggregate the per-head bounds. Returns [k, n_blocks]."""
    Q = Q.mean(1, keepdim=True) if mode == "mean" else Q
    b = torch.einsum("kgd,nd->kgn", Q.clamp(min=0), kmax) - torch.einsum("kgd,nd->kgn", (-Q).clamp(min=0), kmin)
    return b[:, 0] if mode == "mean" else (b.max(1).values if mode == "max" else b.sum(1))


def _stage1_block_scores(mod, sig, a, kw, mode, Hkv, grp, D):
    """Block scores from the SHIPPED compressed keys with a chosen aggregation over the query heads of each GQA group
    (decode only). Each query head's softmax over the compressed keys is computed in PyTorch (a flat re-implementation
    of stage 1; the kernel's second-level keys are not used), aggregated per group by 'sum' (the shipped rule), 'max'
    or 'top2', and max-pooled to blocks by the kernel's own pooling. Returns [Hkv, n_blocks] (forced blocks are +inf,
    as in the kernel)."""
    B = _bind_selector_args(sig, a, kw)
    if B is None:
        raise RuntimeError("--stage1_agg reached a prefill call")
    ck = B["k"].float()                                                  # [n_c, Hkv, D]
    qd = B["q"][0].float().view(Hkv, grp, D)
    sc = torch.softmax(torch.einsum("hgd,chd->hgc", qd, ck) / math.sqrt(D), dim=-1)      # [Hkv, grp, n_c]
    agg = sc.sum(1) if mode == "sum" else (sc.max(1).values if mode == "max" else sc.topk(2, dim=1).values.sum(1))
    bsc = mod.max_pooling_1d_varlen(agg.unsqueeze(1).to(torch.bfloat16).contiguous(), B["cu_seqlens_q"],
                                    B["cu_seqlens_k"], B["cache_lens"], B["max_seqlen_q"], B["max_seqlen_k"],
                                    local_blocks=B["local_blocks"], init_blocks=B["init_blocks"],
                                    block_size=B["block_size"], stride=B["kernel_stride"])
    return bsc[:, 0, :].float()


def _heads_mass(qs, kk, D):
    """The exact mass criterion: each query head's own softmax over the keys, summed over the group. qs [grp, D],
    kk [tk, D] (full precision, or a low-bit copy from _quantize_keys)."""
    return torch.softmax((qs @ kk.T) / math.sqrt(D), dim=-1).sum(0)


def _bind_selector_args(sig, a, kw):
    """Bind an intercepted compressed_attention call and normalize the two released
    signatures to one dict. Base InfLLM-V2 (MiniCPM4): (q, k, k2, ..., cu_seqlens_k, cu_seqlens_k2,
    ...); DashAttention-release InfLLMv2: (q, k, v, ..., cu_seqlens_k, ...) with v = k.clone()
    and no k2 (its stage-1 kernel takes cu_seqlens_v = cu_seqlens_k through fasa_compat's shim).
    Returns None at prefill. Pure Python (unit-tested); the kernels run in _kernel_block_scores."""
    ba = sig.bind(*a, **kw); ba.apply_defaults(); A = ba.arguments
    cache_lens = A.get("cache_lens")
    if cache_lens is None or bool((cache_lens == 0).all()):
        return None
    if "k2" in A:
        variant, second, cu_second = "base", A["k2"], A["cu_seqlens_k2"]
    elif "v" in A:
        variant, second, cu_second = "fasa", A["v"], None
    else:
        raise RuntimeError(f"unknown compressed_attention signature: {list(A)}")
    return {"variant": variant, "q": A["q"], "k": A["k"], "second": second, "cu_second": cu_second,
            "cu_seqlens_q": A["cu_seqlens_q"], "cu_seqlens_k": A["cu_seqlens_k"],
            "max_seqlen_q": A["max_seqlen_q"], "max_seqlen_k": A["max_seqlen_k"],
            "cache_lens": cache_lens, "block_size": A["block_size"], "kernel_stride": A["kernel_stride"],
            "init_blocks": A["init_blocks"], "local_blocks": A["local_blocks"]}


def _kernel_block_scores(mod, sig, a, kw):
    """Recompute the SHIPPED selector's own block scores from the arguments of the intercepted
    compressed_attention call -- stage-1 compressed attention max-pooled to blocks, with the
    kernel's +inf forcing of the init and local-window blocks -- for either released signature
    (see _bind_selector_args). Decode only (returns None at prefill). Audit fix 2026-09-16."""
    B = _bind_selector_args(sig, a, kw)
    if B is None:
        return None
    q_idx = B["cache_lens"] // B["block_size"]
    if B["variant"] == "base":
        score = mod.infllmv2_attn_stage1(B["q"].contiguous(), B["k"].contiguous(), B["second"].contiguous(),
                                         cu_seqlens_q=B["cu_seqlens_q"], cu_seqlens_k=B["cu_seqlens_k"],
                                         cu_seqlens_v=B["cu_second"], max_seqlen_q=B["max_seqlen_q"],
                                         max_seqlen_k=B["max_seqlen_k"], causal=False)
    else:                                   # fasa: v = k.clone(); the shim supplies cu_seqlens_v
        score = mod.infllmv2_attn_stage1(B["q"].contiguous(), B["k"].contiguous(), B["second"].contiguous(),
                                         cu_seqlens_q=B["cu_seqlens_q"], cu_seqlens_k=B["cu_seqlens_k"],
                                         max_seqlen_q=B["max_seqlen_q"], max_seqlen_k=B["max_seqlen_k"],
                                         causal=False)
    score = score[:, :q_idx.shape[0], :]
    bsc = mod.max_pooling_1d_varlen(score.contiguous(), B["cu_seqlens_q"], B["cu_seqlens_k"],
                                    B["cache_lens"], B["max_seqlen_q"], B["max_seqlen_k"],
                                    local_blocks=B["local_blocks"], init_blocks=B["init_blocks"],
                                    block_size=B["block_size"], stride=B["kernel_stride"])
    return bsc[:, 0, :].float(), int(q_idx[0])


def _forced_and_rank(bsc_h, n_blocks, q_idx):
    """From one head's kernel block scores: forced = blocks the kernel scores +inf (init block
    and local window), valid = blocks the kernel may select (<= the query's block), kf = the
    kernel's own ranking of the remaining valid blocks (the shipped free-slot order)."""
    n = min(n_blocks, bsc_h.shape[0])
    s = bsc_h[:n].clone()
    valid = torch.arange(n, device=s.device) <= q_idx
    forced = torch.nonzero(torch.isinf(s) & (s > 0) & valid).flatten().tolist()
    fset = set(forced)
    s = s.masked_fill(~valid, -float("inf"))
    if forced:
        s[torch.tensor(forced, device=s.device)] = -float("inf")
    order = s.argsort(descending=True).tolist()
    kf = [b for b in order if bool(valid[b]) and b not in fset]
    return forced, valid, kf


def _place_row(sel, h, order, forced, valid, nb, topk_idx, qi=0):
    """Write head h's selection row: the forced blocks first, then the arm's priority order
    (deduplicated, valid blocks only) until the shipped slot count nb is reached; sorted by
    block index like the kernel's own rows. With forced=[] and valid=None this reproduces the
    pre-fix rows exactly (--legacy_rows)."""
    merged, seen = list(forced), set(forced)
    for b in order:
        if len(merged) >= nb:
            break
        if b in seen or (valid is not None and not bool(valid[b])):
            continue
        merged.append(b); seen.add(b)
    idx = torch.tensor(sorted(merged), device=sel.device).to(topk_idx.dtype)
    sel[h, qi, :idx.numel()] = idx
    return merged


def _swap_k_rows(mc, ship, forced, valid, bidx, nb, n_blocks, m, dtype, stats):
    """Minimal-edit rows. Keep every shipped pick, then at most m times: take the heaviest block the shipped row missed and put it in an
    empty slot of that row, or, when the row is full, in place of the lightest shipped free pick when the scorer ranks it heavier.
    Forced blocks are never dropped. mc [rows, n_blocks] block masses, ship / forced / valid bool [rows, n_blocks].
    stats: [rows seen, blocks added, blocks swapped in]."""
    neg, pos = -float("inf"), float("inf")
    newmask = ship | (forced & valid)
    cap = (nb - newmask.sum(-1)).clamp(min=0)
    addable = valid & ~newmask
    droppable = newmask & ~forced
    m = min(m, n_blocks)
    add_v, add_i = mc.masked_fill(~addable, neg).topk(m, dim=-1)
    drop_v, drop_i = (-mc.masked_fill(~droppable, pos)).topk(m, dim=-1)
    drop_v = -drop_v                                           # the m lightest droppable picks, ascending
    stats[0] += mc.shape[0]
    for j in range(m):
        a, av = add_i[:, j:j + 1], add_v[:, j:j + 1]
        ok = av > neg
        fill = ok & (cap.unsqueeze(1) > 0)
        newmask.scatter_(1, a, newmask.gather(1, a) | fill)
        cap = cap - fill.squeeze(1).long()
        d, dv = drop_i[:, j:j + 1], drop_v[:, j:j + 1]
        swap = ok & ~fill & (dv < av) & (dv < pos) & droppable.gather(1, d)
        newmask.scatter_(1, d, newmask.gather(1, d) & ~swap)
        newmask.scatter_(1, a, newmask.gather(1, a) | swap)
        stats[1] += int(fill.sum()); stats[2] += int(swap.sum())
    n_pick = newmask.sum(-1)
    if int(n_pick.max()) > nb:
        raise ValueError(f"--swap_k produced a row of {int(n_pick.max())} blocks, wider than {nb}")
    r = torch.where(newmask, bidx.expand_as(newmask), torch.full_like(bidx.expand_as(newmask), n_blocks))
    idx = r.sort(-1).values[:, :nb]
    out = torch.full((mc.shape[0], nb), -1, dtype=dtype, device=mc.device)
    out[:, :idx.shape[1]] = idx.masked_fill(idx == n_blocks, -1).to(dtype)
    return out


def _prefill_tail_rows(state, topk_idx, q, k, block, init_blocks, local_blocks, li):
    """--prefill_tail K: at PREFILL, rewrite the selection rows of the last K query positions (the question) with the
    armed mass criterion (per-head exact mass, or the group-mean query, optionally on low-bit keys), so that the
    question tokens read the context through the repaired selector in the same single prefill pass. Forced blocks are
    the kernel's own structure (init blocks, local window, the query's block; pheno/free_region.forced_blocks), and
    every forced block must already be in the shipped row (violations are counted and abort the run). All other
    query positions, and the heads --kv_heads leaves out, keep their shipped rows. q [T, Hq, D], k [T, Hkv, D],
    topk_idx [Hkv, T, nb]. Returns the new index tensor."""
    from free_region import forced_blocks
    T, Hq, D = q.shape
    Hkv = k.shape[1]
    grp = Hq // Hkv
    K = min(int(state["prefill_tail"]), T)
    nb = topk_idx.shape[-1]
    n_blocks = math.ceil(T / block)
    pad = n_blocks * block - T
    sel = topk_idx.clone()
    pos = torch.arange(T - K, T, device=q.device)
    kidx = torch.arange(T, device=q.device)
    kvh = state.get("kv_heads")
    st = state.setdefault("prefill_tail_stats", [0, 0])              # rows rewritten, forced blocks missing from shipped rows
    # Many rows (--prefill_tail covering the prompt): the same masses, computed in the same 32-row chunks, but the rows are
    # built in batch instead of one Python iteration per row (unit-tested to give identical rows and counters).
    gate = state.get("row_gate")
    swap_k = state.get("swap_k")
    impl = state.get("prefill_rows_impl") or ("batched" if (K > 512 or gate is not None or swap_k) else "loop")
    if (gate is not None or swap_k) and impl != "batched":
        raise ValueError("--row_gate and --swap_k are implemented in the batched row builder only")
    bidx = torch.arange(n_blocks, device=q.device).view(1, n_blocks)
    for h in range(Hkv):
        if kvh is not None and (int(li), h) not in kvh:
            continue
        qt = q[T - K:, h * grp:(h + 1) * grp].float()                # [K, grp, D]
        kk = _score_keys(k[:, h].float(), state, qt.mean(0))         # 'channels' ranks with the tail's mean |q|
        quest = str(state.get("oracle_query")).startswith("quest_")
        if quest:                                                     # Quest's bound on the true keys, per query position
            kmin, kmax = _quest_bounds(k[:, h].float(), n_blocks, block)
            state["n_quest_tail_rows"] = state.get("n_quest_tail_rows", 0) + K
        if state.get("oracle_query") == "meanq":
            qt = qt.mean(1, keepdim=True)
        rows = []
        for c0 in range(0, K, 32):
            if quest:
                mc = _quest_rows(qt[c0:c0 + 32], kmin, kmax, state["oracle_query"][6:])   # future blocks masked below
            else:
                lg = torch.einsum("kgd,td->kgt", qt[c0:c0 + 32], kk) / math.sqrt(D)
                causal = kidx.view(1, 1, T) <= pos[c0:c0 + 32].view(-1, 1, 1)
                pm = torch.softmax(lg.masked_fill(~causal, -float("inf")), dim=-1).sum(1)          # [k, T]
                if pad:
                    pm = torch.cat([pm, pm.new_zeros(pm.shape[0], pad)], dim=-1)
                mc = pm.view(pm.shape[0], n_blocks, block).sum(-1)
            if impl == "batched":
                pc = pos[c0:c0 + 32]
                qb = (pc // block).view(-1, 1)
                valid = bidx <= qb
                forced = ((bidx < init_blocks) | ((bidx >= qb - local_blocks) & (bidx <= qb))) & valid
                ship = torch.zeros(pc.numel(), n_blocks + 1, dtype=torch.bool, device=q.device)
                srow = topk_idx[h, T - K + c0:T - K + c0 + pc.numel()].long()
                ship.scatter_(1, torch.where(srow >= 0, srow, torch.full_like(srow, n_blocks)), True)
                st[1] += int((forced & ~ship[:, :n_blocks]).sum())
                rank = mc.masked_fill(~valid, -float("inf")).masked_fill(forced, float("inf"))
                top = rank.topk(min(nb, n_blocks), dim=-1)
                idx = top.indices.masked_fill(top.values == -float("inf"), n_blocks).sort(-1).values
                out = torch.full((pc.numel(), nb), -1, dtype=topk_idx.dtype, device=q.device)
                out[:, :idx.shape[1]] = idx.masked_fill(idx == n_blocks, -1).to(topk_idx.dtype)
                if swap_k:
                    out = _swap_k_rows(mc, ship[:, :n_blocks], forced, valid, bidx, nb, n_blocks, int(swap_k), topk_idx.dtype,
                                       state.setdefault("swap_stats", [0, 0, 0]))
                if gate is not None:
                    # keep the shipped row unless it forfeits more than `gate` of the group's attention mass under the scorer
                    newm = torch.zeros(pc.numel(), n_blocks + 1, dtype=torch.bool, device=q.device)
                    newm.scatter_(1, idx, True)
                    if state.get("row_gate_mode", "total") == "peak":
                        short = (mc * (newm[:, :n_blocks] & ~ship[:, :n_blocks])).amax(-1) / qt.shape[1]
                    else:
                        short = ((mc * newm[:, :n_blocks]).sum(-1) - (mc * ship[:, :n_blocks]).sum(-1)) / qt.shape[1]
                    keep = short <= gate
                    out[keep] = topk_idx[h, T - K + c0:T - K + c0 + pc.numel()][keep]
                    gs = state.setdefault("row_gate_stats", [0, 0, 0, 0])      # decode rows seen / kept shipped, prefill rows seen / kept shipped
                    gs[2] += pc.numel(); gs[3] += int(keep.sum())
                sel[h, T - K + c0:T - K + c0 + pc.numel()] = out
                st[0] += pc.numel()
            else:
                rows.append(mc)
        if impl == "batched":
            if state.get("log_prefill_rows") is not None:
                state["log_prefill_rows"][(int(li), h)] = sel[h, T - K:].to("cpu", torch.int32)
            continue
        mass = torch.cat(rows, 0)                                     # [K, n_blocks]
        for i in range(K):
            p_i = int(pos[i])
            forced = forced_blocks(p_i, block, local_blocks, init_blocks)
            valid = torch.arange(n_blocks, device=q.device) <= p_i // block
            ship = set(topk_idx[h, T - K + i].tolist())
            st[1] += sum(1 for b in forced if b not in ship)
            sel[h, T - K + i, :] = -1
            _place_row(sel, h, mass[i].argsort(descending=True).tolist(), forced, valid, nb, topk_idx, qi=T - K + i)
            st[0] += 1
        if state.get("log_prefill_rows") is not None:            # bench_qk's gate compares these rows with the fast path's
            state["log_prefill_rows"][(int(li), h)] = sel[h, T - K:].to("cpu", torch.int32)
    return sel


def _dash_repair_prior(topk_idx, ps, state):
    """fasa Dash-8B: blocks outside the release's entmax support have a ZERO pooled score,
    which sparse_forward turns into a -inf routing prior (log 0) and the decode kernel then
    reads with weight 0 -- every block the harness injected outside the shipped support was
    selected but unreadable. Give each SELECTED block with a
    zero raw score the row's mean positive score, so its centred, sigma-divided prior is ~0
    like every in-support block. Returns the repaired prior; counts are recorded in state."""
    Hkv = topk_idx.shape[0]; Hq = ps.shape[0]; grp = Hq // Hkv
    ps2 = ps.clone()
    fixed = 0
    for h in range(Hkv):
        row = [b for b in topk_idx[h, 0].tolist() if 0 <= b < ps2.shape[-1]]
        if not row:
            continue
        bi = torch.tensor(row, device=ps2.device)
        for qh in range(h * grp, (h + 1) * grp):
            v = ps2[qh, 0]
            pos = v[v > 0]
            fill = pos.mean() if pos.numel() else torch.tensor(1.0 / 32, device=v.device, dtype=v.dtype)
            z = v[bi] <= 0
            if bool(z.any()):
                v[bi[z]] = fill.to(v.dtype); fixed += int(z.sum())
    st = state.setdefault("dash_prior_fixed", [0, 0]); st[0] += fixed; st[1] += 1
    return ps2


def _make_dash_ca(cfg, state, orig_ca):
    """fasa Dash-8B: compressed_attention returns (topk_idx, max_k, pooled_score)
    with ADAPTIVE per-row budgets (entmax support, score-ordered, -1 padded) at
    block_size 32. Decode-side interventions are matched per head to the shipped
    row's realized budget (the panel dash protocol, RESULTS 77.4). The dense arm
    overrides BOTH stages with the full causal support, which equals dense
    because the release's sigma (1e6 hardcoded) makes the routing bias ~1e-5
    (RESULTS 78.6). pooled_score is never touched."""
    import math as _m
    blk = cfg.sparse_config["block_size"]

    def override(topk_idx, max_k):
        dense = state.get("dash_dense")
        active = (state["oracle"] or state.get("learned") is not None
                  or state["random_sel"] or dense)
        if not active:
            return None
        qk = STASH.get("qk")
        if qk is None:
            return None
        q, k, _v = qk
        tq, Hq, D = q.shape
        Hkv = k.shape[1]
        grp = Hq // Hkv
        tk_len = k.shape[0]
        nbc = _m.ceil(tk_len / blk)
        if dense:
            Tq = topk_idx.shape[1]
            pos = torch.arange(tk_len - Tq, tk_len, device=q.device)
            cj = torch.arange(nbc, device=q.device)
            row = torch.where(cj.unsqueeze(0) <= (pos // blk).unsqueeze(1),
                              cj.unsqueeze(0).expand(Tq, nbc),
                              torch.full((Tq, nbc), -1, device=q.device,
                                         dtype=torch.long))
            out = (row.unsqueeze(0).expand(Hkv, Tq, nbc).contiguous()
                   .to(topk_idx.dtype))
            state["n_oracle_calls"] += 1
            return out, nbc
        if tq != 1:
            return None                                  # decode-only interventions
        nb_full = tk_len // blk
        if nb_full < 2:
            return None
        li_now = int(STASH.get("li", 0))
        kvh = state.get("kv_heads")
        if kvh is not None and not any(l == li_now for l, _ in kvh):
            return None                                  # --kv_heads: no listed head in this layer, shipped rows stay
        S = topk_idx.shape[-1]
        out = topk_idx.clone()
        mix = state.get("learned_mix")
        if state["random_sel"]:
            mix = mix or (32, 32)
        for h in range(Hkv):
            if kvh is not None and (li_now, h) not in kvh:
                hk = state.setdefault("kv_heads_kept", [0]); hk[0] += 1
                continue                                 # an unlisted head keeps its shipped row
            shipped = [b for b in topk_idx[h, 0].tolist() if b >= 0]
            B = len(shipped)
            if B == 0:
                continue
            if state["random_sel"]:
                sb = torch.rand(nb_full, device=q.device)
            else:
                qs = q[0, h * grp:(h + 1) * grp, :].float()
                kk = k[:, h, :].float()
                if state.get("learned") is not None:
                    li = STASH.get("li", 0)
                    LWq, LWk, ltau, LM1, Lm2, _Lwd = state["learned"][0]
                    prod = ((qs.mean(0) @ LWq[li, h]).unsqueeze(0)
                            * (kk[: nb_full * blk] @ LWk[li, h]))
                    lg6 = (torch.relu(prod @ LM1) @ Lm2) / ltau
                    sb = torch.logsumexp(lg6.view(nb_full, blk), -1)
                elif state.get("oracle_query") == "meanq":   # mean-query mass, optionally on low-bit keys
                    p = _meanq_mass(qs, _quantize_keys(kk, state.get("key_bits")), D)
                    sb = p[: nb_full * blk].view(nb_full, blk).sum(-1)
                elif state.get("key_bits") is not None:      # per-head criterion on a low-bit copy of the keys
                    p = _heads_mass(qs, _quantize_keys(kk, state["key_bits"]), D)
                    sb = p[: nb_full * blk].view(nb_full, blk).sum(-1)
                else:                                    # exact-mass oracle
                    lg = (qs @ kk.T) / _m.sqrt(D)
                    p = torch.softmax(lg, dim=-1).sum(0)
                    sb = p[: nb_full * blk].view(nb_full, blk).sum(-1)
            picks = sb.topk(min(B + (mix[1] if mix else 0), nb_full)
                            ).indices.tolist()
            lg_l = state.get("sel_log")
            if (lg_l is not None and state.get("sample_i", 1 << 30) < state["log_n"]
                    and tk_len % state.get("log_every", 4) == 0):
                lg_l.append({"sample": state["sample_i"], "tk": tk_len,
                             "li": int(STASH.get("li", 0)), "h": h,
                             "arm_picks": picks[:64]})
            # decode forcing: their code infs the last 1-2 block columns (current
            # partial chunk + last full block) before top-k, so they sit first in
            # the score-ordered shipped row. Retain them in EVERY arm -- dropping
            # them blinds attention to the just-generated tokens and degenerates
            # generation (the NOSA protocol lesson: the oracle keeps the blocks
            # the kernel structurally assumes; disclosed adaptation)
            forced = shipped[:2]
            keep = shipped[: mix[0]] if mix else []
            merged, seen = [], set()
            for b in forced + keep + picks + shipped:
                if b >= 0 and b not in seen:
                    merged.append(b); seen.add(b)
                if len(merged) == B:
                    break
            newrow = torch.full((S,), -1, dtype=topk_idx.dtype,
                                device=topk_idx.device)
            newrow[: len(merged)] = torch.tensor(merged, dtype=topk_idx.dtype,
                                                 device=topk_idx.device)
            out[h, 0] = newrow
        state["n_oracle_calls"] += 1
        return out, max_k

    def dash_ca(*a, **kw):
        topk_idx, mk, ps = orig_ca(*a, **kw)
        try:
            lg_l = state.get("sel_log")
            qk0 = STASH.get("qk")
            if (lg_l is not None and qk0 is not None and qk0[0].shape[0] == 1
                    and state.get("sample_i", 1 << 30) < state["log_n"]
                    and qk0[1].shape[0] % state.get("log_every", 4) == 0):
                lg_l.append({"sample": state["sample_i"],
                             "tk": int(qk0[1].shape[0]),
                             "li": int(STASH.get("li", 0)),
                             "shipped": [[b for b in topk_idx[h, 0].tolist() if b >= 0]
                                         for h in range(topk_idx.shape[0])]})
            r = override(topk_idx, mk)
            if r is not None:
                topk_idx, mk = r
                if not state.get("legacy_rows"):
                    ps = _dash_repair_prior(topk_idx, ps, state)
        except Exception as e:
            state["errors"].append(f"{type(e).__name__}: {e}")
        return topk_idx, mk, ps

    return dash_ca


def install_oracle(model, cfg, state):
    """Replace the selector's pooled estimate with the EXACT attention mass per block."""
    mod = sys.modules[type(model).__module__]
    block = cfg.sparse_config["block_size"]
    raw_ca = mod.compressed_attention
    sig_ca = inspect.signature(raw_ca)
    _tup = {"rest": None}

    def orig_ca(*a, **kw):
        # fasa-NSA's compressed_attention returns (topk_idx, comp_attn_output);
        # ca_inner works on the index tensor alone and ca repacks the untouched
        # extras on the way out. Tensor-returning variants pass through as before.
        out = raw_ca(*a, **kw)
        if isinstance(out, tuple):
            _tup["rest"] = out[1:]
            return out[0]
        _tup["rest"] = None
        return out
    if hasattr(mod, "triton_entmax"):                    # fasa Dash-8B variant
        if (state.get("layers") is not None and state.get("kv_heads") is None) or state.get("layer_budget") is not None \
                or state.get("key_summary") or state.get("prefill_tail") or state.get("stage1_agg") \
                or str(state.get("oracle_query")).startswith("quest_") \
                or (state.get("learned") and isinstance(state["learned"][0][0], str)):
            raise SystemExit("--layers, --layer_budget, --key_summary, --prefill_tail, --stage1_agg, the Quest bound and "
                             "--quant_indexer are not implemented for the Dash-8B hook (--kv_heads is) -- ABORT")
        mod.compressed_attention = _make_dash_ca(cfg, state, raw_ca)
        _install_stash(model)
        print("dash-8B adaptive-budget override armed", flush=True)
        return

    def ca_inner(*a, **kw):
        topk_idx = orig_ca(*a, **kw)
        if (not state["oracle"] and state.get("layer_band") is None
                and state.get("indexer") is None and state.get("salience") is None
                and state.get("learned") is None and not state.get("keep_only")):
            return topk_idx
        if not _layer_active(state.get("layers"), STASH.get("li", 0)):
            return topk_idx                          # --layers: this layer keeps the shipped selection
        qk = STASH.get("qk")
        if qk is None:
            return topk_idx
        q, k, v = qk                                # q:[tq,Hq,D]  k,v:[tk,Hkv,D] varlen
        try:
            tq, Hq, D = q.shape
            tk_len, Hkv, _ = k.shape
            if tq != 1:                             # prefill: untouched unless --prefill_tail rewrites the last rows
                if state.get("snapkv_window"):      # SnapKV: score the blocks once, from the observation window
                    W = min(int(state["snapkv_window"]), tq)
                    grp_ = Hq // Hkv
                    n_b = math.ceil(tk_len / block)
                    li_ = int(STASH.get("li", 0))
                    for h in range(Hkv):
                        state.setdefault("snapkv_scores", {})[(li_, h)] = snapkv_block_scores(
                            q[tq - W:, h * grp_:(h + 1) * grp_], k[:, h], tk_len, block, n_b)
                    state["snapkv_prefill_calls"] = state.get("snapkv_prefill_calls", 0) + 1
                    return topk_idx
                if not state.get("prefill_tail") or not state["oracle"]:
                    return topk_idx
                ba = sig_ca.bind(*a, **kw); ba.apply_defaults()
                return _prefill_tail_rows(state, topk_idx, q, k, block, int(ba.arguments["init_blocks"]),
                                          int(ba.arguments["local_blocks"]), STASH.get("li", 0))
            grp = Hq // Hkv
            nb = topk_idx.shape[-1]
            n_blocks = math.ceil(tk_len / block)
            sel = torch.full_like(topk_idx, -1)
            kinfo = None
            if state.get("forced_mode", "kernel") == "kernel":
                kinfo = _kernel_block_scores(mod, sig_ca, a, kw)
                if kinfo is None:                        # prefill: never intervened
                    return topk_idx
                if kinfo[0].shape[0] != Hkv:
                    raise RuntimeError(f"kernel block scores have {kinfo[0].shape[0]} heads, "
                                       f"expected {Hkv} kv heads")
            fst = state.setdefault("forced_stats", [0, 0, 0])
            kvh = state.get("kv_heads")
            s1 = None
            if state.get("stage1_agg"):
                s1 = _stage1_block_scores(mod, sig_ca, a, kw, state["stage1_agg"], Hkv, grp, D)
            for h in range(Hkv):
                if kvh is not None and (int(STASH.get("li", 0)), h) not in kvh:
                    sel[h] = topk_idx[h]                 # --kv_heads: every other KV head keeps its shipped row
                    hk = state.setdefault("kv_heads_kept", [0]); hk[0] += 1
                    continue
                if kinfo is not None:
                    forced, valid, kf = _forced_and_rank(kinfo[0][h], n_blocks, kinfo[1])
                else:
                    forced, valid, kf = [], None, None
                fst[0] += 1; fst[1] += len(forced); fst[2] += nb - len(forced)
                if s1 is not None:
                    # --stage1_agg: the shipped compressed keys, re-scored in PyTorch with another aggregation over the
                    # query heads of the GQA group. 'sum' is the shipped rule: its rows are compared with the shipped
                    # rows to measure how faithful this flat re-implementation of stage 1 is.
                    n1 = min(n_blocks, s1.shape[1])
                    sc1 = torch.full((n_blocks,), -float("inf"), device=q.device)
                    sc1[:n1] = s1[h, :n1]
                    merged = _place_row(sel, h, sc1.argsort(descending=True).tolist(), forced, valid, nb, topk_idx)
                    ship = set(b for b in topk_idx[h, 0].tolist() if b >= 0)
                    sm = state.setdefault("stage1_mismatch", [0, 0])
                    sm[0] += len(ship ^ set(merged)) // 2; sm[1] += len(ship)
                    continue
                if state.get("keep_only"):
                    # fidelity check: forced + the kernel's own free ranking must reproduce
                    # the shipped row (mismatch count recorded in the JSON)
                    merged = _place_row(sel, h, kf, forced, valid, nb, topk_idx)
                    ship = set(b for b in topk_idx[h, 0].tolist() if b >= 0)
                    km = state.setdefault("keeponly_mismatch", [0, 0])
                    km[0] += len(ship ^ set(merged)); km[1] += len(ship)
                    continue
                if state.get("snapkv_window"):
                    sk = state.get("snapkv_scores", {}).get((int(STASH.get("li", 0)), h))
                    if sk is None:
                        raise RuntimeError("SnapKV: no prefill scores for this head")
                    scr = torch.full((n_blocks,), -float("inf"), device=q.device)
                    n1 = min(n_blocks, sk.shape[0]); scr[:n1] = sk[:n1]
                    _place_row(sel, h, scr.argsort(descending=True).tolist(), forced, valid, nb, topk_idx)
                    state["snapkv_decode_rows"] = state.get("snapkv_decode_rows", 0) + 1
                    continue
                if state["random_sel"]:
                    _place_row(sel, h, torch.randperm(n_blocks, device=q.device).tolist(),
                               forced, valid, nb, topk_idx)
                    continue
                qs = q[0, h * grp:(h + 1) * grp, :].float()          # [grp, D]
                kk = k[:, h, :].float()                              # [tk_len, D]
                logits = (qs @ kk.T) / math.sqrt(D)                  # [grp, tk_len]
                pg = torch.softmax(logits, dim=-1)                   # [grp, tk_len]
                pad = n_blocks * block - tk_len
                if state["criterion"] == "output":
                    # ⭐ OUTPUT-CONTRIBUTION CRITERION. Every shipped selector ranks blocks by
                    # attention MASS and ignores V entirely. But the quantity that determines
                    # output error is the block's contribution to the attention OUTPUT:
                    #     score_b = || sum_{j in b} p_j v_j ||
                    # Two blocks with equal mass contribute differently when their value vectors
                    # differ in norm or cancel. the project notes shows a CRITERION gap that no better
                    # mass-estimator can close; this tests whether the criterion is simply the
                    # wrong quantity.
                    vv = v[:, h, :].float()                          # [tk_len, D]
                    if pad:
                        pg = torch.cat([pg, pg.new_zeros(grp, pad)], dim=-1)
                        vv = torch.cat([vv, vv.new_zeros(pad, D)], dim=0)
                    pb = pg.view(grp, n_blocks, block)
                    vb = vv.view(n_blocks, block, D)
                    contrib = torch.einsum('gbj,bjd->gbd', pb, vb)   # [grp, n_blocks, D]
                    mass = contrib.sum(0).norm(dim=-1)               # [n_blocks] output contribution
                elif state.get("salience") is not None:
                    li_now = STASH.get("li", 0)
                    if li_now <= state.get("_sal_last_li", -1):
                        state["_sal_counts"] = {}                     # new decode step -> reset
                    state["_sal_last_li"] = li_now
                    pad_n = n_blocks * block - tk_len
                    kk2 = torch.cat([kk, kk.new_zeros(pad_n, D)], 0) if pad_n else kk
                    kbar = kk2.view(n_blocks, block, D).mean(1)
                    cheap = kbar @ qs.mean(0)
                    kk_n = min(nb, n_blocks)
                    n_res = min(state["salience"], kk_n)
                    counts = state.setdefault("_sal_counts", {}).setdefault(h, {})
                    # propagated set: most-frequently-selected blocks in earlier layers this step
                    prop = sorted(counts, key=lambda b: -counts[b])[: n_res] if n_res else []
                    fresh = cheap.topk(kk_n).indices.tolist()
                    merged, seen = [], set()
                    for b in prop + fresh:
                        if b not in seen:
                            merged.append(b); seen.add(b)
                        if len(merged) == kk_n:
                            break
                    idx = torch.tensor(sorted(merged), device=q.device).to(topk_idx.dtype)
                    sel[h, 0, : idx.numel()] = idx
                    for b in merged:
                        counts[b] = counts.get(b, 0) + 1
                    continue
                elif state.get("layer_band") is not None:
                    bnd, ke, kl = state["layer_band"]
                    k_here = ke if STASH.get("li", 0) < bnd else kl
                    pad_n = n_blocks * block - tk_len
                    kk2 = torch.cat([kk, kk.new_zeros(pad_n, D)], 0) if pad_n else kk
                    kbar = kk2.view(n_blocks, block, D).mean(1)
                    cheap = kbar @ qs.mean(0)                        # deployable mean-pool score
                    kk_n = min(k_here, n_blocks, nb)
                    idx = cheap.topk(kk_n).indices.sort().values.to(topk_idx.dtype)
                    sel[h, 0, :kk_n] = idx                            # remaining slots stay -1
                    continue
                elif state.get("learned") is not None:
                    li_L = STASH.get("li", 0)
                    parts = []
                    for LWq, LWk, ltau, LM1, Lm2, Lwd in state["learned"]:
                        if isinstance(LWq, str):
                            # --quant_indexer: mean-query softmax mass over LWk-bit keys, summed per block
                            pq = _meanq_mass(qs, _quantize_keys(kk, LWk), D)
                            pad_q = n_blocks * block - tk_len
                            if pad_q:
                                pq = torch.cat([pq, pq.new_zeros(pad_q)])
                            parts.append(pq.view(n_blocks, block).sum(-1))
                            continue
                        if LM1 is not None:
                            # v6 nonlinear per-key scorer (shared MLP on projected product)
                            pad_v6 = n_blocks * block - tk_len
                            prod = (qs.mean(0) @ LWq[STASH.get("li", 0), h]).unsqueeze(0)                                    * (kk @ LWk[STASH.get("li", 0), h])
                            lg6 = (torch.relu(prod @ LM1) @ Lm2) / ltau
                            if pad_v6:
                                lg6 = torch.cat([lg6,
                                                 lg6.new_full((pad_v6,), -float("inf"))])
                            parts.append(torch.logsumexp(lg6.view(n_blocks, block), -1))
                            continue
                        if Lwd is not None:
                            # DSA/SpotAttention-form baseline: 4 relu subheads, learned weights
                            rL = LWq.shape[-1]
                            pad_d = n_blocks * block - tk_len
                            qp_d = (qs.mean(0) @ LWq[STASH.get("li", 0), h]).view(4, rL // 4)
                            kp_d = (kk @ LWk[STASH.get("li", 0), h]).view(-1, 4, rL // 4)
                            lgd = (torch.relu(torch.einsum('hd,thd->th', qp_d, kp_d))
                                   @ Lwd) / ltau
                            if pad_d:
                                lgd = torch.cat([lgd,
                                                 lgd.new_full((pad_d,), -float("inf"))])
                            parts.append(torch.logsumexp(lgd.view(n_blocks, block), -1))
                            continue
                        if state.get("qagg") == "persum":
                            # per-head: sum of per-head logsumexp block scores (oracle-like
                            # aggregation; the group-mean query blurs head-specific needles)
                            pad_L = n_blocks * block - tk_len
                            klo = kk @ LWk[li_L, h]                        # [tk, r]
                            acc = None
                            for g in range(grp):
                                lg_g = klo @ (qs[g] @ LWq[li_L, h]) / ltau
                                if pad_L:
                                    lg_g = torch.cat([lg_g,
                                                      lg_g.new_full((pad_L,), -float("inf"))])
                                s_g = torch.logsumexp(lg_g.view(n_blocks, block), -1)
                                acc = s_g if acc is None else acc + s_g
                            parts.append(acc / grp)
                        else:
                            parts.append(learned_block_scores(
                                qs, kk, LWq[li_L, h], LWk[li_L, h], ltau, n_blocks, block))
                    mass = torch.stack(parts).mean(0)
                    ema = state.get("score_ema")
                    if ema is not None:
                        # §60.2: smooth indexer scores across decode steps (sticky-SELECTION
                        # failed at 79.0 — persistence locks in wrong picks; score-EMA lets a
                        # hot block persist while wrong picks decay). Per (layer,head), reset
                        # on new sample via cache-length tracking.
                        key_e = (STASH.get("li", 0), h)
                        rec = state.setdefault("_ema", {}).get(key_e)
                        if rec is not None and rec[0] == tk_len - 1:
                            prev_m = rec[1]
                            if prev_m.shape[0] < n_blocks:      # context grew by a block
                                prev_m = torch.cat([prev_m,
                                                    mass[prev_m.shape[0]:].detach()])
                            mass = ema * mass + (1.0 - ema) * prev_m
                        state["_ema"][key_e] = (tk_len, mass.detach())
                    kk_n = min(nb, n_blocks)
                    mix = state.get("learned_mix")
                    cg = state.get("conf_gate")
                    if cg is not None and mix is not None:
                        # §61 CONFIDENCE GATE (aggregate-safe union): the indexer gets its
                        # slots only when its score distribution is peaked — z of its top
                        # picks vs the field. At 32K needles it is confident (why it wins);
                        # at 64K its scores flatten to noise (why it loses, §58.12). Below
                        # threshold: pure mean-pool (do no harm).
                        btop = mass.topk(min(mix[1], n_blocks)).values
                        z = float((btop.mean() - mass.mean()) / mass.std().clamp(min=1e-6))
                        if state.get("conf_soft"):
                            # continuous slot allocation: confidence smoothly buys slots
                            g = 1.0 / (1.0 + math.exp(-(z - cg) / state["conf_soft"]))
                            b_soft = int(round(g * mix[1]))
                            mix = (nb - b_soft, b_soft)
                        elif z < cg:
                            mix = (nb, 0)
                    arb = state.get("arbiter")
                    if arb is not None and mix is not None:
                        # §61.2 learned continuous arbiter (self-supervised marginal-mass gate)
                        import arbiter_train as AT
                        mlp, mu, sd, qdep = arb
                        pad_u2 = n_blocks * block - tk_len
                        kk3 = torch.cat([kk, kk.new_zeros(pad_u2, D)], 0) if pad_u2 else kk
                        mpr = (kk3.view(n_blocks, block, D).mean(1)
                               @ qs.mean(0)).argsort(descending=True)
                        ipick = mass.topk(min(mix[1], n_blocks)).indices
                        x = torch.tensor(AT.features(mass, mpr, ipick, mix[1], n_blocks,
                                                     STASH.get("li", 0)),
                                         device=q.device, dtype=torch.float32)
                        if qdep:
                            g = float(torch.sigmoid(mlp(((x - mu) / sd).unsqueeze(0),
                                                        qs.mean(0).unsqueeze(0))))
                        else:
                            g = float(torch.sigmoid(mlp((x - mu) / sd)))
                        b_soft = int(round(g * mix[1]))
                        mix = (nb - b_soft, b_soft)
                    mb = state.get("mix_bands")
                    if mb is not None:
                        # per-layer mix: (boundary, amp_early, bidx_early, amp_late, bidx_late)
                        bnd_m, a1, b1, a2, b2 = mb
                        mix = (a1, b1) if STASH.get("li", 0) < bnd_m else (a2, b2)
                    if mix is not None:
                        # UNION arm: mean-pool top-a ∪ indexer top-b at the SAME total budget,
                        # remaining slots refilled from the mean-pool ranking. Tests whether the
                        # trained indexer surfaces DIFFERENT decisive blocks (complementarity)
                        # even though alone it scores below mean-pool.
                        amp, bidx = mix
                        if kf is not None:
                            mp_rank = kf                 # the shipped selector's OWN ranking
                        else:                            # --legacy_rows: block-64 mean-pool proxy
                            pad_u = n_blocks * block - tk_len
                            kk2 = torch.cat([kk, kk.new_zeros(pad_u, D)], 0) if pad_u else kk
                            mp_rank = (kk2.view(n_blocks, block, D).mean(1)
                                       @ qs.mean(0)).argsort(descending=True).tolist()
                        idx_rank = mass.argsort(descending=True).tolist()
                        prev = []
                        if state.get("sticky"):
                            # §60 autopsy: the union FINDS the needle then LOSES it mid-copy —
                            # refill spare slots from the PREVIOUS decode step's selection
                            # (per (layer,head), reset on new sample via cache-length tracking)
                            key_s = (STASH.get("li", 0), h)
                            rec = state.setdefault("_sticky", {}).get(key_s)
                            if rec is not None and rec[0] == tk_len - 1:
                                prev = rec[1]
                        merged = _place_row(sel, h, mp_rank[:amp] + idx_rank[:bidx] + prev + mp_rank[amp:],
                                            forced, valid, nb, topk_idx)
                        if state.get("sticky"):
                            state["_sticky"][(STASH.get("li", 0), h)] = (tk_len, merged)
                        continue
                    _place_row(sel, h, mass.argsort(descending=True).tolist(), forced, valid, nb, topk_idx)
                    continue
                elif state.get("indexer") is not None:
                    # TRAINING-FREE INDEXER: project q and EVERY key onto the top-r query PCs,
                    # rank blocks by logsumexp of projected per-key scores. Per-key structure is
                    # preserved (no within-block smoothing), so the decisive delta-scale
                    # distinctions survive -- the failure mode of every summary (§50).
                    P = state["indexer"].get((STASH.get("li", 0), h))
                    if P is None:
                        p = pg.sum(0)
                    else:
                        P = P[: state["indexer_r"]].to(k.device).float()
                        qp = P @ qs.mean(0)
                        lg = torch.einsum('r,ldr->ld' if False else 'r,lr->l', qp,
                                          (kk @ P.T)) / math.sqrt(D)
                        if pad:
                            lg = torch.cat([lg, lg.new_full((pad,), -float('inf'))])
                        mass = torch.logsumexp(lg.view(n_blocks, block), dim=-1)
                        kk_n = min(nb, n_blocks)
                        idx = mass.topk(kk_n).indices.sort().values.to(topk_idx.dtype)
                        sel[h, 0, :kk_n] = idx
                        continue
                    mass = p.view(n_blocks, block).sum(-1)
                    kk_n = min(nb, n_blocks)
                    idx = mass.topk(kk_n).indices.sort().values.to(topk_idx.dtype)
                    sel[h, 0, :kk_n] = idx
                    continue
                else:
                    if str(state.get("oracle_query")).startswith("quest_"):
                        # baseline: Quest's min/max page bound ranks the free slots (block scores, no per-key mass)
                        mass = _quest_scores(qs, kk, n_blocks, block, state["oracle_query"][6:])
                        _place_row(sel, h, mass.argsort(descending=True).tolist(), forced, valid, nb, topk_idx)
                        continue
                    if state.get("oracle_query") == "meanq":
                        # the indexer's distillation target; with --key_bits, scored on a low-bit copy of the keys
                        p = _meanq_mass(qs, _score_keys(kk, state, qs), D)
                    elif state.get("key_bits") is not None or state.get("key_summary"):
                        # the exact criterion (every head's own softmax, summed) on a low-bit copy or a summary of the keys
                        p = _heads_mass(qs, _score_keys(kk, state, qs), D)
                    else:
                        p = pg.sum(0)                                # GQA group mass
                    if pad:
                        p = torch.cat([p, p.new_zeros(pad)])
                    mass = p.view(n_blocks, block).sum(-1)           # [n_blocks] TRUE mass
                if (state.get("sel_log") is not None
                        and state.get("sample_i", 1 << 30) < state["log_n"]
                        and tk_len % state.get("log_every", 4) == 0):
                    # LongCat Fig-3 replication on this checkpoint: what fraction of THIS
                    # layer's true mass do the PREVIOUS layer's top-k picks capture, vs its
                    # own top-k (their model: ~57% shared picks, ~93% mass retained)
                    li_m = int(STASH.get("li", 0))
                    prev = state.get("_ml_prev")
                    tot = float(mass.sum().clamp(min=1e-9))
                    kk_m = min(nb, n_blocks)
                    own_idx = mass.topk(kk_m).indices
                    if prev is not None and prev[0] == tk_len and prev[1] == li_m - 1 \
                            and h in prev[2]:
                        pi = prev[2][h]
                        pi = pi[pi < n_blocks]
                        state["mass_log"].append({
                            "sample": state["sample_i"], "tk": int(tk_len), "li": li_m,
                            "h": h,
                            "own": float(mass[own_idx].sum()) / tot,
                            "prev": float(mass[pi].sum()) / tot,
                            "pick_ov": len(set(own_idx.tolist()) & set(pi.tolist()))
                                       / max(kk_m, 1)})
                    state.setdefault("_ml_cur", {})[h] = own_idx
                if state["shortlist"] > 0:
                    # ---- TWO-STAGE. Stage 1: cheap mean-pool shortlist of size m (what a real
                    # system can afford). Stage 2: EXACT mass, but only within that shortlist.
                    # m = k reproduces mean-pool; m = n_blocks reproduces the oracle. So sweeping m
                    # traces the recovery curve of the ESTIMATION gap (§44.3: the dominant
                    # recoverable term, +22.5 on MK3). Cost is m/k x the selection KV reads;
                    # attention compute is unchanged at k. NOT free -- the point is the exchange rate.
                    kpad = torch.cat([kk, kk.new_zeros(n_blocks * block - tk_len, D)], 0) \
                        if (n_blocks * block - tk_len) else kk
                    kbar = kpad.view(n_blocks, block, D).mean(1)          # [n_blocks, D]
                    cheap = kbar @ qs.mean(0)                             # mean-pool score
                    m = min(max(state["shortlist"], nb), n_blocks)
                    cand = cheap.topk(m).indices
                    keep = torch.full_like(mass, -float("inf"))
                    keep[cand] = mass[cand]
                    mass = keep
                _place_row(sel, h, mass.argsort(descending=True).tolist(), forced, valid, nb, topk_idx)
                if state.get("row_gate") is not None:
                    shp, new = topk_idx[h, 0], sel[h, 0]
                    g_eff = 1 if state.get("oracle_query") == "meanq" else qs.shape[0]
                    if state.get("row_gate_mode", "total") == "peak":
                        added = [b for b in new[new >= 0].tolist() if b not in set(shp[shp >= 0].tolist())]
                        short = (float(mass[added].max()) / g_eff) if added else 0.0
                    else:
                        short = float(mass[new[new >= 0].long()].sum() - mass[shp[shp >= 0].long()].sum()) / g_eff
                    gs = state.setdefault("row_gate_stats", [0, 0, 0, 0])
                    gs[0] += 1
                    if short <= state["row_gate"]:
                        sel[h, 0] = shp; gs[1] += 1
            if state.get("_ml_cur"):
                state["_ml_prev"] = (tk_len, int(STASH.get("li", 0)),
                                     state.pop("_ml_cur"))
            state["n_oracle_calls"] += 1
            return sel
        except Exception as e:
            state["errors"].append(f"{type(e).__name__}: {e}")
            return topk_idx

    def ca(*a, **kw):
        # CROSS-LAYER wrapper (LongCat CLI / GLM IndexShare premise, 2026-08-19).
        # reuse_k > 1: layers with li % k != 0 return the owner layer's FINAL
        # selection at the same decode step (scoring skipped there entirely —
        # the measured arm is exactly the deployed reuse). sel_log: per-step
        # per-layer pick dumps for the overlap-vs-layer-distance analysis
        # (pheno/xlayer_overlap.py). Both decode-only; prefill passes through.
        qk0 = STASH.get("qk")
        li_now = STASH.get("li", 0)
        decode = qk0 is not None and qk0[0].shape[0] == 1
        tk_now = int(qk0[1].shape[0]) if qk0 is not None else -1
        rk = state.get("reuse_k") or 0
        if rk > 1 and decode and li_now % rk:
            rec = state.get("_xl")
            if rec is not None and rec[0] == tk_now and rec[1] == li_now - (li_now % rk):
                state["n_reused"] = state.get("n_reused", 0) + 1
                # tuple variants: reuse would pair a stale comp branch; only the
                # protagonist (tensor return) runs reuse experiments
                return rec[2] if _tup["rest"] is None else (rec[2],) + tuple(_tup["rest"])
        out = ca_inner(*a, **kw)
        if rk > 1 and decode and li_now % rk == 0:
            state["_xl"] = (tk_now, li_now, out)
        lg = state.get("sel_log")
        if (lg is not None and decode and state.get("sample_i", 1 << 30) < state["log_n"]
                and tk_now % state.get("log_every", 4) == 0):
            try:
                lg.append({"sample": state["sample_i"], "tk": tk_now, "li": int(li_now),
                           "picks": out[:, 0, :].tolist()})
            except Exception:
                pass
        rest = _tup["rest"]
        return out if rest is None else (out,) + tuple(rest)
    mod.compressed_attention = ca
    _install_stash(model)


_DASH_DENSE_DECODE = {"on": False, "calls": 0}


def _install_stash(model):
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if not hasattr(attn, "sparse_forward") and not hasattr(attn, "_sparse_attention_forward"):
            # a hybrid model's layer without a block-sparse path (MiniCPM-SALA's lightning attention): nothing to hook.
            # The hook-count gates count only the layers with a sparse path (_sparse_layer_ids), so an all-sparse model
            # that lost its hook points still aborts there.
            STASH.setdefault("unhooked_layers", []).append(li)
            continue
        if hasattr(attn, "sparse_forward"):
            orig_sf = attn.sparse_forward

            def make(orig_sf, li, attn):
                def sf(q, k, v, csq, csk, msq, msk, *a, **kw):
                    STASH["qk"] = (q, k, v); STASH["li"] = li
                    if q.shape[0] == 1 and k.shape[0] > 1:     # one unpadded query row = a decode-path forward; shapes,
                        STASH["n_decode_sf"] = STASH.get("n_decode_sf", 0) + 1   # because msq/msk are tensors in the fasa releases
                    dw = STASH.get("decode_topk")
                    # msq/msk are ints on the base InfLLM-V2 attention (the only one --layer_budget arms) but tensors in the
                    # fasa releases, so the decode test is evaluated only when a budget is armed (short-circuit kept)
                    is_dec = (dw is not None) and bool(msq == 1 and msk > 1)
                    if dw is not None and (is_dec or STASH.get("budget_every")):
                        # --layer_budget: this layer's DECODE rows get their own width, and with --layer_budget_every its
                        # prefill rows too (the every-token budget control in the flagged layers). The
                        # shipped kernel selects as always (forced blocks plus its own top free blocks); without the
                        # every-token flag the sparse prefill keeps the shipped width, so only decode traffic moves
                        # between layers.
                        shipped_w = attn.topk
                        attn.topk = dw.get(li, shipped_w)
                        bst = STASH.setdefault("budget_stats", [0, 0])
                        bst[0] += 1; bst[1] += attn.topk              # mean row width over the calls of EVERY layer
                        if not is_dec and li in dw:
                            STASH["budget_prefill_calls"] = STASH.get("budget_prefill_calls", 0) + 1   # widened prefill calls
                        try:
                            return orig_sf(q, k, v, csq, csk, msq, msk, *a, **kw)
                        finally:
                            attn.topk = shipped_w
                    return orig_sf(q, k, v, csq, csk, msq, msk, *a, **kw)
                return sf
            attn.sparse_forward = make(orig_sf, li, attn)
            if _DASH_DENSE_DECODE.get("on"):
                orig_sf2 = attn.sparse_forward

                def make_dd(orig_sf2):
                    def sfdd(q, k, v, *ar, **kw):
                        if q.shape[0] != 1:            # prefill: shipped sparse
                            return orig_sf2(q, k, v, *ar, **kw)
                        import math as _m
                        _DASH_DENSE_DECODE["calls"] = _DASH_DENSE_DECODE.get("calls", 0) + 1
                        Hq, D = q.shape[1], q.shape[2]
                        grp = Hq // k.shape[1]
                        kh = k.repeat_interleave(grp, 1).float()
                        vh = v.repeat_interleave(grp, 1).float()
                        sc = torch.einsum('hd,khd->hk', q[0].float(), kh) / _m.sqrt(D)
                        pr = torch.softmax(sc, dim=-1)
                        out = torch.einsum('hk,khd->hd', pr, vh)
                        return out.unsqueeze(0).to(q.dtype)
                    return sfdd
                attn.sparse_forward = make_dd(orig_sf2)
        else:
            # fasa SparseLlama: batched [b, s, H, D] layout at
            # _sparse_attention_forward; bsz is 1 throughout this harness, so
            # the [0] views restore the varlen [t, H, D] stash contract
            orig_sf = attn._sparse_attention_forward

            def make_f(orig_sf, li):
                def sf(q, k, v, *ar, **kw):
                    STASH["qk"] = (q[0], k[0], v[0]); STASH["li"] = li
                    return orig_sf(q, k, v, *ar, **kw)
                return sf
            attn._sparse_attention_forward = make_f(orig_sf, li)


def snapkv_block_scores(qw, kk, T, block, n_blocks, bias=None):
    """SnapKV (Li et al., 2024) at block granularity: the attention that the last W prompt tokens (the observation window)
    put on each block, summed over the window and the query heads of the group. qw [W, G, D] the window's queries at
    positions T-W..T-1, kk [T, D], bias [T] or None an additive logit bias per key. Returns [n_blocks] float32."""
    W, G, D = qw.shape
    pos = torch.arange(T - W, T, device=kk.device).view(W, 1, 1)
    lg = torch.einsum("wgd,td->wgt", qw.float(), kk.float()) / math.sqrt(D)
    if bias is not None:
        lg = lg + bias.float().view(1, 1, -1)
    lg = lg.masked_fill(torch.arange(T, device=kk.device).view(1, 1, T) > pos, -float("inf"))
    p = torch.softmax(lg, dim=-1).sum((0, 1))                                  # [T]
    pad = n_blocks * block - T
    if pad:
        p = torch.cat([p, p.new_zeros(pad)])
    return p.view(n_blocks, block).sum(-1)


def install_dense_heads(model, heads, stats, phase="all"):
    """--dense_heads: the listed KV heads attend to EVERY key (full attention, as DuoAttention gives its retrieval heads),
    the other heads keep the release's selection. Each layer's sparse_forward runs as released, then the output of the
    listed heads' query heads is replaced by exact causal attention from flash_attn_varlen_func on the same q, k, v, which
    is what the release's dense mode computes. heads: set of (layer, kv_head). stats: dict of counters. phase "prefill"
    replaces the output only in the prefill forward and leaves every decode forward to the heads' own selection (the hybrid:
    full attention where the prefill is compute-bound, the arm's selection where decoding reads the cache).
    fasa NSA-8B (gate_proj on the attention: g0 * compressed branch + g1 * selected branch, no dense path) is spliced
    one level down instead: only the selected branch of the listed heads reads every key, and the compressed branch and
    the gates stay, so the listed heads select every block, the fix pheno/fixgain_probe_infllm.py linearizes and the
    model's own full attention (--topk 100000) in every head (2026-10-04)."""
    import inspect
    from flash_attn import flash_attn_varlen_func
    if hasattr(model.model.layers[0].self_attn, "gate_proj"):
        return _install_select_all_heads_nsa(model, heads, stats, phase, flash_attn_varlen_func)
    for li, layer in enumerate(model.model.layers):
        hs = sorted(h for (l_, h) in heads if l_ == li)
        attn = layer.self_attn
        if not hs:
            continue
        if not hasattr(attn, "sparse_forward"):
            raise SystemExit(f"--dense_heads: layer {li} has no sparse_forward -- ABORT")
        orig = attn.sparse_forward
        # the fasa releases (the InfLLMv2 retraining) pass the rotary table `cis` before the lengths, (q, k, v, cis,
        # cu_seqlens_q, cu_seqlens_k, max_q, max_k); read the class's own signature, since the instance method may already
        # be a hook's wrapper with a generic one
        cls_params = list(inspect.signature(type(attn).sparse_forward).parameters)
        shift = 1 if "cis" in cls_params[:5] else 0

        def make(orig, hs, li, shift):
            def sf(q, k, v, *a, **kw):
                out = orig(q, k, v, *a, **kw)
                csq, csk, msq, msk = a[shift:shift + 4]
                if phase == "prefill" and int(msq) == 1:
                    stats["decode_calls_selected"] = stats.get("decode_calls_selected", 0) + 1
                    return out
                Hq, Hkv = q.shape[1], k.shape[1]
                grp = Hq // Hkv
                causal = int(msq) != 1
                for h in hs:
                    o = flash_attn_varlen_func(q[:, h * grp:(h + 1) * grp].contiguous(), k[:, h:h + 1].contiguous(),
                                               v[:, h:h + 1].contiguous(), csq, csk, int(msq), int(msk),
                                               dropout_p=0.0, causal=causal)
                    out[:, h * grp:(h + 1) * grp] = o.to(out.dtype)
                stats["calls"] = stats.get("calls", 0) + 1
                stats["heads"] = stats.get("heads", 0) + len(hs)
                stats["decode_calls" if int(msq) == 1 else "prefill_calls"] = \
                    stats.get("decode_calls" if int(msq) == 1 else "prefill_calls", 0) + 1
                return out
            return sf
        attn.sparse_forward = make(orig, hs, li, shift)


def _install_select_all_heads_nsa(model, heads, stats, phase, flash_attn_varlen_func):
    """--dense_heads on fasa NSA-8B: the release's sparse_forward calls its module's infllmv2_attn_varlen_func once, for
    the selected branch; inside a layer with listed heads, that call's output for the listed heads' query heads is
    replaced by exact attention over every key (causal at prefill, every cached key at decode), and the compressed branch
    and the gates of sparse_forward act on it unchanged."""
    mod = sys.modules[type(model.model.layers[0].self_attn).__module__]
    orig_sel = mod.infllmv2_attn_varlen_func
    ctx = {"hs": None, "calls": 0}

    def sel(q, k, v, cu_q, cu_k, mq, mk, *ar, **kw):
        out = orig_sel(q, k, v, cu_q, cu_k, mq, mk, *ar, **kw)
        hs = ctx["hs"]
        if not hs:
            return out
        if isinstance(out, tuple):
            raise SystemExit("--dense_heads (NSA): the selected branch returned a tuple -- ABORT")
        ctx["calls"] += 1
        grp = q.shape[1] // k.shape[1]
        causal = int(mq) != 1
        for h in hs:
            o = flash_attn_varlen_func(q[:, h * grp:(h + 1) * grp].contiguous(), k[:, h:h + 1].contiguous(),
                                       v[:, h:h + 1].contiguous(), cu_q, cu_k, int(mq), int(mk),
                                       dropout_p=0.0, causal=causal)
            out[:, h * grp:(h + 1) * grp] = o.to(out.dtype)
        return out
    mod.infllmv2_attn_varlen_func = sel
    for li, layer in enumerate(model.model.layers):
        hs = sorted(h for (l_, h) in heads if l_ == li)
        if not hs:
            continue
        attn = layer.self_attn
        orig = attn.sparse_forward

        def make(orig, hs):
            def sf(q, k, v, cis, csq, csk, msq, msk, *a, **kw):
                if phase == "prefill" and int(msq) == 1:
                    stats["decode_calls_selected"] = stats.get("decode_calls_selected", 0) + 1
                    return orig(q, k, v, cis, csq, csk, msq, msk, *a, **kw)
                before = ctx["calls"]
                ctx["hs"] = hs
                try:
                    out = orig(q, k, v, cis, csq, csk, msq, msk, *a, **kw)
                finally:
                    ctx["hs"] = None
                if ctx["calls"] != before + 1:
                    raise SystemExit(f"--dense_heads (NSA): the selected branch ran {ctx['calls'] - before} times in one "
                                     "sparse_forward (want 1) -- ABORT")
                stats["calls"] = stats.get("calls", 0) + 1
                stats["heads"] = stats.get("heads", 0) + len(hs)
                stats["decode_calls" if int(msq) == 1 else "prefill_calls"] = \
                    stats.get("decode_calls" if int(msq) == 1 else "prefill_calls", 0) + 1
                stats["nsa_selected_branch_only"] = True
                return out
            return sf
        attn.sparse_forward = make(orig, hs)


def build_parser():
    """Every command-line flag of the harness. Shared with other benchmark runners."""
    p = argparse.ArgumentParser()
    for f in ("--model_dir", "--data", "--task", "--ruler_root", "--out"):
        p.add_argument(f, required=True)
    p.add_argument("--target_len", type=int, required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max_new", default="canonical")
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--salience", type=int, default=None,
                   help="SELECTION-CONSISTENCY PROPAGATION. Reserve this many of the k slots for "
                        "blocks most-selected by EARLIER LAYERS in the same decode step "
                        "(frequency-ranked union), fresh mean-pool fills the rest. Targets the "
                        "JOINT: 50/51 show marginal recall and per-cell gold-hit are both at "
                        "ceiling, so the oracle's +24 must come from cross-layer consistency. "
                        "salience=0 reduces to plain mean-pool (the gate). Same total budget.")
    p.add_argument("--layer_band", default=None,
                   help="'boundary,k_early,k_late' e.g. '16,16,112'. LAYER-BAND BUDGET from §43's "
                        "retrieval map: layers 0-15 are near-blind (gold-hit ~0.2 even by EXACT "
                        "mass) while L16-26 does the retrieval, so uniform k misallocates. "
                        "Selection uses CHEAP mean-pool (deployable), slot width = --topk, unused "
                        "slots = -1. Set boundary,k,k as a UNIFORM control that must reproduce "
                        "mean-pool. Opposite direction from PyramidKV -- that is the point.")
    p.add_argument("--indexer_pcs", default=None,
                   help="path to .qpcs.pt (per layer_head query principal directions from "
                        "qacp_probe calibration). Selection scores every key at rank r and ranks "
                        "blocks by projected logsumexp mass -- the TRAINING-FREE INDEXER. Cost "
                        "model: r/D of full-key traffic. d_eff~2 (RESULTS 48) is the license.")
    p.add_argument("--indexer_r", type=int, default=8)
    p.add_argument("--learned_indexer", default=None,
                   help="path to indexer.pt from pheno/indexer_train.py: the TRAINED rank-r "
                        "per-(layer,kv-head) indexer replaces decode block selection entirely. "
                        "Scoring matches training exactly (group-mean q, (qWq)·(kWk)/τ, no √D). "
                        "Combine with --no_oracle. Gates: a random-init indexer.pt must land near "
                        "the random-selection floor; window on MK3@32K is mean-pool 63-67, "
                        "exact-mass oracle 87.")
    p.add_argument("--qagg", choices=["mean", "persum"], default="mean",
                   help="indexer query aggregation: mean = group-mean query (training-matched); "
                        "persum = per-head logsumexp summed (oracle-like; §62)")
    p.add_argument("--learned_arbiter", default=None,
                   help="arbiter.pt from arbiter_train.py: learned continuous indexer-slot "
                        "share per (layer,head,step). Supersedes --conf_gate.")
    p.add_argument("--conf_soft", type=float, default=None,
                   help="with --conf_gate: sigmoid temperature for CONTINUOUS slot allocation "
                        "(indexer slots = round(sigmoid((z-conf_gate)/conf_soft) * b)). "
                        "Omit for the binary gate.")
    p.add_argument("--conf_gate", type=float, default=None,
                   help="with --learned_mix: indexer slots granted only when its top-score "
                        "z-margin exceeds this; else pure mean-pool. Aggregate-safe union "
                        "(61): keeps 32K gains, removes 64K losses.")
    p.add_argument("--score_ema", type=float, default=None,
                   help="with --learned_indexer: EMA coefficient for indexer block scores "
                        "across decode steps (0.5 = half new, half history). §60.2's gentler "
                        "fix for mid-copy selection instability.")
    p.add_argument("--sticky", action="store_true",
                   help="with --learned_mix: refill spare union slots from the PREVIOUS decode "
                        "step's selection instead of mean-pool tail ranks. Targets the §60 "
                        "mid-copy selection-instability failure (union finds the needle, loses "
                        "it partway through emitting the answer).")
    p.add_argument("--mix_bands", default=None,
                   help="'b,amp1,bidx1,amp2,bidx2' with --learned_indexer: layers <b use mix "
                        "(amp1,bidx1), layers >=b use (amp2,bidx2). §52.5: L0-7 waste budget, "
                        "L16-23 load-bearing — concentrate indexer slots where selection matters.")
    p.add_argument("--learned_mix", default=None,
                   help="'a,b' with --learned_indexer: mean-pool top-a UNION indexer top-b at "
                        "the same total slot budget, rest refilled by mean-pool. Complementarity "
                        "probe: union > mean-pool alone means the indexer surfaces different "
                        "decisive blocks even if it loses head-to-head.")
    p.add_argument("--shortlist", type=int, default=0,
                   help="TWO-STAGE selection: 0=off (pure oracle). If >0, shortlist the top-m "
                        "blocks by cheap mean-pool, then take the true top-k by EXACT mass within "
                        "the shortlist. m=k -> mean-pool, m=n_blocks -> oracle.")
    p.add_argument("--criterion", choices=["mass", "output"], default="mass",
                   help="mass = sum_j p_j (what Quest/COBS/HiLS/InfLLM-V2 all approximate). "
                        "output = ||sum_j p_j v_j||, the block's actual contribution to the "
                        "attention output. Tests whether §35's CRITERION gap is because mass is "
                        "the wrong quantity to rank by.")
    p.add_argument("--dash_dense_decode", action="store_true",
                   help="fasa Dash-8B forcing rung: SPARSE prefill (shipped), DENSE "
                        "decode -- every decode step attends the full sparse-built "
                        "cache. Splits the deficit into prefill-cache damage vs "
                        "decode-side reading (the NOSA forcing-ladder instrument)")
    p.add_argument("--no_install", action="store_true",
                   help="artifact check: install NO patches at all (no stash, no "
                        "selection wrapper) -- the pure released model")
    p.add_argument("--dash_fa", action="store_true",
                   help="fasa Dash-8B dense arm: load the release's own DENSE class "
                        "(LlamaFlashAttention2) on the same weights -- the authors' "
                        "DA+FA inference mode (their Table 4). No selection exists in "
                        "this mode; incompatible with oracle/indexer/dash_dense arms")
    p.add_argument("--dash_dense", action="store_true",
                   help="fasa Dash-8B only: override selection with the full causal "
                        "support at BOTH stages (== dense at the release's sigma; "
                        "RESULTS 78.6) -- the dense-equivalent arm for a model with "
                        "no dense class and no fixed top-k to inflate")
    p.add_argument("--random_sel", action="store_true",
                   help="CONTROL: replace the selector with a RANDOM block set of the same size. "
                        "Without this we cannot separate 'the oracle ranking does real work' from "
                        "'any change to the selection helps here'.")
    p.add_argument("--no_oracle", action="store_true",
                   help="VALIDATION MODE: run the ordinary selector. Must reproduce the known "
                        "sparse number for this cell, else the harness is wrong.")
    p.add_argument("--reuse_layers", type=int, default=0,
                   help="CROSS-LAYER SELECTION REUSE (the LongCat CLI / GLM IndexShare "
                        "premise, zero-training variant): layers with li %% K != 0 reuse "
                        "the owner layer's final block selection at the same decode step; "
                        "scoring is skipped there (indexer traffic /= K). Works in any arm "
                        "(shipped/oracle/union). 0 = off.")
    p.add_argument("--log_selections", type=int, default=0,
                   help="for the first N samples, dump per-decode-step per-(layer,head) "
                        "selected block ids (whatever arm is active) to <out>.sellog.json; "
                        "in oracle arms additionally log LongCat-style adjacent-layer mass "
                        "coverage. Analyze with pheno/xlayer_overlap.py. 0 = off.")
    p.add_argument("--log_every", type=int, default=4,
                   help="with --log_selections: log only decode steps where cache length "
                        "%% this == 0")
    p.add_argument("--allow_trunc", action="store_true")
    p.add_argument("--legacy_rows", action="store_true",
                   help="reproduce pre-2026-09-16 rows: no forced-block retention, mean-pool proxy ranking, Dash prior untouched")
    p.add_argument("--keep_only", action="store_true",
                   help="fidelity check: forced blocks + the kernel's own free ranking (must equal the shipped row)")
    p.add_argument("--seed", type=int, default=42, help="seeds torch/random (random controls)")
    p.add_argument("--split_prefill", action="store_true",
                   help="prefill all but the last prompt token, then let generate() feed the last prompt token through "
                        "the DECODE path (HELMET's protocol, model_utils.py). The decode-time selection hooks then also "
                        "choose the blocks behind the FIRST generated token, which otherwise comes from the sparse "
                        "prefill's last row and is out of reach of every decode-side arm")
    p.add_argument("--key_rot", choices=["none", "hadamard"], default="none",
                   help="with --key_bits / --quant_indexer: rotate the keys by a fixed randomized Hadamard matrix before the "
                        "per-channel quantizer (outlier channels are spread over all coordinates)")
    p.add_argument("--quant_stats_prefix", type=int, default=None,
                   help="with --key_bits: take the quantizer's per-channel mean and standard deviation from the first N cached "
                        "keys only (a causal quantizer; with InfLLM-V2 every row below 6144 tokens still reads every block)")
    p.add_argument("--quant_form", choices=["gauss", "minmax", "pooled"], default="gauss",
                   help="with --key_bits >= 2, the quantizer ablations: gauss = each channel standardized, Gaussian step (the "
                        "repair); minmax = KIVI's per-channel min-max grid; pooled = one mean and standard deviation per head")
    p.add_argument("--quant_step_scale", type=float, default=1.0,
                   help="with --key_bits >= 2 (gauss or pooled): multiply the Gaussian step, which moves the clamp with it")
    p.add_argument("--quant_log", action="store_true",
                   help="with --key_bits >= 2: record the clip rate, channel kurtosis and best uniform step of the keys as "
                        "the repair's quantizer standardizes them (oracle_run._log_quant)")
    p.add_argument("--swap_k", type=int, default=None,
                   help="minimal-edit repair at the PREFILL rows of a flagged head: keep the shipped row and replace at most this many of "
                        "its weakest free picks with the scorer's heaviest blocks that the shipped row missed (Proposition 1 says one "
                        "decisive block is what a pooled score loses, so the rest of the row's coverage is left alone). Decode rows keep "
                        "the full replacement. Needs --prefill_tail")
    p.add_argument("--row_gate", type=float, default=None,
                   help="rewrite a row of a listed head ONLY when the shipped row forfeits more than this much of the group's "
                        "attention mass under the armed scorer (captured mass of the new row minus captured mass of the shipped "
                        "row, divided by the number of query heads that vote); every other row keeps the shipped picks exactly. "
                        "Same unit as the head-flagging rule (0.01 = one percent of the attention mass)")
    p.add_argument("--row_gate_mode", choices=["total", "peak"], default="total",
                   help="what --row_gate thresholds: 'total' = the mass the shipped row forfeits in total; 'peak' = the largest mass of "
                        "a SINGLE block that the new row adds to the shipped row (the single decisive block of Proposition 1)")
    p.add_argument("--stage1_agg", choices=["sum", "max", "top2"], default=None,
                   help="exact-arm slot: rank the free slots by the SHIPPED compressed keys re-scored with this "
                        "aggregation over the GQA group's query heads ('sum' = the shipped rule, used as a fidelity "
                        "check of the re-implementation; 'max' tests whether summing dilutes the few retrieval heads)")
    p.add_argument("--prefill_tail", type=int, default=0,
                   help="exact arm only: at PREFILL, the last K query positions (the question) get their selection rows "
                        "from the armed criterion instead of the shipped selector, inside the same single prefill pass "
                        "(no second pass, no sequential tail). Decode rows are governed by the arm as always")
    p.add_argument("--tail_dense", action="store_true",
                   help="with --split_prefill --split_tail K>1: the K-1 tail tokens fed through the decode path read "
                        "every block (dense question tail, as SeerAttention ships and Tri-shape / TriangleMix do for "
                        "dense models); the last prompt token and all generated tokens keep the armed decode selection")
    p.add_argument("--split_tail", type=int, default=1,
                   help="with --split_prefill: how many trailing prompt tokens go through the decode path, one forward "
                        "each (1 = HELMET's protocol). A tail that covers the question lets the decode-time hooks also "
                        "choose the blocks that the QUESTION tokens read while the model processes them")
    p.add_argument("--row_offset", type=int, default=0,
                   help="evaluate rows [row_offset, row_offset + n) of --data, so that one cell can run as several "
                        "short jobs (pheno/merge_chunks.py reassembles them; greedy decoding, same draws)")
    p.add_argument("--layer_budget", default=None,
                   help="TRAINING-FREE per-layer budget, e.g. '0-15:41,16-23:126,24-31:40': each layer range keeps the "
                        "forced blocks plus that many free blocks of the shipped selector's OWN ranking. --topk sets the "
                        "row width and must cover the largest grant. The realized mean blocks per row is recorded so "
                        "that a cell can be held at the shipped traffic (96 blocks per row at 32K)")
    p.add_argument("--layer_budget_every", action="store_true",
                   help="apply --layer_budget at the prefill rows of those layers too (every token), not only at decode: "
                        "the zero-engineering control of more budget in the flagged layers")
    p.add_argument("--layers", default=None,
                   help="inclusive layer range 'a-b', or a comma list such as '9,16,22' (ranges allowed inside), in which "
                        "the armed override (exact mass, union, replacement) acts. "
                        "Every other layer keeps the shipped selection. Localizes where selection repair matters and "
                        "prices a repair that runs its indexer in only those layers")
    p.add_argument("--kv_heads", default=None,
                   help="comma list of LAYER.KVHEAD pairs, e.g. '9.0,16.0,22.0': the armed override acts on these KV heads "
                        "only; every other head of every layer keeps its shipped row (replaces --layers). For repairs "
                        "aimed at the few heads a calibration probe flags (pheno/needle_probe.py)")
    p.add_argument("--quant_indexer", type=int, default=None, choices=[1, 2, 3, 4, 8],
                   help="TRAINING-FREE indexer in place of --learned_indexer: block scores are the group-mean query's "
                        "softmax mass over a copy of the keys quantized to this many bits (see --key_bits). Works with "
                        "--learned_mix (union with the shipped selector) or alone (replacement)")
    p.add_argument("--key_bits", type=int, default=None, choices=[1, 2, 3, 4, 8],
                   help="exact arm (either --oracle_query): score blocks on a per-channel quantized copy of the keys with this "
                        "many bits (a training-free per-key scorer; 4 bits cost what the rank-32 indexer costs)")
    p.add_argument("--key_summary", default=None,
                   help="exact arm (either --oracle_query), instead of --key_bits: score blocks on a FULL-PRECISION summary "
                        "of the keys, pool:L (mean of L consecutive keys), channels:R (R label channels) or pca:R (top-R "
                        "principal directions). Comparison arms for where a selector's side memory is best spent")
    p.add_argument("--snapkv_window", type=int, default=0,
                   help="exact arm: SnapKV at block granularity, the free slots of every decode token filled from the blocks "
                        "the last W prompt tokens attend to most (scored once, at the end of the prefill); the prefill "
                        "itself is shipped. SnapKV's default window is 32")
    p.add_argument("--dense_heads", default=None,
                   help="KV heads 'L.h,...' that attend to every key (full attention, as DuoAttention gives its retrieval "
                        "heads) at every token; the other heads keep the release's selection. Use with --no_oracle")
    p.add_argument("--dense_heads_phase", choices=["all", "prefill"], default="all",
                   help="all (default): --dense_heads at every token. prefill: full attention in those heads only in the "
                        "prefill, and at decode the exact/4-bit arm selects their blocks; needs the oracle arm with "
                        "--kv_heads equal to --dense_heads and no --prefill_tail (the hybrid fix)")
    p.add_argument("--oracle_query", choices=["heads", "meanq", "quest_mean", "quest_max", "quest_sum"], default="heads",
                   help="DIAGNOSTIC for the exact mass arm. heads (default): the sum over the group's query heads of "
                        "each head's own softmax, the exact mass criterion. meanq: the softmax of the group-MEAN "
                        "query, which is the target the v6 indexer is distilled from (indexer_train.py)")
    p.add_argument("--tag", default="")
    return p


def build_model_and_state(a):
    """Load the sparse checkpoint and arm the selection hooks exactly as every corrected-protocol cell
    does. Returns (model, tok, cfg, state). Shared by main() and by other benchmark runners
    (pheno/helmet_run.py), so that no runner carries its own copy of the hook state. A stale copy once
    turned every union call into a silent replica of the shipped selector (2026-08-14)."""
    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
    from fasa_compat import ensure_sparse_config, is_fasa, apply_topk
    ensure_sparse_config(cfg)
    if getattr(cfg, "sparse_config", None) is None:
        raise SystemExit("ARM CHECK FAILED: no sparse_config -- ABORT")
    if a.topk is not None:
        cfg.sparse_config["topk"] = a.topk
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    # Dash-8B maps flash_attention_2 to a DENSE class; only sdpa is sparse
    # (the NOSA registration trap again, RESULTS 78.6). --dash_fa deliberately
    # loads that dense class: the authors' own DA+FA inference mode (Table 4),
    # which is the release's dense arm on identical weights.
    if a.dash_fa:
        if not hasattr(cfg, "sigma"):
            raise SystemExit("--dash_fa is only for the fasa Dash-8B variant -- ABORT")
        if not a.no_oracle or a.learned_indexer or a.dash_dense or a.random_sel:
            raise SystemExit("--dash_fa has no selection to override -- ABORT")
        impl = "flash_attention_2"
    else:
        # every fasa release maps sdpa to its sparse class; NSA-8B and Dash-8B
        # map flash_attention_2 to a DENSE class (config-invisible for NSA)
        impl = "sdpa" if is_fasa(cfg) else "flash_attention_2"
    if getattr(cfg, "mixer_types", None):        # MiniCPM-SALA: see pheno/compat_fla.py (the release's files stay untouched)
        import fla.ops.simple_gla as _sg
        from compat_fla import allow_explicit_head_first_false
        allow_explicit_head_first_false(_sg)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, config=cfg, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation=impl).cuda().eval()
    if is_fasa(cfg):
        acls = type(model.model.layers[0].self_attn).__name__
        want = "LlamaFlashAttention2" if a.dash_fa else "LlamaSdpaAttention"
        if acls != want:
            raise SystemExit(f"ARM CHECK FAILED: fasa arm instantiated {acls}, "
                             f"wanted {want} -- ABORT")
    if is_fasa(cfg) and a.topk is not None:
        apply_topk(model, a.topk)

    torch.manual_seed(a.seed); random.seed(a.seed)
    _KEY_ROT["on"] = (a.key_rot == "hadamard")
    if a.quant_stats_prefix is not None and (a.key_bits is None or a.quant_stats_prefix < 64):
        raise SystemExit("--quant_stats_prefix needs --key_bits and at least 64 keys -- ABORT")
    _QUANT_STATS["prefix"] = a.quant_stats_prefix
    arm_quantizer(a)
    if a.swap_k is not None and (a.swap_k < 1 or not a.prefill_tail or a.no_oracle or a.keep_only or a.random_sel):
        raise SystemExit("--swap_k needs --prefill_tail, a mass-criterion arm, and a value >= 1 -- ABORT")
    if a.row_gate is not None and (a.no_oracle or a.keep_only or a.random_sel or a.shortlist > 0 or a.learned_indexer or a.row_gate < 0):
        raise SystemExit("--row_gate needs a mass-criterion arm (exact or --key_bits / --key_summary), no shortlist, and a value >= 0 -- ABORT")
    state = {"oracle": not a.no_oracle, "random_sel": a.random_sel,
             "dash_dense": a.dash_dense,
             "forced_mode": ("legacy" if a.legacy_rows else "kernel"), "legacy_rows": a.legacy_rows,
             "keep_only": a.keep_only, "oracle_query": a.oracle_query, "layers": (_parse_layer_set(a.layers) if a.kv_heads is None
                        else frozenset(l for l, _ in _parse_kv_heads(a.kv_heads))),
             "kv_heads": _parse_kv_heads(a.kv_heads), "prefill_tail": a.prefill_tail, "stage1_agg": a.stage1_agg,
             "row_gate": a.row_gate, "row_gate_mode": a.row_gate_mode, "swap_k": a.swap_k,
             "snapkv_window": a.snapkv_window,
             "key_bits": a.key_bits, "key_summary": _parse_key_summary(a.key_summary),
             "layer_budget": _parse_layer_budget(a.layer_budget),
             "n_oracle_calls": 0, "errors": [], "criterion": a.criterion,
             "shortlist": a.shortlist, "indexer_r": a.indexer_r,
             "layer_band": (tuple(int(x) for x in a.layer_band.split(","))
                            if a.layer_band else None),
             "salience": a.salience,
             "indexer": (lambda pth: ({tuple(int(x) for x in kk.split("_")): v
                                       for kk, v in torch.load(pth, map_location="cpu").items()}
                                      if pth else None))(a.indexer_pcs),
             "reuse_k": a.reuse_layers,
             "sel_log": [] if a.log_selections else None,
             "log_n": a.log_selections, "log_every": a.log_every, "mass_log": []}
    if a.quant_indexer is not None:
        if a.learned_indexer or not a.no_oracle:
            raise SystemExit("--quant_indexer replaces --learned_indexer and needs --no_oracle -- ABORT")
        # same 6-tuple layout as a learned indexer; a string in slot 0 marks the training-free scorer, slot 1 = bits
        state["learned"] = [("meanq_quant", a.quant_indexer, 1.0, None, None, None)]
        if a.learned_mix:
            state["learned_mix"] = tuple(int(x) for x in a.learned_mix.split(","))
        print(f"training-free indexer: mean-query mass on {a.quant_indexer}-bit keys, mix={a.learned_mix}", flush=True)
    if a.learned_indexer:
        cks = [load_final(p) for p in a.learned_indexer.split(",")]
        state["learned"] = ([(c["Wq"].cuda().float(), c["Wk"].cuda().float(),
                              float(c.get("tau", 1.0)),
                              c["M1"].cuda().float() if "M1" in c else None,
                              c["m2"].cuda().float() if "m2" in c else None,
                              c["wdsa"].cuda().float() if "wdsa" in c else None) for c in cks])
        state["qagg"] = a.qagg
        ck = cks[0]
        if a.learned_mix:
            state["learned_mix"] = tuple(int(x) for x in a.learned_mix.split(","))
        if a.mix_bands:
            state["mix_bands"] = tuple(int(x) for x in a.mix_bands.split(","))
            state.setdefault("learned_mix", (32, 32))   # arm the mix branch
        state["sticky"] = bool(a.sticky)
        state["score_ema"] = a.score_ema
        state["conf_gate"] = a.conf_gate
        state["conf_soft"] = a.conf_soft
        if a.learned_arbiter:
            import arbiter_train as AT
            ak = torch.load(a.learned_arbiter, map_location="cpu")
            if ak.get("query_dep"):
                amlp = AT.Arbiter().cuda()
            else:
                amlp = torch.nn.Sequential(torch.nn.Linear(AT.FEAT, 16), torch.nn.ReLU(),
                                           torch.nn.Linear(16, 1)).cuda()
            amlp.load_state_dict(ak["mlp"]); amlp.eval()
            state["arbiter"] = (amlp, ak["mu"].cuda(), ak["sd"].cuda(),
                                bool(ak.get("query_dep")))
        print(f"learned indexer: {a.learned_indexer} rank={ck['rank']} tau={ck.get('tau', 1.0)} "
              f"mix={a.learned_mix}", flush=True)
    if a.dash_dense_decode:
        if not hasattr(cfg, "sigma"):
            raise SystemExit("--dash_dense_decode is fasa Dash-8B only -- ABORT")
        _DASH_DENSE_DECODE["on"] = True
    if not a.dash_fa and not a.no_install:   # DA+FA dense class has no hook points
        install_oracle(model, cfg, state)
    STASH.pop("decode_topk", None); STASH.pop("budget_stats", None); STASH.pop("n_decode_sf", None)
    STASH.pop("budget_every", None); STASH.pop("budget_prefill_calls", None)
    if state.get("layer_budget") is not None:
        if a.no_install or not hasattr(model.model.layers[0].self_attn, "topk"):
            raise SystemExit("--layer_budget needs the hooked base InfLLM-V2 attention (attn.topk) -- ABORT")
        shipped_w = model.model.layers[0].self_attn.topk
        shipped_free = cfg.sparse_config["topk"] - 2          # 64 slots minus the init block and the query's own block
        STASH["decode_topk"] = {li: shipped_w + (n - shipped_free)
                                for li in range(cfg.num_hidden_layers)
                                for n in [_layer_free_budget(state["layer_budget"], li)] if n is not None}
        STASH["budget_every"] = bool(getattr(a, "layer_budget_every", False))
        print(f"layer budget armed: shipped row {shipped_w} ({shipped_free} free); "
              f"{'every-token' if STASH['budget_every'] else 'decode'} rows "
              f"{sorted(set(STASH['decode_topk'].values()))} in {len(STASH['decode_topk'])} layers", flush=True)
    return model, tok, cfg, state


def main():
    a = build_parser().parse_args()
    if a.layer_budget is not None and (a.learned_indexer or a.keep_only or a.legacy_rows or a.topk is not None
                                       or a.random_sel or a.criterion != "mass" or a.shortlist > 0):
        # gated combinations only: the shipped kernel (PARITY_BUDGET) and the mass arms that fill whatever row width
        # the kernel returns (exact arm and --quant_indexer, PARITY_BUDGET_EXACT)
        raise SystemExit("--layer_budget runs with the shipped kernel, the exact arm or --quant_indexer only -- ABORT")
    if a.key_summary is not None:
        _parse_key_summary(a.key_summary)
        if (a.no_oracle or a.key_bits is not None or a.key_rot != "none" or a.criterion != "mass" or a.shortlist > 0
                or a.legacy_rows or a.learned_indexer or a.quant_indexer is not None or a.stage1_agg
                or a.oracle_query.startswith("quest_") or a.random_sel or a.keep_only or a.dash_dense):
            raise SystemExit("--key_summary is a plain exact-arm scorer: no other scorer flags -- ABORT")
    if a.key_rot != "none" and a.key_bits is None and a.quant_indexer is None:
        raise SystemExit("--key_rot needs --key_bits or --quant_indexer -- ABORT")
    if a.stage1_agg and (a.no_oracle or a.key_bits is not None or a.oracle_query != "heads" or a.prefill_tail
                         or a.criterion != "mass" or a.shortlist > 0 or a.legacy_rows):
        raise SystemExit("--stage1_agg is a plain exact-arm slot: no other scorer flags -- ABORT")
    if a.prefill_tail and (a.no_oracle or a.split_prefill or a.criterion != "mass" or a.shortlist > 0 or a.legacy_rows
                           or (a.oracle_query.startswith("quest_") and (a.key_bits is not None or a.key_summary))):
        raise SystemExit("--prefill_tail needs the exact arm (mass criterion or Quest's bound on the true keys) and "
                         "excludes --split_prefill -- ABORT")
    if a.kv_heads is not None and a.layers is not None:
        raise SystemExit("--kv_heads replaces --layers: give one of them -- ABORT")
    if a.tail_dense and (not a.split_prefill or a.split_tail < 2 or a.no_install):
        raise SystemExit("--tail_dense needs --split_prefill with --split_tail >= 2 and installed hooks -- ABORT")
    if a.split_tail < 1 or (a.split_tail != 1 and not a.split_prefill):
        raise SystemExit("--split_tail needs --split_prefill and a value >= 1 -- ABORT")
    if a.snapkv_window and (a.no_oracle or a.key_bits is not None or a.key_summary or a.prefill_tail or a.kv_heads
                            or a.oracle_query != "heads" or a.random_sel or a.keep_only):
        raise SystemExit("--snapkv_window is its own exact-arm baseline in every head, alone -- ABORT")
    if a.oracle_query.startswith("quest_") and (a.no_oracle or a.key_bits is not None or a.criterion != "mass" or a.shortlist > 0
                                                 or a.log_selections):
        raise SystemExit("--oracle_query quest_* is a plain exact-arm baseline: no --key_bits, --shortlist, --criterion "
                         "output or --log_selections -- ABORT")
    if a.key_bits is not None and (a.no_oracle or a.criterion != "mass" or a.shortlist > 0):
        raise SystemExit("--key_bits needs the exact arm (mass criterion, no shortlist), with either --oracle_query -- ABORT")
    metric_fn, base_task, canon = load_task(a.ruler_root, a.task)
    max_new = canon if a.max_new == "canonical" else int(a.max_new)
    model, tok, cfg, state = build_model_and_state(a)
    dense_stats = {}
    if a.dense_heads:
        if a.dense_heads_phase == "all" and (not a.no_oracle or a.no_install):
            raise SystemExit("--dense_heads is its own arm: use it with --no_oracle and installed hooks -- ABORT")
        if a.dense_heads_phase == "prefill" and (a.no_oracle or a.no_install or a.prefill_tail or a.keep_only or a.random_sel
                                                 or not a.kv_heads or _parse_kv_heads(a.kv_heads) != _parse_kv_heads(a.dense_heads)):
            raise SystemExit("--dense_heads_phase prefill is the hybrid: the oracle arm at decode in exactly the dense heads "
                             "(--kv_heads equal to --dense_heads), no --prefill_tail -- ABORT")
        install_dense_heads(model, _parse_kv_heads(a.dense_heads), dense_stats, phase=a.dense_heads_phase)
        print(f"dense heads armed: {a.dense_heads} ({a.dense_heads_phase})", flush=True)
    elif a.dense_heads_phase != "all":
        raise SystemExit("--dense_heads_phase needs --dense_heads -- ABORT")

    rows = [json.loads(l) for l in open(a.data)][a.row_offset: a.row_offset + a.n]
    if len(rows) < a.n:
        raise SystemExit(f"data has {len(rows)} rows < n={a.n} -- ABORT")
    preds, refs, gens, ptoks, term = [], [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        state["sample_i"] = i
        enc = tok(r["input"] + r.get("answer_prefix", ""), return_tensors="pt")
        ids = enc.input_ids.cuda(); ptoks.append(int(ids.shape[1]))
        with torch.inference_mode():
            mask = enc.attention_mask.cuda()
            gkw = {}
            if a.split_prefill:
                from transformers.cache_utils import DynamicCache
                K = a.split_tail
                pre = model.model(input_ids=ids[:, :-K], attention_mask=mask[:, :-K], past_key_values=DynamicCache(),
                                  use_cache=True)
                if pre.past_key_values is None:
                    raise SystemExit("--split_prefill: the model returned no cache -- ABORT")
                cache = pre.past_key_values
                prev_dw = STASH.get("decode_topk")
                if a.tail_dense:                      # the tail tokens read EVERY block (a dense question tail)
                    STASH["decode_topk"] = {li: 1_000_000 for li in range(cfg.num_hidden_layers)}
                try:
                    for pos in range(ids.shape[1] - K, ids.shape[1] - 1):  # tail tokens except the last: decode path
                        cache = model.model(input_ids=ids[:, pos:pos + 1], attention_mask=mask[:, :pos + 1],
                                            past_key_values=cache, use_cache=True).past_key_values
                finally:
                    if a.tail_dense:
                        if prev_dw is None:
                            STASH.pop("decode_topk", None)
                        else:
                            STASH["decode_topk"] = prev_dw
                gkw["past_key_values"] = cache
            out = model.generate(input_ids=ids, attention_mask=mask,
                                 max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id, **gkw)
        g = out[0, ids.shape[1]:]
        gens.append(int(g.shape[0])); term.append(bool(g.shape[0] < max_new))
        preds.append(tok.decode(g, skip_special_tokens=True)); refs.append(r["outputs"])
        if i % 20 == 0:
            print(f"  [{i}/{len(rows)}] oracle_calls={state['n_oracle_calls']}", flush=True)
    elapsed = time.time() - t0

    if a.dash_dense_decode and _DASH_DENSE_DECODE.get("calls", 0) == 0:
        raise SystemExit("ABORT: dense-decode rung requested but never fired")
    if a.dash_dense_decode:
        res_extra_dd = _DASH_DENSE_DECODE.get("calls", 0)
        print(f"dense-decode calls: {res_extra_dd}", flush=True)
    if a.keep_only and state["n_oracle_calls"] == 0:
        raise SystemExit("ABORT: --keep_only requested but the hook never fired")
    if a.snapkv_window and not (state.get("snapkv_prefill_calls") and state.get("snapkv_decode_rows")):
        raise SystemExit(f"ABORT: --snapkv_window armed but prefill captures {state.get('snapkv_prefill_calls')}, decode rows "
                         f"{state.get('snapkv_decode_rows')}")
    if a.dense_heads and a.dense_heads_phase == "all" and not (dense_stats.get("prefill_calls") and dense_stats.get("decode_calls")):
        raise SystemExit(f"ABORT: --dense_heads requested but the splice fired {dense_stats} -- ABORT")
    if a.dense_heads and a.dense_heads_phase == "prefill" and not (dense_stats.get("prefill_calls") and dense_stats.get("decode_calls_selected")
                                                                   and not dense_stats.get("decode_calls")):
        raise SystemExit(f"ABORT: --dense_heads_phase prefill but the splice fired {dense_stats} -- ABORT")
    if a.prefill_tail and str(a.oracle_query).startswith("quest_") and not state.get("n_quest_tail_rows"):
        raise SystemExit("ABORT: Quest with --prefill_tail but no prompt row was scored by the bound")
    if (state["oracle"] or state["dash_dense"]) and state["n_oracle_calls"] == 0:
        raise SystemExit(f"ABORT: oracle requested but NEVER invoked -- the patch did not take. "
                         f"errors={state['errors'][:3]}")
    # every generated token costs one decode-path forward under --split_prefill (the first token included) and one
    # fewer per sample without it; hooked arms count one call per active layer per decode forward
    sparse_ids = _sparse_layer_ids(model)
    n_active = _n_active_layers(state.get("layers"), cfg.num_hidden_layers, sparse_ids)
    fwd = (sum(gens) + len(gens) * (a.split_tail - 1)) if a.split_prefill else sum(max(gl - 1, 0) for gl in gens)
    calls_expected = n_active * fwd
    hooked = ((state["oracle"] or state.get("learned") is not None or a.keep_only or a.random_sel)
              and not a.no_install and not hasattr(sys.modules[type(model).__module__], "triton_entmax")
              and not (state.get("reuse_k") or 0) > 1)
    sf_expected = len(sparse_ids) * fwd
    if a.split_prefill and not a.no_install and not a.dash_fa and STASH.get("n_decode_sf", 0) != sf_expected:
        raise SystemExit(f"ABORT: --split_prefill expected {sf_expected} decode-path sparse forwards, saw "
                         f"{STASH.get('n_decode_sf', 0)}: the first token did not go through the decode path")
    if hooked and state["n_oracle_calls"] != calls_expected:
        # enforced in every mode since 2026-09-19 (an independent audit noted it was only enforced under --split_prefill;
        # all 615 chunk files on disk at that date satisfy it)
        raise SystemExit(f"ABORT: expected {calls_expected} hook calls ({n_active} layers x {fwd} decode forwards), saw "
                         f"{state['n_oracle_calls']}: the hook did not see every decode row"
                         + (" (--split_prefill: the first token did not go through the decode path)" if a.split_prefill else ""))
    pts = state.get("prefill_tail_stats")
    if a.prefill_tail:
        n_heads_hit = (n_active * cfg.num_key_value_heads if state.get("kv_heads") is None else len(state["kv_heads"]))
        want = sum(min(a.prefill_tail, pt) for pt in ptoks) * n_heads_hit      # a tail longer than the prompt covers every row
        if pts is None or pts[0] != want or pts[1] != 0:
            raise SystemExit(f"ABORT: --prefill_tail rewrote {pts and pts[0]} rows (expected {want}) with "
                             f"{pts and pts[1]} forced blocks missing from shipped rows (expected 0)")
    per = [metric_fn([x], [y]) for x, y in zip(preds, refs)]
    idx_name = a.learned_indexer or (f"meanq_quant:{a.quant_indexer}bit" if a.quant_indexer is not None else None)
    arm = ("random-union" if (a.random_sel and a.learned_mix) else "random" if a.random_sel
           else "exact-mass" if state["oracle"] else "keep-only" if a.keep_only
           else ("layer-budget-every" if a.layer_budget_every else "layer-budget") if a.layer_budget
           else "union" if (idx_name and a.learned_mix) else "replacement" if idx_name
           else "dense-forced" if (a.topk and a.topk >= 100000) or a.dash_fa or a.dash_dense else "shipped")
    fs = state.get("forced_stats", [0, 0, 0])
    try:
        git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        git_rev = None
    res = {"unhooked_layers": STASH.get("unhooked_layers"), "tag": a.tag, "arm": arm, "arm_legacy_label": ("random-control" if state["random_sel"] else "oracle") if state["oracle"] else "sparse-control",
           "forced_mode": state.get("forced_mode"), "forced_mean": (fs[1] / fs[0] if fs[0] else None),
           "free_mean": (fs[2] / fs[0] if fs[0] else None), "keeponly_mismatch": state.get("keeponly_mismatch"),
           "dash_prior_fixed": state.get("dash_prior_fixed"), "seed": a.seed,
           "row_offset": a.row_offset, "oracle_query": a.oracle_query, "layers": a.layers, "key_bits": a.key_bits,
           "prefill_tail": a.prefill_tail, "prefill_tail_rows": (pts[0] if pts else None),
           "stage1_agg": a.stage1_agg, "stage1_mismatch": state.get("stage1_mismatch"), "key_rot": a.key_rot, "key_summary": a.key_summary,
           "dense_heads": a.dense_heads, "dense_head_stats": (dense_stats or None),
           "dense_heads_phase": (a.dense_heads_phase if a.dense_heads else None), "n_quest_tail_rows": state.get("n_quest_tail_rows"),
           "snapkv_window": a.snapkv_window, "snapkv_prefill_calls": state.get("snapkv_prefill_calls"),
           "snapkv_decode_rows": state.get("snapkv_decode_rows"),
           "quant_stats_prefix": a.quant_stats_prefix, "quant_form": a.quant_form, "quant_step_scale": a.quant_step_scale,
           "quant_log": quant_log_summary(), "row_gate": a.row_gate, "row_gate_mode": (a.row_gate_mode if a.row_gate is not None else None), "row_gate_stats": state.get("row_gate_stats"),
           "swap_k": a.swap_k, "swap_stats": state.get("swap_stats"),
           "prefill_rows_impl": (("batched" if (a.prefill_tail > 512 or a.row_gate is not None) else "loop") if a.prefill_tail else None),
           "kv_heads": a.kv_heads, "kv_heads_kept_rows": (state.get("kv_heads_kept") or [None])[0],
           "split_prefill": bool(a.split_prefill), "split_tail": (a.split_tail if a.split_prefill else None),
           "tail_dense": bool(a.tail_dense),
           "hook_calls_expected": (calls_expected if hooked else None),
           "decode_sparse_forwards": STASH.get("n_decode_sf"), "decode_sparse_forwards_expected": sf_expected,
           "layer_budget": a.layer_budget, "layer_budget_every": bool(a.layer_budget_every),
           "budget_prefill_calls": STASH.get("budget_prefill_calls"),
           "blocks_per_row_mean": ((lambda b: b[1] / b[0] if b[0] else None)(STASH.get("budget_stats", [0, 0]))),
           "task": a.task, "target_len": a.target_len, "n": len(rows), "max_new": max_new,
           "string_match_all": metric_fn(preds, refs), "criterion": a.criterion,
           "shortlist": a.shortlist, "indexer_pcs": a.indexer_pcs, "indexer_r": a.indexer_r,
           "layer_band": a.layer_band, "salience": a.salience,
           "learned_indexer": idx_name, "quant_indexer": a.quant_indexer,
           "all_correct": sum(1 for s in per if s >= 100.0) / len(per),
           "scores": per, "predictions": preds, "references": refs,
           "mean_gen_len": statistics.fmean(gens), "trunc_frac": 1.0 - sum(term) / len(term),
           "mean_prompt_tokens": statistics.fmean(ptoks),
           "n_oracle_calls": state["n_oracle_calls"], "n_errors": len(state["errors"]),
           "reuse_layers": a.reuse_layers, "n_reused": state.get("n_reused", 0),
           "errors": state["errors"][:5], "topk_override": a.topk,
           "sparse_config": dict(cfg.sparse_config),
           "provenance": {"data": os.path.abspath(a.data), "elapsed_s": elapsed,
                          "data_sha256_12": hashlib.sha256(open(a.data, "rb").read()).hexdigest()[:12],
                          "model_dir": os.path.abspath(a.model_dir), "argv": sys.argv, "git": git_rev,
                          "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                          "torch": torch.__version__}}
    if state["errors"]:
        res["invalid"] = True
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    if state["errors"]:
        raise SystemExit(f"HOOK ERRORS {len(state['errors'])}: {state['errors'][:3]} -- cell INVALID (JSON written with invalid=true)")
    if state.get("sel_log") is not None:
        Path(a.out + ".sellog.json").write_text(json.dumps(
            {"arm": res["arm"], "learned_indexer": a.learned_indexer,
             "learned_mix": a.learned_mix, "log_every": a.log_every,
             "sel": state["sel_log"], "mass": state["mass_log"]}))
        print(f"selection log: {len(state['sel_log'])} pick rows, "
              f"{len(state['mass_log'])} mass rows -> {a.out}.sellog.json", flush=True)
    print(json.dumps({k: res[k] for k in ("arm", "task", "string_match_all", "all_correct",
                                          "n_oracle_calls", "n_errors", "trunc_frac")}, indent=2))


if __name__ == "__main__":
    main()
