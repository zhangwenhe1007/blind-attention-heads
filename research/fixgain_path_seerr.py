#!/usr/bin/env python
"""Complete and integrated fix gain on SeerAttention-R.

Gate form (DuoAttention; PruLong): at every position where the selector acts (SeerAttention-R: the decode tokens, its
prefill being dense in every arm) query head q's attention output is
    o_q(alpha) = o_q^ship + alpha_u (o_q^full - o_q^ship),
alpha_u the dial of q's KV head u (the unit the gate selects for). F_x(alpha) = sum over the teacher-forced answer tokens
of log p(next answer token): F_x(0) is the shipped model, F_x(1) full attention in every unit.
  complete fix gain (option 1):  G_u(x) = dF_x/dalpha_u at alpha = 0, exactly: one backward pass over the whole answer,
      the answer tokens' keys and values kept in the graph (the prompt's are the same in every alpha state, the prefill
      being dense), signed sum over u's query heads and the answer tokens. Score: mean over prompts of |G_u(x)|
      (|.| per prompt, Michel et al. 2019; Nanda 2023 / Kramar et al. 2024 attribution patching).
  integrated fix gain (option 2): IFG_u(x) = int_0^1 dF_x/dalpha_u at alpha = beta * 1 dbeta (integrated gradients,
      Sundararajan et al. 2017, the Aumann-Shapley share of the gap), by Gauss-Legendre quadrature on --nodes points.
      Completeness: sum_u IFG_u(x) = F_x(1) - F_x(0) up to quadrature error and gate switches; the residual is stored.
The gate selects in each alpha state from the blended model (its selection is a constant of the derivative). Decode
attention is recomputed in PyTorch (fp32) over the gate's blocks plus block 0, and checked against the kernel on the
first prompt (abort beyond --tol). The protocol is pheno/fixgain_probe_seerr.py's: calibration prompts (never seed 42),
dense prefill, then the needle's tokens teacher-forced one at a time (feed token j, score token j + 1).
    python pheno/fixgain_path_seerr.py --repo ... --gates ... --base ... --data <seed-7 draw> --out X.json
--detach_kv holds the answer tokens' keys and values constant (the frozen probe's convention); with --nodes 0 its signed
per-prompt sums must reproduce fixgain_probe_seerr.py's gain_full_kv (the plumbing check).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def gauss_legendre01(m):
    """Nodes and weights of m-point Gauss-Legendre quadrature on [0, 1] (weights sum to 1)."""
    if m == 0:
        return [], []
    x, w = np.polynomial.legendre.leggauss(m)
    return [float((xi + 1) / 2) for xi in x], [float(wi / 2) for wi in w]


def midpoint01(m):
    """Nodes and weights of the m-point midpoint rule on [0, 1]."""
    return [(i + 0.5) / m for i in range(m)], [1.0 / m] * m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True); p.add_argument("--gates", required=True); p.add_argument("--base", required=True)
    p.add_argument("--data", required=True); p.add_argument("--n", type=int, default=50); p.add_argument("--row_offset", type=int, default=0)
    p.add_argument("--steps", type=int, default=10); p.add_argument("--token_budget", type=int, default=4096)
    p.add_argument("--nodes", type=int, default=4, help="quadrature nodes of the integrated fix gain (0: first order only)")
    p.add_argument("--rule", choices=["gl", "mid"], default="gl",
                   help="gl: Gauss-Legendre; mid: the midpoint rule on a uniform grid, robust to the sharp transitions and "
                        "selection switches of F along the path (2026-10-04 diagnostics)")
    p.add_argument("--tol", type=float, default=0.05, help="worst relative gap between the torch shipped output and the kernel")
    p.add_argument("--detach_kv", action="store_true")
    p.add_argument("--fpath", type=int, default=0, help="diagnostic: F at fpath + 1 evenly spaced beta (no gradient)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if "/seed=42/" in a.data:
        raise SystemExit("the fix-gain probe runs on calibration prompts, not the evaluation draw -- ABORT")
    sys.path.insert(0, a.repo)
    from seerr_run import compressed_key_cache, local_gate_dir
    from needle_probe import needle_token_span
    from transformers import AutoConfig, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    if AutoConfig.from_pretrained(a.base).model_type == "qwen2":
        import seer_attn.decode_sparse.qwen2.modeling_qwen2_seerattn_inference as M
        from seer_attn import SeerDecodingQwen2ForCausalLM as SeerDecoding
    else:
        import seer_attn.decode_sparse.qwen3.modeling_qwen3_seerattn_inference as M
        from seer_attn import SeerDecodingQwen3ForCausalLM as SeerDecoding
    tok = AutoTokenizer.from_pretrained(a.base)
    model = SeerDecoding.from_pretrained(local_gate_dir(a.gates, a.base), torch_dtype=torch.bfloat16,
                                         seerattn_sparsity_method="token_budget", seerattn_token_budget=a.token_budget,
                                         seerattn_implementation="seer_sparse").cuda().eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    cfg = model.config
    L, Hq, Hkv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    G = Hq // Hkv
    nodes, weights = gauss_legendre01(a.nodes) if a.rule == "gl" else midpoint01(a.nodes)
    ST = {"li": None, "on": False, "beta": 0.0, "grad": False, "P": None, "check": True, "worst": 0.0, "checked": 0}
    PK, PV, CAPS = [None] * L, [None] * L, []

    def attend(q, kpT, vp, kaT, va, rows, block, scale):
        """q [Hq, D] fp32; kpT [Hkv, D, P], vp [Hkv, P, D] the prompt's keys and values (fp32 constants, kept in these
        layouts so that the matmuls save references, not copies); kaT [Hkv, D, A], va [Hkv, A, D] the answer tokens'
        (graph); rows [Hkv, nb] bool, the blocks read. Returns [Hq, D] fp32."""
        P, A = kpT.shape[-1], kaT.shape[-1]
        tok_mask = rows.repeat_interleave(block, dim=1)[:, :P + A]                       # [Hkv, T]
        qh = q.view(Hkv, G, -1)
        lg = torch.cat([torch.matmul(qh, kpT), torch.matmul(qh, kaT)], dim=-1) * scale  # [Hkv, G, T]
        pr = torch.softmax(lg.masked_fill(~tok_mask[:, None, :], -float("inf")), dim=-1)
        return (torch.matmul(pr[..., :P], vp) + torch.matmul(pr[..., P:], va)).reshape(Hq, -1)

    def wrapped(query_states, key_states, value_states, attention_mask, query_length, softmax_scale=None,
                cache_seqlens=None, block_mask=None, block_size=None, **kw):
        if not (ST["on"] and query_length == 1 and block_mask is not None):
            return orig(query_states, key_states, value_states, attention_mask, query_length, softmax_scale=softmax_scale,
                        cache_seqlens=cache_seqlens, block_mask=block_mask, block_size=block_size, **kw)
        li, P = ST["li"], ST["P"]
        T = int(cache_seqlens[0])
        nbv = math.ceil(T / block_size)
        q = query_states.reshape(-1, query_states.shape[-2], query_states.shape[-1])[0].float()      # [Hq, D]
        if PK[li] is None:            # first decode call of this prompt: the prompt's keys and values, fp32, once
            PK[li] = key_states[0, :P].detach().float().permute(1, 2, 0).contiguous()
            PV[li] = value_states[0, :P].detach().float().permute(1, 0, 2).contiguous()
        ka, va = key_states[0, P:T].float(), value_states[0, P:T].float()                           # [A, Hkv, D]
        if a.detach_kv:
            ka, va = ka.detach(), va.detach()
        kaT, vaH = ka.permute(1, 2, 0), va.permute(1, 0, 2)
        ship = block_mask[0, :, :nbv].clone()
        ship[:, 0] = True                                                           # first_block_unmasked
        full = torch.ones_like(ship)
        beta = ST["beta"]
        o_ship = attend(q, PK[li], PV[li], kaT, vaH, ship, block_size, softmax_scale)
        if beta == 0.0:
            with torch.no_grad():
                o_full = attend(q, PK[li], PV[li], kaT, vaH, full, block_size, softmax_scale)
            o = o_ship
        else:
            o_full = attend(q, PK[li], PV[li], kaT, vaH, full, block_size, softmax_scale)
            o = o_ship + beta * (o_full - o_ship)
        if ST["check"] and beta == 0.0:
            with torch.no_grad():
                ok = orig(query_states, key_states, value_states, attention_mask, query_length,
                          softmax_scale=softmax_scale, cache_seqlens=cache_seqlens, block_mask=block_mask,
                          block_size=block_size, **kw).reshape(Hq, -1).float()
            rel = ((o_ship.detach() - ok).abs().max(-1).values / ok.abs().max(-1).values.clamp(min=1e-3)).max().item()
            ST["worst"] = max(ST["worst"], rel); ST["checked"] += 1
            if rel > a.tol:
                raise RuntimeError(f"layer {li}: torch shipped output differs from the kernel's by {rel:.3f} relative -- ABORT")
        out = o.to(query_states.dtype).view(1, 1, Hq, -1)
        if ST["grad"]:
            out.retain_grad()
            CAPS.append((li, out, (o_full - o_ship).detach()))
        return out

    orig = M.sparse_flash_attention_forward
    M.sparse_flash_attention_forward = wrapped
    for li, layer in enumerate(model.model.layers):
        of = layer.self_attn.forward

        def mk(of, li):
            def f(*args, **kw):
                ST["li"] = li
                return of(*args, **kw)
            return f
        layer.self_attn.forward = mk(of, li)
    model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.detach().requires_grad_(True) if ST["grad"] else o)

    passes = ([("g0", 0.0, True)] + [(f"n{k}", b, True) for k, b in enumerate(nodes)] + [("f1", 1.0, False)]
              + [(f"p{i}", i / a.fpath, False) for i in range(a.fpath + 1) if a.fpath])
    rows = [json.loads(l) for l in open(a.data)][a.row_offset: a.row_offset + a.n]
    per_prompt, layers_seen = [], set()
    t0 = time.time()
    for ri, r in enumerate(rows):
        text = r["input"] + r.get("answer_prefix", "")
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        ids = enc.input_ids.cuda()
        s, e = needle_token_span(text, r["outputs"][0], enc.offset_mapping[0].tolist())
        cont = ids[0, s:e][: a.steps].tolist()
        kv, kc = DynamicCache(), compressed_key_cache(model.num_layers, model.block_size)
        am = torch.ones_like(ids)
        with torch.no_grad():
            o = model(ids, attention_mask=am, past_key_values=kv, k_compressed_cache=kc, use_cache=True, logits_to_keep=1)
        kv, kc = o.past_key_values, o.k_compressed_cache
        pre_kv = [(lay.keys, lay.values) for lay in kv.layers]
        clone = lambda d: {k_: (t.clone() if isinstance(t, torch.Tensor) else t) for k_, t in d.items()}   # noqa: E731
        pre_kc = (clone(kc.k_compressed), clone(kc.k_remainder), getattr(kc, "remainder_len", None))
        ST["P"] = ids.shape[1]
        for li in range(L):
            PK[li] = PV[li] = None
        rec = {"n": len(cont) - 1}
        for name, beta, grad in passes:
            for lay, (k0, v0) in zip(kv.layers, pre_kv):
                lay.keys, lay.values = k0, v0
            kc.k_compressed, kc.k_remainder = clone(pre_kc[0]), clone(pre_kc[1])
            if pre_kc[2] is not None:
                kc.remainder_len = pre_kc[2]                 # the gate's write position in its partial block
            CAPS.clear()
            ST.update(on=True, beta=beta, grad=grad)
            amj, F = am, None
            with torch.set_grad_enabled(grad):
                for j in range(len(cont) - 1):
                    amj = torch.cat([amj, amj.new_ones(1, 1)], dim=1)
                    o = model(torch.tensor([[cont[j]]], device=ids.device), attention_mask=amj, past_key_values=kv,
                              k_compressed_cache=kc, use_cache=True, logits_to_keep=1)
                    lp = torch.log_softmax(o.logits[0, -1].float(), dim=-1)[cont[j + 1]]
                    F = lp if F is None else F + lp
                    kv, kc = o.past_key_values, o.k_compressed_cache
                    for dct in (kc.k_compressed, kc.k_remainder):        # the gate's inputs: constants of the derivative
                        for kk in list(dct):
                            if isinstance(dct[kk], torch.Tensor):
                                dct[kk] = dct[kk].detach()
                    if a.detach_kv or not grad:
                        for lay in kv.layers:
                            lay.keys, lay.values = lay.keys.detach(), lay.values.detach()
            ST.update(on=False, grad=False, check=False)
            rec[f"F_{name}"] = float(F)
            if grad:
                F.backward()
                g = torch.zeros(L, Hq, dtype=torch.float64)
                for li, out, delta in CAPS:
                    g[li] += (out.grad[0, 0].float() * delta).sum(-1).double().cpu()
                    layers_seen.add(li)
                rec[f"G_{name}"] = g.view(L, Hkv, G).sum(-1).tolist()          # signed, summed over the KV group
                CAPS.clear()
            del F, o
        if nodes:
            ifg = sum(w * torch.tensor(rec[f"G_n{k}"], dtype=torch.float64) for k, w in enumerate(weights))
            rec["IFG"] = ifg.tolist()
            rec["dF_dbeta"] = [float(torch.tensor(rec.pop(f"G_n{k}"), dtype=torch.float64).sum()) for k in range(len(nodes))]
            rec["completeness"] = {"sum_ifg": float(ifg.sum()), "gap": rec["F_f1"] - rec["F_g0"]}
        per_prompt.append(rec)
        if ri % 5 == 0:
            c = rec.get("completeness")
            print(f"  [{ri}/{len(rows)}] F0 {rec['F_g0']:.3f} F1 {rec['F_f1']:.3f}"
                  + (f" sum IFG {c['sum_ifg']:.3f}" if c else "") + f" worst kernel gap {ST['worst']:.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    if not layers_seen:
        raise SystemExit("the decode hook never fired -- ABORT")
    mean_abs = lambda key: (torch.tensor([x[key] for x in per_prompt], dtype=torch.float64).abs().mean(0)).tolist()  # noqa: E731
    res = {"definition": "complete fix gain G_u(x) = dF_x/dalpha_u at alpha = 0 and integrated fix gain IFG_u(x) (Gauss-"
                         "Legendre on [0, 1] along alpha = beta * 1), F_x = sum of log p(next needle token) over the "
                         "teacher-forced steps, the gate form o(alpha) = o_ship + alpha (o_full - o_ship) at every decode "
                         "token, KV unit = signed sum over its query heads; score = mean over prompts of |.|",
           "n_prompts": len(rows), "row_offset": a.row_offset, "rule": a.rule, "nodes": nodes, "weights": weights,
           "detach_kv": a.detach_kv, "layers_hooked": sorted(layers_seen),
           "kernel_check": {"calls": ST["checked"], "worst_rel": ST["worst"]},
           "score_first_order_kv": mean_abs("G_g0"), "score_integrated_kv": (mean_abs("IFG") if nodes else None),
           "per_prompt": per_prompt,
           "provenance": {"argv": sys.argv, "data": os.path.abspath(a.data), "gpu": torch.cuda.get_device_name(0),
                          "elapsed_s": time.time() - t0}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"FIXGAIN_PATH_DONE prompts {len(rows)} kernel worst {ST['worst']:.4f}", flush=True)


if __name__ == "__main__":
    main()
