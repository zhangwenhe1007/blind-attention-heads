#!/usr/bin/env python
"""Per-layer NEEDLE RECALL of the decode-time selector on a calibration draw of the synthetic retrieval task.

Generic text cannot locate the layers a retrieval repair must target (pheno/layer_calib.py on PG-19: the shipped rows
capture about as much mass as any scorer in every layer). This probe asks the question the downstream task asks: at
the decode steps that copy the answer, is the block that holds the token being copied inside the row?

Protocol. A calibration draw of RULER niah_multikey_3 generated with a random seed that no evaluated cell uses.
The prompt is prefilled by the model's own sparse path (all but its last token), then the last prompt token and the
answer's own context tokens (": <uuid>", taken from the needle sentence in the prompt) are fed one at a time through
the DECODE path with the shipped selector untouched (teacher forcing, on-policy cache). At step i the token to be
produced is the i-th needle token, so the block that must be read is the one holding context position start + i.
For every layer and KV head the hook records whether that block is
  forced     (inside the kernel's init / local-window / own-block set: then no selector is tested),
  in the SHIPPED row, in the EXACT-mass top-n_free, in the LOW-BIT mean-query scorer's top-n_free,
and the exact attention mass on it (head-summed softmax / group size).
Output: per-layer and per-(layer, head) recalls, the needle mass, teacher-forced next-token accuracy, and the
contiguous window of --band_width layers with the largest summed (exact recall - shipped recall) * needle mass.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def needle_token_span(text, answer, offsets, lead=": "):
    """Token span [s, e) of `lead + answer` inside `text` given the tokenizer's (char_start, char_end) offsets.
    The answer must occur exactly once; if it is not preceded by `lead`, the span covers the answer alone.
    Tokens that straddle the span's first character are included."""
    if text.count(answer) != 1:
        raise ValueError(f"answer occurs {text.count(answer)} times in the text")
    a0 = text.index(answer)
    c0 = a0 - len(lead) if text[max(0, a0 - len(lead)):a0] == lead else a0
    c1 = a0 + len(answer)
    toks = [i for i, (s, e) in enumerate(offsets) if e > c0 and s < c1 and e > s]
    if not toks:
        raise ValueError("no token overlaps the needle")
    return toks[0], toks[-1] + 1


def dash_forced_and_rank(row, n_blocks, nb_full, device):
    """The Dash-8B selector's structure in the probe's terms (as in oracle_run._make_dash_ca): the shipped row is score
    ordered and -1 padded, its first two entries are the blocks the release always reads (the current partial chunk and
    the last full block), and candidates are the full blocks. Returns (forced list, valid mask [n_blocks], a rank list
    that starts with the shipped row and contains every block)."""
    shipped = [b for b in row if b >= 0]
    forced = shipped[:2]
    valid = torch.arange(n_blocks, device=device) < nb_full
    seen = set(shipped)
    return forced, valid, shipped + [b for b in range(n_blocks) if b not in seen]


def flag_heads(per_layer_head, tau=0.01):
    """The calibration rule. A KV head is flagged when its mass-weighted recall gap, (recall of the needle block under
    exact mass - recall under the shipped selector) x (the needle block's mean attention mass in that head's group),
    exceeds tau: the shipped selector misses the needle there AND the head reads the needle. per_layer_head is the
    probe's summary (lists [layer][kv head] under 'exact', 'ship', 'mass'). Returns ([(layer, head), ...] sorted by
    layer, {(layer, head): score})."""
    sc = {(l, h): (e - s) * m
          for l, (er, sr, mr) in enumerate(zip(per_layer_head["exact"], per_layer_head["ship"], per_layer_head["mass"]))
          for h, (e, s, m) in enumerate(zip(er, sr, mr))}
    return sorted(lh for lh, v in sc.items() if v > tau), sc


def shortfall_heads(per_layer_head, tau=0.01):
    """A LABEL-FREE variant of flag_heads: a KV head is flagged when exact mass selection captures at least tau more of the
    group's attention mass in the free slots than the shipped row does (means over the probed rows). It needs no needle
    position. Returns ([(layer, head), ...], {(layer, head): shortfall})."""
    sc = {(l, h): e - s
          for l, (er, sr) in enumerate(zip(per_layer_head["cap_exact"], per_layer_head["cap_ship"]))
          for h, (e, s) in enumerate(zip(er, sr))}
    return sorted(lh for lh, v in sc.items() if v > tau), sc


def best_window(values, width):
    sums = [sum(values[i:i + width]) for i in range(len(values) - width + 1)]
    return max(range(len(sums)), key=lambda i: sums[i])


def theory_record(qs, kk, nb, free, n_free, n_blocks, block, D, OR):
    """Statistics of one decode row of one KV head for the precision / resolution analysis.
    qs [grp, D] queries of the group, kk [tk, D] keys, nb the needle block, free [n_blocks] bool mask of free blocks,
    n_free the number of free slots. Scores are per-block log-masses; a block is SELECTED if its score is among the
    n_free largest free scores. margin = needle score minus the (n_free+1)-th largest free score of the other blocks
    (positive = selected with that much room), in nats."""
    tk = kk.shape[0]
    pad = n_blocks * block - tk
    neg_inf = -float("inf")

    def block_lse(logits):                                   # [g, tk] -> [g, n_blocks]
        if pad:
            logits = torch.cat([logits, logits.new_full((logits.shape[0], pad), neg_inf)], dim=-1)
        return torch.logsumexp(logits.view(logits.shape[0], n_blocks, block), dim=-1)

    def margin(score):                                       # score [n_blocks] (higher = better)
        sc = torch.where(free, score, torch.full_like(score, neg_inf))
        others = sc.clone(); others[nb] = neg_inf
        cut = others.topk(min(n_free, int(free.sum()) - 1)).values[-1]          # the n_free-th best OTHER free block
        return float(sc[nb] - cut)

    lg = qs @ kk.T / math.sqrt(D)                            # [grp, tk] exact per-head logits
    lse_h = block_lse(lg)                                    # per-head block log-mass
    tot_h = torch.logsumexp(lg, dim=-1, keepdim=True)
    mass_h = (lse_h - tot_h).exp()                           # [grp, n_blocks] per-head block mass
    heads_on_needle = int((mass_h[:, nb] > 0.05).sum())
    exact_score = mass_h.sum(0).clamp(min=1e-30).log()       # log of the head-summed mass (the exact criterion)
    lgm = qs.mean(0, keepdim=True) @ kk.T / math.sqrt(D)     # mean-query logits
    meanq_score = block_lse(lgm)[0]
    def ipr(logits, blocks):
        """Concentration of each head's attention INSIDE the given blocks, sum_j p(j | block)^2 (1 = one token holds
        the block's mass, 1/L = uniform), averaged over heads with the heads' masses on the block as weights."""
        out = []
        for bk in blocks:
            seg = logits[:, bk * block:min((bk + 1) * block, tk)]
            pj = torch.softmax(seg, dim=-1)
            w = torch.logsumexp(seg, dim=-1) - torch.logsumexp(logits, dim=-1)       # log mass of the block per head
            w = torch.softmax(w, dim=0)
            out.append(float((w * pj.pow(2).sum(-1)).sum()))
        return out

    sc_ex = torch.where(free, exact_score, torch.full_like(exact_score, neg_inf)); sc_ex[nb] = neg_inf
    cut_blocks = sc_ex.topk(min(n_free + 4, int(free.sum()) - 1)).indices[-8:].tolist()   # blocks around the cutoff
    rec = {"heads_on_needle": heads_on_needle, "needle_mass": float(mass_h[:, nb].sum() / qs.shape[0]),
           "ipr_needle_heads": ipr(lg, [nb])[0], "ipr_cut_heads": sum(ipr(lg, cut_blocks)) / len(cut_blocks),
           "ipr_needle_meanq": ipr(lgm, [nb])[0], "ipr_cut_meanq": sum(ipr(lgm, cut_blocks)) / len(cut_blocks),
           "margin_exact": margin(exact_score), "margin_meanq": margin(meanq_score),
           "logit_std_meanq": float(lgm.std()), "logit_std_heads": float(lg.std(dim=-1).mean()),
           "needle_z_meanq": float((lgm[0, nb * block:(nb + 1) * block].max() - lgm.mean()) / lgm.std().clamp(min=1e-9)),
           "bits": {}, "anatomy": None}
    # Proposition 1 on the data: in the query head with the most needle mass, the needle key's logit gap over the rest of
    # its block, the pooled (window-mean) elevation of the needle block and of the free block at the pooled cutoff, both
    # over the needle block's background, with InfLLM-V2's windows of 32 keys at a stride of 16 (five overlap a block)
    g = int(mass_h[:, nb].argmax())
    s_g = lg[g]
    lo_, hi_ = nb * block, min((nb + 1) * block, tk)
    blk = s_g[lo_:hi_]
    jl = int(blk.argmax())
    rest = torch.cat([blk[:jl], blk[jl + 1:]])
    mu = float(rest.mean()) if rest.numel() else float(blk[jl])
    W_, S_ = 32, 16
    if tk >= W_:
        wm = s_g.unfold(0, W_, S_).mean(-1)                                    # [n_windows] window-mean logits
        pooled = torch.full((n_blocks,), -float("inf"), device=wm.device)
        for off in (-16, 0, 16, 32, 48):                                      # windows overlapping each block
            st = torch.arange(n_blocks, device=wm.device) * block + off
            ok = (st >= 0) & (st + W_ <= tk)
            mi = (st.clamp(min=0) // S_).clamp(max=wm.shape[0] - 1)
            pooled = torch.where(ok & (wm[mi] > pooled), wm[mi], pooled)
        pf = torch.where(free, pooled, torch.full_like(pooled, -float("inf"))); pf_other = pf.clone(); pf_other[nb] = -float("inf")
        cut_p = pf_other.topk(min(n_free, int(free.sum()) - 1)).values[-1]
        lse_g = block_lse(lg[g:g + 1])[0]
        ef = torch.where(free, lse_g, torch.full_like(lse_g, -float("inf"))); ef_other = ef.clone(); ef_other[nb] = -float("inf")
        cut_e = ef_other.topk(min(n_free, int(free.sum()) - 1)).values[-1]
        fin = pooled[torch.isfinite(pooled)]
        med_win = float(fin.median()) if fin.numel() else mu                   # a reference independent of the needle block
        anatomy = {"Delta": float(blk[jl]) - mu, "needle_pool_elev": float(pooled[nb]) - mu, "cut_pool_elev": float(cut_p) - mu,
                   "needle_pool_vs_median": float(pooled[nb]) - med_win, "cut_pool_vs_median": float(cut_p) - med_win,
                   "background_vs_median": mu - med_win,
                   "in_pool_head": int(bool(pooled[nb] > cut_p)), "in_exact_head": int(bool(lse_g[nb] > cut_e)),
                   "needle_lse_minus_cut": float(lse_g[nb] - cut_e)}
    else:
        anatomy = None
    for b in (1, 2, 3, 4):
        kq = OR._quantize_keys(kk, b)
        lq = qs @ kq.T / math.sqrt(D)
        lqm = qs.mean(0, keepdim=True) @ kq.T / math.sqrt(D)
        hs = ((block_lse(lq) - torch.logsumexp(lq, dim=-1, keepdim=True)).exp().sum(0)).clamp(min=1e-30).log()
        ms = block_lse(lqm)[0]
        rec["bits"][b] = {"noise_std_meanq": float((lqm - lgm).std()), "noise_std_heads": float((lq - lg).std(dim=-1).mean()),
                          "margin_meanq": margin(ms), "margin_heads": margin(hs)}
    rec["anatomy"] = anatomy
    return rec


class Acc:
    """Per (layer, head) counters."""

    def __init__(self, L, H):
        z = lambda: torch.zeros(L, H, dtype=torch.float64)  # noqa: E731
        self.n, self.forced, self.ship, self.exact, self.low, self.mass = z(), z(), z(), z(), z(), z()
        self.rank_ship, self.rank_exact, self.rank_low = z(), z(), z()
        # label-free: attention mass captured by the free blocks of the shipped row / of exact mass / of the low-bit scorer
        self.cap_ship, self.cap_exact, self.cap_low = z(), z(), z()
        # needle mass the shipped row FORFEITS relative to exact selection, row by row: mass x (in_exact - in_ship). Its mean is
        # the expectation that the flagging score (a product of means) approximates, in units of attention mass.
        self.lost = z()

    def add(self, li, h, forced, in_ship, in_exact, in_low, mass, ranks, captured=(0.0, 0.0, 0.0)):
        if forced:
            self.forced[li, h] += 1
            return
        self.n[li, h] += 1
        self.ship[li, h] += in_ship; self.exact[li, h] += in_exact; self.low[li, h] += in_low
        self.mass[li, h] += mass
        self.lost[li, h] += mass * (int(bool(in_exact)) - int(bool(in_ship)))
        self.rank_ship[li, h] += ranks[0]; self.rank_exact[li, h] += ranks[1]; self.rank_low[li, h] += ranks[2]
        self.cap_ship[li, h] += captured[0]; self.cap_exact[li, h] += captured[1]; self.cap_low[li, h] += captured[2]

    def summary(self):
        n = self.n.clamp(min=1)
        keys = ("ship", "exact", "low", "mass", "rank_ship", "rank_exact", "rank_low", "cap_ship", "cap_exact", "cap_low", "lost")
        per_lh = {k: (getattr(self, k) / n).tolist() for k in keys}
        nl = self.n.sum(1).clamp(min=1)
        per_l = {k: (getattr(self, k).sum(1) / nl).tolist() for k in keys}
        return {"per_layer": per_l, "per_layer_head": per_lh, "n_per_layer": self.n.sum(1).tolist(),
                "forced_per_layer": self.forced.sum(1).tolist()}


PP_KEYS = ("n", "ship", "exact", "mass", "cap_ship", "cap_exact", "lost")


class PromptLog:
    """Per-prompt SUMS of the accumulator's counters (differences of successive snapshots), written to the result file so
    that the statistics, and hence the flagged heads, of ANY subset of the calibration prompts can be recomputed exactly
    offline (pheno/calib_stability.py: subsampling and bootstrap of the calibration set)."""

    def __init__(self, acc):
        self.prev = {k: getattr(acc, k).clone() for k in PP_KEYS}
        self.rows = []

    def mark(self, acc):
        cur = {k: getattr(acc, k).clone() for k in PP_KEYS}
        self.rows.append({k: (cur[k] - self.prev[k]).tolist() for k in PP_KEYS})
        self.prev = cur


def flags_from_prompt_sums(rows, idx, tau=0.01):
    """The flagging rule on a subset `idx` of per-prompt sums (PromptLog rows). Identical to flag_heads on the aggregate of
    those prompts: a head's recalls and needle mass are sums over the subset divided by its number of counted rows."""
    tot = {k: None for k in ("n", "ship", "exact", "mass")}
    for i in idx:
        for k in tot:
            t = torch.tensor(rows[i][k], dtype=torch.float64)
            tot[k] = t if tot[k] is None else tot[k] + t
    n = tot["n"].clamp(min=1)
    sc = ((tot["exact"] - tot["ship"]) / n) * (tot["mass"] / n)
    return sorted((int(l), int(h)) for l, h in (sc > tau).nonzero().tolist()), sc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data", default=None, help="calibration draw (RULER jsonl); must NOT be an evaluated draw")
    p.add_argument("--duo_booksum", default=None,
                   help="calibrate on DuoAttention's released multi-passkey items instead of --data (pheno/duo_calib.py): "
                        "path of BookSum's csv, whose chapters are the haystack")
    p.add_argument("--duo_items", type=int, default=10, help="items (ten passkeys each) with --duo_booksum")
    p.add_argument("--duo_len", type=int, default=32000, help="context length of the items (upstream trains up to 32000)")
    p.add_argument("--duo_passkey_len", type=int, default=32,
                   help="words per passkey (upstream default 32, about a whole 64-token block; 4 to 8 keeps a needle inside a block)")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--steps", type=int, default=12, help="needle tokens teacher-forced per prompt")
    p.add_argument("--dump_heads", default=None,
                   help="KV heads 'L.h,...' whose per-token attention is saved for figures (2026-10-01): per-query-head block "
                        "masses under full attention, the kernel's own group score, the shipped and exact selections, and the "
                        "per-key logits inside the needle block and its strongest competitor; written to <out>_dump.pt")
    p.add_argument("--dump_rows", type=int, default=2, help="prompts whose probed tokens --dump_heads saves")
    p.add_argument("--flag_tau", type=float, default=0.01,
                   help="a KV head is flagged for repair when (exact recall - shipped recall) x needle mass exceeds this")
    p.add_argument("--bits", type=int, default=2)
    p.add_argument("--band_width", type=int, default=8)
    p.add_argument("--theory_heads", default=None,
                   help="comma list of LAYER.KVHEAD pairs: for these KV heads record what the precision / resolution theory "
                        "needs: the needle block's score margin over the selection cutoff under exact per-head mass, under "
                        "the mean query and under pooled keys; the logit standard deviation over keys; the measured logit "
                        "noise and the needle recall at 1, 2, 3, 4 bits for the mean-query and the per-head criterion; "
                        "the number of query heads of the group that put more than 5 percent of their mass on the block")
    p.add_argument("--qhead_detail", default=None,
                   help="comma list of LAYER.KVHEAD pairs: for these KV heads also record, per QUERY head of the group, the "
                        "needle block's mass and rank under that head's own token-level softmax and under its own "
                        "softmax over the SHIPPED compressed keys (max-pooled to blocks by the kernel's pooling)")
    p.add_argument("--prefill_rows", type=int, default=0,
                   help="PREFILL-TIME probe: also record, at the last K query positions of the prompt (the "
                        "question), whether the needle block is in the shipped selection row and in the exact-mass row at the "
                        "same free budget, and the needle mass there, per KV head, from the single prefill pass; reported as "
                        "'prefill' with its own flagged set. 0 = off")
    p.add_argument("--retrieval_score", action="store_true",
                   help="also record Wu et al.'s retrieval score (arXiv 2404.15574) per query head: at each teacher-forced needle "
                        "token, a hit when the head's most-attended key is a needle position holding the token being copied")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if (a.data is None) == (a.duo_booksum is None):
        raise SystemExit("give exactly one of --data and --duo_booksum -- ABORT")
    if a.data is not None and "seed=42" in a.data:
        raise SystemExit("seed=42 draws are the evaluated ones: calibrate on a different seed -- ABORT")
    import oracle_run as OR
    from transformers.cache_utils import DynamicCache
    ha = OR.build_parser().parse_args(["--model_dir", a.model_dir, "--data", a.data or "unused", "--task", "niah_multikey_3",
                                       "--ruler_root", "unused", "--target_len", "0", "--out", "/dev/null",
                                       "--no_oracle", "--no_install"])
    model, tok, cfg, _state = OR.build_model_and_state(ha)
    mod = sys.modules[type(model).__module__]
    is_dash = hasattr(mod, "triton_entmax")          # adaptive-budget selector: rows are score-ordered with realized widths
    if is_dash and (a.theory_heads or a.qhead_detail):
        raise SystemExit("--theory_heads / --qhead_detail read the InfLLM-V2 kernel's compressed keys -- ABORT")
    block = cfg.sparse_config["block_size"]
    raw_ca = mod.compressed_attention
    sig = inspect.signature(raw_ca)
    L, Hkv = cfg.num_hidden_layers, cfg.num_key_value_heads
    acc = Acc(L, Hkv)
    acc_pre = Acc(L, Hkv)                                        # the prefill-time probe (--prefill_rows)
    plog = PromptLog(acc)
    PROBE, errors = {"src_block": None}, []
    theory_heads = OR._parse_kv_heads(a.theory_heads) or frozenset()
    theory = {lh: [] for lh in theory_heads}                     # one record per (row, step)
    dump_heads = OR._parse_kv_heads(a.dump_heads) or frozenset()
    DUMP = {"heads": dump_heads, "rows": a.dump_rows, "records": []} if dump_heads else None
    detail_heads = OR._parse_kv_heads(a.qhead_detail) or frozenset()
    grp_n = cfg.num_attention_heads // Hkv
    detail = {lh: {k: torch.zeros(grp_n, dtype=torch.float64) for k in
                   ("n", "mass_tok", "rank_tok", "in_tok", "rank_pool", "in_pool")} for lh in detail_heads}
    retr_hits = torch.zeros(L, Hkv, grp_n, dtype=torch.float64) if a.retrieval_score else None
    retr_n = torch.zeros(L, Hkv, dtype=torch.float64) if a.retrieval_score else None
    razor = ({"echo": torch.zeros(L, Hkv, grp_n, dtype=torch.float64), "induction": torch.zeros(L, Hkv, grp_n, dtype=torch.float64)}
             if a.retrieval_score else None)

    def _prefill_probe(qk, topk_idx, B, nb, li):
        """At the last K = --prefill_rows query positions of the prompt, per KV head: is the needle block nb in the shipped
        row, in the exact-mass row at the same free budget, and how much mass does the group put on it. The forced
        blocks are the kernel's own structure (init blocks, local window, the query's block), as in
        oracle_run._prefill_tail_rows, and the masses are the per-head exact masses of those rows (chunks of 32)."""
        q, k, _v = qk
        T, Hq, D = q.shape
        Hkv = k.shape[1]
        grp = Hq // Hkv
        K = min(int(a.prefill_rows), T)
        n_blocks = math.ceil(T / block)
        pad = n_blocks * block - T
        init_blocks, local_blocks = int(B["init_blocks"]), int(B["local_blocks"])
        pos = torch.arange(T - K, T, device=q.device)
        kidx = torch.arange(T, device=q.device)
        bidx = torch.arange(n_blocks, device=q.device).view(1, n_blocks)
        qb = (pos // block).view(-1, 1)
        valid = bidx <= qb
        forced = ((bidx < init_blocks) | ((bidx >= qb - local_blocks) & (bidx <= qb))) & valid       # [K, n_blocks]
        nb_forced = forced[:, nb]
        for h in range(Hkv):
            qt = q[T - K:, h * grp:(h + 1) * grp].float()                 # [K, grp, D]
            kk = k[:, h].float()
            srow = topk_idx[h, T - K:T].long()                             # [K, nbk] shipped rows
            ship = torch.zeros(K, n_blocks + 1, dtype=torch.bool, device=q.device)
            ship.scatter_(1, torch.where(srow >= 0, srow, torch.full_like(srow, n_blocks)), True)
            ship = ship[:, :n_blocks]
            n_free = (ship & ~forced).sum(-1)                              # free slots the shipped row fills
            in_ship = ship[:, nb]
            masses = []
            for c0 in range(0, K, 32):
                lg = torch.einsum("kgd,td->kgt", qt[c0:c0 + 32], kk) / math.sqrt(D)
                causal = kidx.view(1, 1, T) <= pos[c0:c0 + 32].view(-1, 1, 1)
                pm = torch.softmax(lg.masked_fill(~causal, -float("inf")), dim=-1).sum(1)          # [k, T]
                if pad:
                    pm = torch.cat([pm, pm.new_zeros(pm.shape[0], pad)], dim=-1)
                masses.append(pm.view(pm.shape[0], n_blocks, block).sum(-1))
            mc = torch.cat(masses)                                          # [K, n_blocks] group mass per block
            rank = mc.masked_fill(~valid | forced, -float("inf"))
            order = rank.argsort(dim=-1, descending=True)                   # exact free order per row
            nb_rank = (order == nb).float().argmax(-1)                       # position of the needle block in that order
            in_exact = nb_rank < n_free
            mass = mc[:, nb] / grp
            for r in range(K):
                if bool(nb_forced[r]):
                    acc_pre.add(li, h, True, 0, 0, 0, 0.0, (0, 0, 0)); continue
                acc_pre.add(li, h, False, int(in_ship[r]), int(in_exact[r]), int(in_exact[r]), float(mass[r]), (0, 0, 0))

    def ca(*args, **kw):
        out = raw_ca(*args, **kw)
        topk_idx = out[0] if isinstance(out, tuple) else out
        qk = OR.STASH.get("qk")
        if a.prefill_rows and PROBE.get("pre_block") is not None and qk is not None and qk[0].shape[0] > 1 and not is_dash:
            try:                                             # _bind_selector_args returns None at prefill: bind the raw signature
                ba = sig.bind(*args, **kw); ba.apply_defaults()
                _prefill_probe(qk, topk_idx, ba.arguments, PROBE["pre_block"], OR.STASH.get("li", 0))
            except Exception as e:
                errors.append(f"prefill {type(e).__name__}: {e}")
            return out
        if PROBE["src_block"] is None or qk is None or qk[0].shape[0] != 1:
            return out
        try:
            kinfo = None if is_dash else OR._kernel_block_scores(mod, sig, args, kw)
            if kinfo is None and not is_dash:
                return out
            q, k, _v = qk
            li = OR.STASH.get("li", 0)
            Hq, D = q.shape[1], q.shape[2]
            tk = k.shape[0]
            grp = Hq // k.shape[1]
            if retr_hits is not None and PROBE.get("span") is not None:
                s_, e_ = PROBE["span"]
                ids_full = PROBE.get("ids_full")         # prompt ids followed by the teacher-forced tokens fed so far
                for h in range(k.shape[1]):              # Wu et al.'s retrieval score, every KV head, from the exact logits
                    top = (q[0, h * grp:(h + 1) * grp].float() @ k[:, h].float().T).argmax(-1)     # [grp] most-attended key
                    inside = (top >= s_) & (top < e_)
                    same = PROBE["ids"][top.clamp(max=PROBE["ids"].shape[0] - 1)] == PROBE["target"]
                    retr_hits[li, h] += (inside & same).double().cpu()
                    retr_n[li, h] += 1
                    if razor is not None and ids_full is not None and PROBE.get("cur") is not None:
                        # RazorAttention's criteria (Tang et al. 2024): an ECHO head attends to an earlier copy of
                        # the current token; an INDUCTION head attends to the token that followed an earlier copy of it
                        tt = top.clamp(max=ids_full.shape[0] - 1)
                        razor["echo"][li, h] += (ids_full[tt] == PROBE["cur"]).double().cpu()
                        razor["induction"][li, h] += (ids_full[(tt - 1).clamp(min=0)] == PROBE["cur"]).double().cpu()
            n_blocks = math.ceil(tk / block)
            pad = n_blocks * block - tk
            nb = PROBE["src_block"]
            for h in range(k.shape[1]):
                if is_dash:
                    forced, valid, kf = dash_forced_and_rank(topk_idx[h, 0].tolist(), n_blocks, tk // block, q.device)
                else:
                    forced, valid, kf = OR._forced_and_rank(kinfo[0][h], n_blocks, kinfo[1])
                if not forced:
                    continue                                     # an empty adaptive row: nothing to compare
                if nb in forced:
                    acc.add(li, h, True, 0, 0, 0, 0.0, (0, 0, 0)); continue
                qs = q[0, h * grp:(h + 1) * grp].float()
                kk = k[:, h].float()
                pm = torch.softmax(qs @ kk.T / math.sqrt(D), dim=-1).sum(0) / grp
                pl = OR._meanq_mass(qs, OR._quantize_keys(kk, a.bits), D)
                if pad:
                    pm = torch.cat([pm, pm.new_zeros(pad)]); pl = torch.cat([pl, pl.new_zeros(pad)])
                mass = pm.view(n_blocks, block).sum(-1)
                low = pl.view(n_blocks, block).sum(-1)
                row = [b for b in topk_idx[h, 0].tolist() if b >= 0]
                fset = set(forced)
                n_free = len([b for b in row if b not in fset])
                free = valid.clone()
                free[torch.tensor(forced, device=free.device)] = False
                neg = torch.full_like(mass, -1.0)
                ex_order = torch.where(free, mass, neg).argsort(descending=True).tolist()
                lo_order = torch.where(free, low, neg).argsort(descending=True).tolist()
                free_ship = [b for b in row if b not in fset]
                captured = (float(mass[free_ship].sum()) if free_ship else 0.0, float(mass[ex_order[:n_free]].sum()),
                            float(mass[lo_order[:n_free]].sum()))
                acc.add(li, h, False, int(nb in row), int(nb in ex_order[:n_free]), int(nb in lo_order[:n_free]),
                        float(mass[nb]), (kf.index(nb), ex_order.index(nb), lo_order.index(nb)), captured)
                if DUMP is not None and (li, h) in DUMP["heads"] and PROBE.get("row_i", 0) < DUMP["rows"]:
                    # attention for figures: this token's per-query-head block masses under full attention, the kernel's own
                    # block scores (the criterion the shipped selector ranks by), the shipped and exact selections, and every
                    # query head's per-key logits inside the needle block and inside the best other free block under that
                    # group score (the strongest competitor)
                    lg_d = qs @ kk.T / math.sqrt(D)                                            # [grp, tk]
                    ph_d = torch.softmax(lg_d, dim=-1)
                    if pad:
                        ph_d = torch.cat([ph_d, ph_d.new_zeros(grp, pad)], dim=-1)
                    mq = ph_d.view(grp, n_blocks, block).sum(-1)                               # [grp, n_blocks]
                    # the kernel's own block scores (oracle_run._kernel_block_scores: the stage-1 kernel on both compressed
                    # key sets, max-pooled onto blocks, +inf on the forced blocks), the scores the shipped row is the top of
                    gp = kinfo[0][h].float()
                    gpf = torch.full((n_blocks,), float("nan"), device=gp.device)
                    n1 = min(n_blocks, gp.shape[0])
                    gpf[:n1] = gp[:n1]
                    cand = torch.where(free & torch.isfinite(gpf), gpf, torch.full_like(gpf, -float("inf")))
                    cand[nb] = -float("inf")
                    cb = int(cand.argmax())
                    seg = lambda b_: lg_d[:, b_ * block:min((b_ + 1) * block, tk)]           # noqa: E731
                    DUMP["records"].append({"layer": li, "head": h, "row": PROBE.get("row_i"), "step": PROBE.get("step_i"),
                                            "tk": tk, "block": block, "nb": nb, "n_free": n_free, "forced": list(forced),
                                            "ship_free": list(free_ship), "exact_top": list(ex_order[:n_free]),
                                            "competitor": cb, "mass_q": mq.half().cpu(), "group_pool": gpf.cpu(),
                                            "needle_logits": seg(nb).half().cpu(), "competitor_logits": seg(cb).half().cpu()})
                if (li, h) in theory:
                    trec = theory_record(qs, kk, nb, free, n_free, n_blocks, block, D, OR)
                    if trec["anatomy"] is not None:
                        trec["anatomy"]["in_ship"] = int(nb in row)            # the shipped selector's verdict on this token
                    theory[(li, h)].append(trec)
                if (li, h) in detail:
                    # per QUERY head of this group: token-level mass vs the shipped compressed keys, each head alone
                    B = OR._bind_selector_args(sig, args, kw)
                    ph = torch.softmax(qs @ kk.T / math.sqrt(D), dim=-1)                       # [grp, tk]
                    if pad:
                        ph = torch.cat([ph, ph.new_zeros(grp, pad)], dim=-1)
                    mtok = ph.view(grp, n_blocks, block).sum(-1)                               # [grp, n_blocks]
                    ck = B["k"][:, h].float()                                                  # [n_c, D] compressed keys
                    qd = B["q"][0, h * grp:(h + 1) * grp].float()                              # [grp, D]
                    sc = torch.softmax(qd @ ck.T / math.sqrt(D), dim=-1)                       # [grp, n_c]
                    pooled = mod.max_pooling_1d_varlen(sc.unsqueeze(1).to(torch.bfloat16).contiguous(), B["cu_seqlens_q"],
                                                       B["cu_seqlens_k"], B["cache_lens"], B["max_seqlen_q"],
                                                       B["max_seqlen_k"], local_blocks=B["local_blocks"],
                                                       init_blocks=B["init_blocks"], block_size=B["block_size"],
                                                       stride=B["kernel_stride"])[:, 0, :].float()   # [grp, n_blocks']
                    n1 = min(n_blocks, pooled.shape[1])
                    if (li, h) in theory and theory[(li, h)] and theory[(li, h)][-1].get("anatomy"):
                        # The GROUP score the kernel ranks with, on this token. Stage 1 sums
                        # the per-head softmaxes over the compressed keys across the group's query heads and then
                        # max-pools the windows of a block, so the group score is the pool of the summed softmax, not
                        # the sum of the per-head pools above. Recorded next to the single-head verdicts of the anatomy:
                        # the needle block's group score, the n_free-th best OTHER free block's (the cutoff), their log
                        # margin, and the group score's own verdict, which must agree with the shipped row (in_ship).
                        gsc = sc.sum(0, keepdim=True)                                              # [1, n_c]
                        gpool = mod.max_pooling_1d_varlen(gsc.unsqueeze(1).to(torch.bfloat16).contiguous(), B["cu_seqlens_q"],
                                                          B["cu_seqlens_k"], B["cache_lens"], B["max_seqlen_q"],
                                                          B["max_seqlen_k"], local_blocks=B["local_blocks"],
                                                          init_blocks=B["init_blocks"], block_size=B["block_size"],
                                                          stride=B["kernel_stride"])[0, 0, :].float()   # [n_blocks']
                        ng = min(n_blocks, gpool.shape[0])
                        gs = torch.full((n_blocks,), -1.0, device=free.device)
                        gs[:ng] = torch.where(free[:ng], gpool[:ng].clamp(max=1e30), neg[:ng])
                        g_other = gs.clone(); g_other[nb] = -1.0
                        cut_g = float(g_other.topk(min(n_free, int(free.sum()) - 1)).values[-1])
                        g_order = gs.argsort(descending=True).tolist()
                        an = theory[(li, h)][-1]["anatomy"]
                        an["group_needle_score"] = float(gs[nb]); an["group_cut_score"] = cut_g
                        an["group_margin_log"] = math.log(max(float(gs[nb]), 1e-30)) - math.log(max(cut_g, 1e-30))
                        an["group_in_top"] = int(nb in g_order[:n_free])
                        an["group_heads_on_needle_pooled"] = int((pooled[:, nb] > 0.05).sum()) if nb < n1 else 0
                        # per QUERY head of the group, for R3's split of the group-sum misses: the head's exact needle mass
                        # (token level) and whether ITS OWN pooled score keeps the needle block at the free budget
                        qh = []
                        for g in range(grp):
                            pp_g = torch.full((n_blocks,), -1.0, device=free.device)
                            pp_g[:n1] = torch.where(free[:n1], pooled[g, :n1].clamp(max=1e30), neg[:n1])
                            qh.append([round(float(mtok[g, nb]), 4), int(nb in pp_g.argsort(descending=True).tolist()[:n_free])])
                        an["qheads_mass_and_pooled_keep"] = qh
                    dd = detail[(li, h)]
                    for g in range(grp):
                        t_order = torch.where(free, mtok[g], neg).argsort(descending=True).tolist()
                        pp = torch.full((n_blocks,), -1.0, device=free.device)
                        pp[:n1] = torch.where(free[:n1], pooled[g, :n1].clamp(max=1e30), neg[:n1])
                        p_order = pp.argsort(descending=True).tolist()
                        rt, rp = t_order.index(nb), p_order.index(nb)
                        dd["n"][g] += 1; dd["mass_tok"][g] += float(mtok[g, nb])
                        dd["rank_tok"][g] += rt; dd["in_tok"][g] += int(rt < n_free)
                        dd["rank_pool"][g] += rp; dd["in_pool"][g] += int(rp < n_free)
        except Exception as e:                                   # a probe must never change the model's behaviour
            errors.append(f"{type(e).__name__}: {e}")
        return out
    mod.compressed_attention = ca
    OR._install_stash(model)

    tf_hits = tf_total = 0
    used = 0
    duo_meta = None
    if a.duo_booksum:
        # DuoAttention's released items: the context is prefilled, the question part is teacher-forced through the decode
        # path, and the probe records the first --steps tokens of each of the ten passkeys (the needle block of a step
        # is the block that holds the passkey token being copied)
        import duo_calib as DC
        items = DC.build_items(tok, DC.booksum_chapters(a.duo_booksum), a.duo_items, context_len=a.duo_len, seed=7,
                               passkey_length=a.duo_passkey_len)
        duo_meta = {"upstream": "mit-han-lab/duo-attention duo_attn/data.py", "commit": DC.UPSTREAM_COMMIT,
                    "haystack": os.path.abspath(a.duo_booksum), "items": len(items), "context_len": a.duo_len,
                    "haystack_sha256_12": hashlib.sha256(open(a.duo_booksum, "rb").read()).hexdigest()[:12],
                    "passkeys_per_item": len(items[0]["spans"]), "tokens_per_passkey_probed": a.steps,
                    "passkey_words": a.duo_passkey_len}
        for ii, it in enumerate(items):
            ids = torch.tensor(it["input_ids"], device="cuda").view(1, -1)
            sched = {dst + t: src + t for (src, dst), pk in zip(it["spans"], it["passkeys"])
                     for t in range(min(a.steps, len(pk)))}
            qa, last = it["qa_start"], max(sched)
            with torch.inference_mode():
                PROBE["src_block"] = None
                pre = model.model(input_ids=ids[:, :qa - 1], attention_mask=torch.ones_like(ids[:, :qa - 1]),
                                  past_key_values=DynamicCache(), use_cache=True)
                cache = pre.past_key_values
                for pos in range(qa - 1, last):                  # feed token pos, the model predicts token pos + 1
                    tgt = pos + 1
                    PROBE["src_block"] = (sched[tgt] // block) if tgt in sched else None
                    o = model(input_ids=ids[:, pos:pos + 1],
                              attention_mask=torch.ones(1, pos + 1, dtype=torch.long, device=ids.device),
                              past_key_values=cache, use_cache=True)
                    cache = o.past_key_values
                    if tgt in sched:
                        tf_hits += int(o.logits[0, -1].argmax() == ids[0, tgt]); tf_total += 1
                PROBE["src_block"] = None
            used += 1
            plog.mark(acc)
            print(f"  [duo item {ii}/{len(items)}] {ids.shape[1]} tokens, {len(sched)} probed steps, tf_acc {tf_hits}/{tf_total} "
                  f"errors {len(errors)}", flush=True)
    rows = [json.loads(l) for l in open(a.data)][: a.n] if a.data else []
    for ri, r in enumerate(rows):
        text = r["input"] + r.get("answer_prefix", "")
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        ids = enc.input_ids.cuda()
        s, e = needle_token_span(text, r["outputs"][0], enc.offset_mapping[0].tolist())
        cont = ids[0, s:e][: a.steps]
        with torch.inference_mode():
            PROBE["row_i"] = ri
            PROBE["src_block"] = None
            PROBE["pre_block"] = (s // block) if a.prefill_rows else None       # the prefill-time probe's needle block
            pre = model.model(input_ids=ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]),
                              past_key_values=DynamicCache(), use_cache=True)
            PROBE["pre_block"] = None
            cache = pre.past_key_values
            nxt = ids[:, -1:]
            for i in range(cont.shape[0]):
                PROBE["src_block"] = (s + i) // block
                PROBE["step_i"] = i
                if retr_hits is not None:
                    PROBE["span"], PROBE["ids"], PROBE["target"] = (s, e), ids[0], cont[i]
                    PROBE["cur"] = nxt.view(-1)[0]                                   # the token fed at this step
                    PROBE["ids_full"] = torch.cat([ids[0], cont[:i]])                # every token in the cache at this step
                T_now = ids.shape[1] + i
                o = model(input_ids=nxt, attention_mask=torch.ones(1, T_now, dtype=torch.long, device=ids.device),
                          past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                tf_hits += int(o.logits[0, -1].argmax() == cont[i]); tf_total += 1
                nxt = cont[i].view(1, 1)
            PROBE["src_block"] = None
            PROBE["span"] = None
        used += 1
        plog.mark(acc)
        if ri % 5 == 0:
            print(f"  [{ri}/{len(rows)}] needle tokens {s}-{e} block {s // block} tf_acc {tf_hits}/{tf_total} "
                  f"errors {len(errors)}", flush=True)
    if errors or used == 0 or float(acc.n.sum()) == 0:
        raise SystemExit(f"NEEDLE_PROBE INVALID: rows={used} counted={float(acc.n.sum())} errors={errors[:3]} -- ABORT")
    S = acc.summary()
    pl = S["per_layer"]
    gap_w = [(x - y) * m for x, y, m in zip(pl["exact"], pl["ship"], pl["mass"])]
    gap = [x - y for x, y in zip(pl["exact"], pl["ship"])]
    s0, s1 = best_window(gap_w, a.band_width), best_window(gap, a.band_width)
    qd_out = {f"{l}.{h}": {k: (v / d["n"].clamp(min=1)).tolist() if k != "n" else v.tolist() for k, v in d.items()}
              for (l, h), d in detail.items()}
    th_out = {}
    for (l, h), recs in theory.items():
        if not recs:
            continue
        n = len(recs)
        agg = {"n": n, "heads_on_needle_mean": sum(r["heads_on_needle"] for r in recs) / n,
               "needle_mass_mean": sum(r["needle_mass"] for r in recs) / n,
               "logit_std_meanq": sum(r["logit_std_meanq"] for r in recs) / n,
               "logit_std_heads": sum(r["logit_std_heads"] for r in recs) / n,
               "needle_z_meanq_mean": sum(r["needle_z_meanq"] for r in recs) / n,
               "ipr_needle_heads": [r["ipr_needle_heads"] for r in recs], "ipr_cut_heads": [r["ipr_cut_heads"] for r in recs],
               "ipr_needle_meanq": [r["ipr_needle_meanq"] for r in recs], "ipr_cut_meanq": [r["ipr_cut_meanq"] for r in recs],
               "recall_exact": sum(r["margin_exact"] > 0 for r in recs) / n,
               "recall_meanq": sum(r["margin_meanq"] > 0 for r in recs) / n,
               "margins_exact": [r["margin_exact"] for r in recs], "margins_meanq": [r["margin_meanq"] for r in recs],
               "anatomy": [r["anatomy"] for r in recs if r.get("anatomy")],
               "bits": {}}
        for b in (1, 2, 3, 4):
            agg["bits"][b] = {"noise_std_meanq": sum(r["bits"][b]["noise_std_meanq"] for r in recs) / n,
                              "noise_std_heads": sum(r["bits"][b]["noise_std_heads"] for r in recs) / n,
                              "recall_meanq": sum(r["bits"][b]["margin_meanq"] > 0 for r in recs) / n,
                              "recall_heads": sum(r["bits"][b]["margin_heads"] > 0 for r in recs) / n}
        th_out[f"{l}.{h}"] = agg
    flagged, flag_sc = flag_heads(S["per_layer_head"], a.flag_tau)
    sf_flagged, sf_sc = shortfall_heads(S["per_layer_head"], a.flag_tau)       # label-free, reported for comparison only
    retr = None
    if retr_hits is not None:
        rs = retr_hits / retr_n.clamp(min=1).unsqueeze(-1)                       # [L, Hkv, grp] per query head
        retr = {"per_query_head": {f"{l}.{h}": rs[l, h].tolist() for l in range(L) for h in range(Hkv)},
                "kv_max": {f"{l}.{h}": float(rs[l, h].max()) for l in range(L) for h in range(Hkv)},
                "kv_mean": {f"{l}.{h}": float(rs[l, h].mean()) for l in range(L) for h in range(Hkv)},
                "steps": retr_n.tolist(), "definition": "Wu et al. 2024 retrieval score under teacher forcing: hit when the "
                "argmax of the exact logits of a query head is a needle position holding the target token"}
        if razor is not None:
            rz = {}
            for name, hits in razor.items():
                r_ = hits / retr_n.clamp(min=1).unsqueeze(-1)
                rz[name] = {"per_query_head": {f"{l}.{h}": r_[l, h].tolist() for l in range(L) for h in range(Hkv)},
                            "kv_max": {f"{l}.{h}": float(r_[l, h].max()) for l in range(L) for h in range(Hkv)},
                            "kv_mean": {f"{l}.{h}": float(r_[l, h].mean()) for l in range(L) for h in range(Hkv)}}
            rz["definition"] = ("RazorAttention (Tang et al. 2024) criteria under the same teacher forcing: echo = the query head's "
                                "argmax key holds an earlier copy of the current token; induction = it holds the token that followed "
                                "an earlier copy of the current token; rates over the probed steps, per query head")
            retr["razor"] = rz
    prefill = None
    if a.prefill_rows and float(acc_pre.n.sum()) > 0:
        SP = acc_pre.summary()
        pf, pf_sc = flag_heads(SP["per_layer_head"], a.flag_tau)
        prefill = {"rows": a.prefill_rows, "flagged_kv_heads": ",".join(f"{l}.{h}" for l, h in pf),
                   "flag_scores": {f"{l}.{h}": v for (l, h), v in pf_sc.items()}, **SP,
                   "definition": "at the last --prefill_rows query positions of the prompt, in the prefill pass: shipped and "
                                 "exact-mass needle recall at the shipped free budget and the needle mass, per KV head"}
    res = {"theory": th_out, "qhead_detail": qd_out, "retrieval": retr, "prefill": prefill,
           "flag_tau": a.flag_tau, "flagged_kv_heads": ",".join(f"{l}.{h}" for l, h in flagged),
           "flag_scores": {f"{l}.{h}": v for (l, h), v in flag_sc.items()},
           "per_prompt": plog.rows,
           "shortfall_flagged_kv_heads": ",".join(f"{l}.{h}" for l, h in sf_flagged),
           "shortfall_scores": {f"{l}.{h}": v for (l, h), v in sf_sc.items()},
           # the expectation the flagging score approximates: mean over rows of needle mass x (in exact row - in shipped row)
           "lost_flagged_kv_heads": ",".join(f"{l}.{h}" for l, row in enumerate(S["per_layer_head"]["lost"]) for h, v in enumerate(row) if v > a.flag_tau),
           "model_dir": os.path.abspath(a.model_dir), "data": os.path.abspath(a.data) if a.data else None, "duo_calib": duo_meta,
           "n_rows": used, "steps": a.steps,
           "bits": a.bits, **S, "teacher_forced_acc": tf_hits / max(tf_total, 1),
           "band_by_mass_weighted_gap": [s0, s0 + a.band_width - 1], "band_by_gap": [s1, s1 + a.band_width - 1],
           "provenance": {"argv": sys.argv, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                          "data_sha256_12": (hashlib.sha256(open(a.data, "rb").read()).hexdigest()[:12] if a.data else None)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    if DUMP is not None:
        dp = str(Path(a.out).with_suffix("")) + "_dump.pt"
        torch.save({"records": DUMP["records"], "argv": sys.argv, "data": a.data}, dp)
        print(f"DUMP {len(DUMP['records'])} records -> {dp}", flush=True)
    print("layer  n     ship   exact  lowbit  mass    rank_ship rank_exact rank_low")
    for li in range(L):
        print(f"{li:5d} {int(S['n_per_layer'][li]):5d}  {pl['ship'][li]:.3f}  {pl['exact'][li]:.3f}  {pl['low'][li]:.3f}  "
              f"{pl['mass'][li]:.4f}  {pl['rank_ship'][li]:8.1f} {pl['rank_exact'][li]:9.1f} {pl['rank_low'][li]:8.1f}")
    for lh, t in th_out.items():
        print(f"theory stats KV head {lh}: n={t['n']} heads_on_needle={t['heads_on_needle_mean']:.1f}/{grp_n} "
              f"logit std meanq/heads={t['logit_std_meanq']:.2f}/{t['logit_std_heads']:.2f} needle z (meanq)={t['needle_z_meanq_mean']:.1f} "
              f"recall exact/meanq={t['recall_exact']:.2f}/{t['recall_meanq']:.2f}")
        for b in (1, 2, 3, 4):
            x = t["bits"][b]
            print(f"     {b} bit: logit noise std meanq/heads {x['noise_std_meanq']:.3f}/{x['noise_std_heads']:.3f}  "
                  f"recall meanq/heads {x['recall_meanq']:.2f}/{x['recall_heads']:.2f}")
    for lh, d in qd_out.items():
        print(f"query heads of KV head {lh}: needle mass (token level) | in top-n_free by own token-level mass | by own pooled-key score")
        for g in range(len(d["n"])):
            print(f"   q{g:2d}  mass {d['mass_tok'][g]:.3f}  tok {d['in_tok'][g]:.2f} (rank {d['rank_tok'][g]:6.1f})  "
                  f"pooled {d['in_pool'][g]:.2f} (rank {d['rank_pool'][g]:6.1f})")
    print(f"FLAGGED_KV_HEADS tau={a.flag_tau}: {res['flagged_kv_heads']}", flush=True)
    if retr is not None:
        top = sorted(retr["kv_max"].items(), key=lambda kv: -kv[1])[:10]
        print("RETRIEVAL_TOP10 (max over the group's query heads): " + " ".join(f"{k}={v:.3f}" for k, v in top), flush=True)
    top_sf = sorted(sf_sc.items(), key=lambda kv: -kv[1])[:8]
    print(f"LOST_MASS_FLAGGED tau={a.flag_tau}: {res['lost_flagged_kv_heads']}", flush=True)
    print("SHORTFALL_TOP8 (label-free captured-mass shortfall): " + " ".join(f"{l}.{h}={v:.4f}" for (l, h), v in top_sf), flush=True)
    print(f"NEEDLE_PROBE_DONE rows={used} tf_acc={res['teacher_forced_acc']:.3f} band_weighted={res['band_by_mass_weighted_gap']} "
          f"band_gap={res['band_by_gap']}", flush=True)


if __name__ == "__main__":
    main()
