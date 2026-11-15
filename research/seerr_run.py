#!/usr/bin/env python
"""RULER cells and the needle probe for SeerAttention-R decode gates on Qwen3 (microsoft/SeerAttention, MIT).

A second selector family for the diagnosis and the training-free repair: a LEARNED gate that scores blocks of 64 keys
from max/min/avg-pooled pre-RoPE keys and one projected query per GQA group, with dense prefill and block-sparse decode
(the release's own kernel). The base weights are untouched, so the dense arm is the same model.

Arms (--arm):
  dense    the release's dense path (seerattn_implementation='seer_dense')
  gate     the released AttnGate at --token_budget (the shipped selector)
  exact    exact GQA-group attention mass in the free slots (our instrument; --kv_heads / --layers restrict it)
  lowbit   the repair: each query head's softmax over a --key_bits copy of the keys, summed over the group
  full     full attention in the --kv_heads: every valid block at decode (prefill is dense in every arm), the selection
           fix of the threshold rule (2026-10-04), which reads more blocks than the gate by design
All selection arms read exactly as many blocks as the gate's row: the kernel always reads block 0
(first_block_unmasked=True) and the gate always keeps the last block, so forced = {0, last}, the free budget of a row
is the number of other blocks in the gate's mask, and an override fills the same number of free slots.
--probe: teacher-forces the answer tokens of a calibration draw through the decode path and records, per layer and
KV head, whether the needle block is in the gate's row, whether it would be under exact mass, and the needle's mass
(pheno/needle_probe.py's statistics and flagging rule).

The hook wraps the module-level `sparse_flash_attention_forward` that Qwen3SeerAttention.forward calls at decode, where
the post-RoPE query, the full key cache and the gate's block mask are all in hand. Every override is counted, errors
abort the run, and a run whose hook never fired is refused.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STASH = {}


def free_override(mask_row, scores, n_blocks_valid):
    """mask_row [n_blocks] bool (the gate's row), scores [n_blocks] (higher is better), n_blocks_valid = number of blocks
    that hold at least one cached key. Returns a new bool row with the same forced blocks {0, last valid} and the same
    number of free blocks, chosen by `scores`."""
    last = n_blocks_valid - 1
    forced = torch.zeros_like(mask_row)
    forced[0] = True; forced[last] = True
    valid = torch.zeros_like(mask_row); valid[:n_blocks_valid] = True
    n_free = int((mask_row & valid & ~forced).sum())
    cand = valid & ~forced
    out = forced.clone()
    if n_free > 0:
        s = scores.masked_fill(~cand, -float("inf"))
        out[s.topk(min(n_free, int(cand.sum()))).indices] = True
    return out


def group_mass(q, k, bits, quantize, block):
    """q [Hq, D] post-RoPE decode queries, k [T, Hkv, D] post-RoPE keys. Returns [Hkv, n_blocks]: each query head's
    softmax over the (optionally b-bit) keys, summed over the group and over each block (pheno/oracle_run._heads_mass)."""
    T, Hkv, D = k.shape
    G = q.shape[0] // Hkv
    kk = k.float()
    if bits is not None:                                   # per-channel statistics: every (head, channel) column on its own
        kk = quantize(kk.reshape(T, Hkv * D), bits, rot=False).view(T, Hkv, D)
    lg = torch.einsum("hgd,thd->hgt", q.float().view(Hkv, G, D), kk) / math.sqrt(D)
    p = torch.softmax(lg, dim=-1).sum(1)                   # [Hkv, T]
    nb = math.ceil(T / block)
    pad = nb * block - T
    if pad:
        p = torch.cat([p, p.new_zeros(Hkv, pad)], dim=-1)
    return p.view(Hkv, nb, block).sum(-1)


def install(model, M, state):
    """Patch the decode attention call. state: arm, key_bits, kv_heads (frozenset of (layer, head)) or None, layers
    (frozenset) or None, probe (dict) or None, counters."""
    from oracle_run import _quantize_keys
    orig = M.sparse_flash_attention_forward

    def wrapped(query_states, key_states, value_states, attention_mask, query_length, softmax_scale=None,
                cache_seqlens=None, block_mask=None, block_size=None, **kw):
        if query_length == 1 and block_mask is not None:
            try:
                li = STASH["li"]
                state["n_decode_calls"] += 1
                T = int(cache_seqlens[0])
                nbv = math.ceil(T / block_size)
                q = query_states.reshape(query_states.shape[0], -1, query_states.shape[-2], query_states.shape[-1])[0, 0] \
                    if query_states.dim() == 4 else query_states[0]
                k = key_states[0, :T]
                Hkv = k.shape[1]
                kvh, lay = state.get("kv_heads"), state.get("layers")
                act = [h for h in range(Hkv) if (kvh is None or (li, h) in kvh) and (lay is None or li in lay)]
                pr = state.get("probe")
                need = (state["arm"] in ("exact", "lowbit") and act) or (pr is not None and pr.get("src_block") is not None)
                if need:
                    mass = group_mass(q, k, None, _quantize_keys, block_size) if (state["arm"] == "exact" or pr is not None) else None
                    low = group_mass(q, k, state["key_bits"], _quantize_keys, block_size) if state["arm"] == "lowbit" else None
                    if pr is not None and pr.get("src_block") is not None:
                        nb = pr["src_block"]
                        G = q.shape[0] // Hkv
                        for h in range(Hkv):
                            row = block_mask[0, h, :nbv]
                            if nb in (0, nbv - 1):
                                pr["acc"].add(li, h, True, 0, 0, 0, 0.0, (0, 0, 0)); continue
                            ex_row = free_override(block_mask[0, h], mass[h], nbv)
                            order = mass[h, :nbv].argsort(descending=True).tolist()
                            pr["acc"].add(li, h, False, int(bool(row[nb])), int(bool(ex_row[nb])), 0,
                                          float(mass[h, nb]) / G, (0, order.index(nb), 0))
                    if state["arm"] in ("exact", "lowbit") and act:
                        sc = mass if state["arm"] == "exact" else low
                        block_mask = block_mask.clone()
                        forced = torch.zeros(block_mask.shape[-1], dtype=torch.bool, device=block_mask.device)
                        forced[0] = True; forced[nbv - 1] = True
                        for h in act:
                            new = free_override(block_mask[0, h], sc[h], nbv)
                            state["blocks_gate"] += int((block_mask[0, h, :nbv] | forced[:nbv]).sum())   # blocks the kernel reads
                            state["blocks_new"] += int(new[:nbv].sum())
                            block_mask[0, h] = new
                        state["n_overrides"] += len(act)
                if state["arm"] == "full" and act:
                    block_mask = block_mask.clone()
                    forced = torch.zeros(block_mask.shape[-1], dtype=torch.bool, device=block_mask.device)
                    forced[0] = True; forced[nbv - 1] = True
                    for h in act:
                        state["blocks_gate"] += int((block_mask[0, h, :nbv] | forced[:nbv]).sum())   # what the gate read
                        block_mask[0, h, :nbv] = True
                        state["blocks_new"] += nbv
                    state["n_overrides"] += len(act)
            except Exception as e:                           # never silently
                state["errors"].append(f"{type(e).__name__}: {e}")
        return orig(query_states, key_states, value_states, attention_mask, query_length, softmax_scale=softmax_scale,
                    cache_seqlens=cache_seqlens, block_mask=block_mask, block_size=block_size, **kw)

    M.sparse_flash_attention_forward = wrapped
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        of = attn.forward

        def mk(of, li):
            def f(*a, **kw):
                STASH["li"] = li
                return of(*a, **kw)
            return f
        attn.forward = mk(of, li)


def local_gate_dir(gates_dir, base_dir):
    """The release's from_pretrained reads config.base_model; point it at the local base weights."""
    d = tempfile.mkdtemp(prefix="seerr_gates_")
    cfg = json.load(open(os.path.join(gates_dir, "config.json")))
    cfg["base_model"] = os.path.abspath(base_dir)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    os.symlink(os.path.abspath(os.path.join(gates_dir, "attn_gate_weights.pth")), os.path.join(d, "attn_gate_weights.pth"))
    return d


def compressed_key_cache(num_layers, block_size):
    """The release's KCompressionCache. Its __init__ calls transformers' Cache.__init__() without arguments, which
    transformers >= 4.54 refuses ("provide exactly one of layers or layer_class_to_replicate"); the class keeps its own
    per-layer dicts and uses nothing of the base class, so the instance is built without the base initializer and its
    fields are set exactly as the release's __init__ sets them."""
    from seer_attn.decode_sparse.cache_utils import KCompressionCache
    kc = KCompressionCache.__new__(KCompressionCache)
    kc.num_layers, kc.block_size = num_layers, block_size
    kc.k_compressed = {l: None for l in range(num_layers)}
    kc.k_remainder = {l: None for l in range(num_layers)}
    return kc


def greedy(model, ids, max_new, eos_ids, forced=None, on_step=None):
    """Dense prefill, then one token at a time through the decode path. forced: token ids to teacher-force (probe).
    Returns the generated ids (list). on_step(i) is called before decode step i (i = 0 is the first decode forward)."""
    from transformers.cache_utils import DynamicCache
    kv, kc = DynamicCache(), compressed_key_cache(model.num_layers, model.block_size)
    am = torch.ones_like(ids)
    out, cur = [], ids
    n = len(forced) if forced is not None else max_new
    hits = 0
    for i in range(n):
        if i > 0 and on_step is not None:
            on_step(i - 1)
        o = model(cur, attention_mask=am, past_key_values=kv, k_compressed_cache=kc, use_cache=True, logits_to_keep=1)
        kv, kc = o.past_key_values, o.k_compressed_cache
        nxt = int(o.logits[0, -1].argmax())
        if forced is not None:
            hits += int(nxt == forced[i]); nxt = forced[i]
        out.append(nxt)
        if forced is None and nxt in eos_ids:
            break
        cur = torch.tensor([[nxt]], device=ids.device)
        am = torch.cat([am, am.new_ones(1, 1)], dim=1)
    return out, hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="checkout of microsoft/SeerAttention")
    p.add_argument("--gates", required=True)
    p.add_argument("--base", default=None, help="local base model directory (Qwen3); --model_dir is an alias")
    p.add_argument("--model_dir", default=None, help="alias of --base, so pheno/manual/sb_chunk.sh can launch this runner")
    p.add_argument("--arm", required=True, choices=["dense", "gate", "exact", "lowbit", "full"])
    p.add_argument("--token_budget", type=int, default=4096)
    p.add_argument("--key_bits", type=int, default=None, choices=[1, 2, 3, 4, 8])
    p.add_argument("--kv_heads", default=None); p.add_argument("--layers", default=None)
    p.add_argument("--data", required=True); p.add_argument("--task", required=True)
    p.add_argument("--ruler_root", required=True); p.add_argument("--target_len", type=int, required=True)
    p.add_argument("--n", type=int, default=200); p.add_argument("--row_offset", type=int, default=0)
    p.add_argument("--probe", action="store_true", help="needle probe on a calibration draw (never seed=42)")
    p.add_argument("--steps", type=int, default=10); p.add_argument("--flag_tau", type=float, default=0.01)
    p.add_argument("--out", required=True); p.add_argument("--tag", default="")
    a = p.parse_args()
    a.base = a.base or a.model_dir
    if not a.base:
        raise SystemExit("--base (or --model_dir) is required -- ABORT")
    if (a.arm == "lowbit") != (a.key_bits is not None):
        raise SystemExit("--key_bits goes with --arm lowbit, and only with it -- ABORT")
    if a.probe and ("seed=42" in a.data or a.arm != "gate"):
        raise SystemExit("--probe runs the gate arm on a calibration draw (not seed=42) -- ABORT")
    sys.path.insert(0, a.repo)
    import oracle_run as OR
    from ruler_run import load_ruler_task, sha256_prefix
    from needle_probe import Acc, PromptLog, flag_heads, needle_token_span
    from transformers import AutoConfig, AutoTokenizer
    base_type = AutoConfig.from_pretrained(a.base).model_type
    if base_type == "qwen3":
        import seer_attn.decode_sparse.qwen3.modeling_qwen3_seerattn_inference as M
        from seer_attn import SeerDecodingQwen3ForCausalLM as SeerDecoding
    elif base_type == "qwen2":                              # DeepSeek-R1-Distill-Qwen (held-out check, 2026-10-01)
        import seer_attn.decode_sparse.qwen2.modeling_qwen2_seerattn_inference as M
        from seer_attn import SeerDecodingQwen2ForCausalLM as SeerDecoding
    else:
        raise SystemExit(f"base model_type {base_type}: the release's decode gates cover qwen2 and qwen3 -- ABORT")

    metric_fn, base_task, canonical_gen = load_ruler_task(a.ruler_root, a.task)
    rows = [json.loads(l) for l in open(a.data)]
    rows = rows[a.row_offset: a.row_offset + a.n]
    if len(rows) < a.n:
        raise SystemExit(f"data has {len(rows)} rows from offset {a.row_offset}, n={a.n} -- ABORT")
    tok = AutoTokenizer.from_pretrained(a.base)
    gd = local_gate_dir(a.gates, a.base)
    model = SeerDecoding.from_pretrained(
        gd, torch_dtype=torch.bfloat16, seerattn_sparsity_method="token_budget", seerattn_token_budget=a.token_budget,
        seerattn_implementation=("seer_dense" if a.arm == "dense" else "seer_sparse")).cuda().eval()
    cfg = model.config
    L, Hkv = cfg.num_hidden_layers, cfg.num_key_value_heads
    block = cfg.seerattn_gate_block_size
    state = {"arm": a.arm, "key_bits": a.key_bits, "kv_heads": OR._parse_kv_heads(a.kv_heads),
             "layers": (frozenset(int(x) for x in a.layers.split(",")) if a.layers else None),
             "probe": ({"acc": Acc(L, Hkv), "src_block": None} if a.probe else None),
             "n_decode_calls": 0, "n_overrides": 0, "blocks_gate": 0, "blocks_new": 0, "errors": []}
    if a.arm != "dense":
        install(model, M, state)
    plog = PromptLog(state["probe"]["acc"]) if a.probe else None   # per-prompt sums: any calibration subset offline
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    eos = {e for e in eos if e is not None and e != tok.unk_token_id}   # R1-Distill has no <|im_end|>
    preds, refs, gen_lens, ptoks = [], [], [], []
    tf_hits = tf_total = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        text = r["input"] + r.get("answer_prefix", "")
        enc = tok(text, return_tensors="pt", return_offsets_mapping=a.probe)
        ids = enc.input_ids.cuda()
        ptoks.append(int(ids.shape[1]))
        with torch.inference_mode():
            if a.probe:
                s, e = needle_token_span(text, r["outputs"][0], enc.offset_mapping[0].tolist())
                cont = ids[0, s:e][: a.steps].tolist()
                pr = state["probe"]

                def on_step(j, s=s):
                    pr["src_block"] = (s + j + 1) // block          # the step that predicts cont[j + 1]
                pr["src_block"] = None
                _, hits = greedy(model, ids, 0, eos, forced=cont, on_step=on_step)
                pr["src_block"] = None
                tf_hits += hits; tf_total += len(cont)
                plog.mark(pr["acc"])
            else:
                out, _ = greedy(model, ids, canonical_gen, eos)
                gen_lens.append(len(out))
                preds.append(tok.decode(out, skip_special_tokens=True)); refs.append(r["outputs"])
        if state["errors"]:
            raise SystemExit(f"SEERR HOOK ERRORS: {state['errors'][:3]} -- ABORT")
        if i % 10 == 0:
            print(f"  [{i}/{len(rows)}] ptok={ptoks[-1]} decode_calls={state['n_decode_calls']} overrides={state['n_overrides']} "
                  f"{(time.time() - t0) / (i + 1):.1f}s/sample", flush=True)
    if a.arm != "dense" and state["n_decode_calls"] == 0:
        raise SystemExit("SEERR: the decode hook never fired -- ABORT")
    if a.arm in ("exact", "lowbit") and (state["n_overrides"] == 0 or state["blocks_gate"] != state["blocks_new"]):
        raise SystemExit(f"SEERR: overrides={state['n_overrides']} blocks gate/new={state['blocks_gate']}/{state['blocks_new']} "
                         f"(must be > 0 and equal) -- ABORT")
    if a.arm == "full" and (state["n_overrides"] == 0 or state["blocks_new"] < state["blocks_gate"]):
        raise SystemExit(f"SEERR: overrides={state['n_overrides']} blocks gate/new={state['blocks_gate']}/{state['blocks_new']} "
                         f"(full attention must override and read at least the gate's blocks) -- ABORT")
    res = {"tag": a.tag, "family": "seerattention-r", "task": a.task, "arm": a.arm, "n": len(rows), "row_offset": a.row_offset,
           "token_budget": a.token_budget, "key_bits": a.key_bits, "kv_heads": a.kv_heads, "layers": a.layers,
           "target_len": a.target_len, "max_new": canonical_gen, "n_decode_calls": state["n_decode_calls"],
           "n_overrides": state["n_overrides"], "blocks_gate": state["blocks_gate"], "blocks_new": state["blocks_new"],
           "errors": [], "n_errors": 0, "mean_prompt_tokens": statistics.fmean(ptoks),
           "provenance": {"argv": sys.argv, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                          "data_sha256_12": sha256_prefix(a.data), "data": os.path.abspath(a.data),
                          "model_dir": os.path.abspath(a.base), "gates": os.path.abspath(a.gates),
                          "elapsed_s": time.time() - t0}}
    if a.probe:
        S = state["probe"]["acc"].summary()
        flagged, sc = flag_heads(S["per_layer_head"], a.flag_tau)
        res.update({**S, "teacher_forced_acc": tf_hits / max(tf_total, 1), "steps": a.steps, "flag_tau": a.flag_tau,
                    "flagged_kv_heads": ",".join(f"{l}.{h}" for l, h in flagged), "per_prompt": plog.rows,
                    "flag_scores": {f"{l}.{h}": v for (l, h), v in sc.items()}})
        print(f"FLAGGED_KV_HEADS tau={a.flag_tau}: {res['flagged_kv_heads']}  tf_acc={res['teacher_forced_acc']:.3f}", flush=True)
    else:
        per = [metric_fn([pd], [rf]) for pd, rf in zip(preds, refs)]
        res.update({"string_match_all": statistics.fmean(per), "scores": per, "predictions": preds, "references": refs,
                    "all_correct": sum(1 for x in per if x >= 100.0) / len(per), "mean_gen_len": statistics.fmean(gen_lens),
                    "trunc_frac": sum(1 for g in gen_lens if g >= canonical_gen) / len(gen_lens)})
        print(f"SEERR_CELL {a.arm} {a.task} n={len(rows)} score={res['string_match_all']:.2f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
