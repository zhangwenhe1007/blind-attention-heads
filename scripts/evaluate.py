#!/usr/bin/env python
"""Score RULER prompts on an InfLLM-V2 checkpoint with the shipped selector, a fix, the diagnostic, or full attention.

    python scripts/evaluate.py --model MODEL --prompts validation.jsonl --mode shipped
    python scripts/evaluate.py --model MODEL --prompts validation.jsonl --mode dense
    python scripts/evaluate.py --model MODEL --prompts validation.jsonl --mode full-attention-heads --heads infllm-v2-8b
    python scripts/evaluate.py --model MODEL --prompts validation.jsonl --mode scorer --tokens every --heads infllm-v2-8b
    python scripts/evaluate.py --model MODEL --prompts validation.jsonl --mode larger-budget --heads infllm-v2-8b
    python scripts/evaluate.py --model MODEL --prompts validation.jsonl --mode exact-mass --tokens question

Modes
    shipped               the release's selector
    dense                 full attention in every head, same weights
    full-attention-heads  the default fix: full attention in the flagged heads
    scorer                the 4-bit scorer in the flagged heads at the shipped budget (--tokens decode, question or every)
    larger-budget         --factor times the free slots in the layers of the flagged heads
    exact-mass            the diagnostic: selection by exact attention mass, in --heads or in every head (--tokens as above)

--heads takes a shipped head set (a name from heads/), a head file written by scripts/calibrate.py, or units such as
9.0,16.0,22.0.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blindheads import infllm_v2, load_heads, parse_heads, ruler  # noqa: E402


def resolve_heads(spec):
    try:
        return load_heads(spec).heads
    except FileNotFoundError:
        return [f"{layer}.{head}" for layer, head in parse_heads(spec)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint folder or hub id, for example openbmb/InfLLM-V2-Long-Sparse-Base")
    ap.add_argument("--prompts", required=True, help="RULER prompt file (scripts/make_ruler_prompts.sh)")
    ap.add_argument("--mode", required=True, choices=["shipped", "dense", "full-attention-heads", "scorer", "larger-budget", "exact-mass"])
    ap.add_argument("--heads", default=None, help="the flagged heads, for the modes that act in them")
    ap.add_argument("--tokens", default="decode", choices=["decode", "question", "every"], help="where scorer and exact-mass act")
    ap.add_argument("--bits", type=int, default=4, help="bits per key channel of the scorer's copy")
    ap.add_argument("--factor", type=int, default=4, help="larger-budget: multiple of the free slots")
    ap.add_argument("--n", type=int, default=None, help="prompts to run (default: all)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=ruler.TOKENS_TO_GENERATE["niah"])
    ap.add_argument("--out", default=None, help="write the per-prompt outputs and scores to this JSON file")
    a = ap.parse_args()
    needs_heads = a.mode in ("full-attention-heads", "scorer", "larger-budget")
    if needs_heads and a.heads is None:
        ap.error(f"--mode {a.mode} needs --heads")
    if a.heads is not None and a.mode in ("shipped", "dense"):
        ap.error(f"--mode {a.mode} takes no --heads")
    heads = resolve_heads(a.heads) if a.heads else None
    rows = ruler.read_prompts(a.prompts, a.n, a.offset)
    model, tok, cfg = infllm_v2.load(a.model, dense=a.mode == "dense")
    t0 = time.time()
    handle = None
    if a.mode == "full-attention-heads":
        handle = infllm_v2.full_attention_heads(model, heads)
    elif a.mode == "scorer":
        handle = infllm_v2.low_bit_scorer(model, cfg, heads, a.tokens, a.bits)
    elif a.mode == "exact-mass":
        handle = infllm_v2.exact_mass(model, cfg, heads, a.tokens)
    elif a.mode == "larger-budget":
        handle = infllm_v2.larger_budget(model, cfg, sorted({int(u.split(".")[0]) for u in heads}), a.factor)
    res = infllm_v2.evaluate(model, tok, rows, a.max_new_tokens, log=print)
    if handle is not None:
        st = handle.stats
        if st.get("errors") or st.get("forced_blocks_missing"):
            raise SystemExit(f"the hook failed: {st}")
        if not any(st.get(k) for k in ("decode_calls", "calls")):
            raise SystemExit(f"the hook never ran: {st}")
        res["hook"] = {k: v for k, v in st.items() if k != "errors"}
        handle.remove()
    res.update(mode=a.mode, heads=heads, tokens=(a.tokens if a.mode in ("scorer", "exact-mass") else None), model=a.model, prompts=os.path.abspath(a.prompts),
               n=len(rows), offset=a.offset, seconds=round(time.time() - t0, 1))
    print(f"{a.mode}: accuracy {res['accuracy']:.1f} on {len(rows)} prompts")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
