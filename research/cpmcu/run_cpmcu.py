#!/usr/bin/env python
"""One arm of InfLLM-V2 / MiniCPM4 inside OpenBMB's CPM.cu engine.

Arms (one per process: CPM.cu keeps a single global model):
  dense    CPM.cu's dense path (apply_sparse=False), the same weights with every block read
  shipped  CPM.cu's InfLLM-V2 path at the release's sparse_config: init_blocks 1 -> sink_window_size 1,
           window 2048 / block 64 -> block_window_size 32, topk 64, dense_len 8192 -> sparse_switch 8192, and the
           release's two-level stage 1 (keys pooled 32/16 and 128/64) -> use_compress_lse True
  fix      shipped plus full attention in the flagged KV heads (env CPMCU_DENSE_KV_HEADS, patch
           research/cpmcu/dense_flagged_heads.patch): every bit of their rows in the block mask
  fixk     the same fix routed to the dense kernel: the shipped mask is kept and the flagged heads' outputs are
           overwritten by a dense flash call over the whole cache on those heads only (CPMCU_DENSE_KV_MODE=kernel)

Modes:
  eval  greedy generation exactly as pheno/oracle_run.py does it: tokenizer(input + answer_prefix) with BOS, up to
        RULER's tokens_to_generate (128 for the niah tasks), stop at the generation_config EOS ids, decode with
        skip_special_tokens, RULER string_match_all per prompt
  time  TTFT (CPM.cu's synchronized prefill time) and time per output token (synchronized decode time over a fixed
        number of decode steps, terminators disabled so every arm decodes the same count), after warm-up prompts
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time


def string_match_all(pred, refs):
    # RULER's metric for the niah tasks, per prompt: share of references contained in the prediction, x100
    return 100.0 * sum(1.0 if r.lower() in pred.lower() else 0.0 for r in refs) / len(refs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--arm", choices=["dense", "shipped", "fix", "fixk"], required=True)
    p.add_argument("--dense_heads", default="9.0,16.0,22.0")
    p.add_argument("--data", required=True)
    p.add_argument("--mode", choices=["eval", "time"], required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--time_decode", type=int, default=128)
    p.add_argument("--chunk_length", type=int, default=2048)
    p.add_argument("--memory_limit", type=float, default=0.9)
    p.add_argument("--cuda_graph", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--block_window", type=int, default=32)
    p.add_argument("--sink_window", type=int, default=1)
    p.add_argument("--sparse_switch", type=int, default=8192)
    p.add_argument("--use_compress_lse", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if a.arm in ("fix", "fixk"):
        os.environ["CPMCU_DENSE_KV_HEADS"] = a.dense_heads      # read by the patched engine at model construction
        # fix: every bit of the flagged heads' rows in the shipped block mask (the shipped sparse kernel reads them)
        # fixk: the shipped mask unchanged, then the dense kernel over the whole cache for the flagged heads only,
        #       written over their query heads' outputs
        os.environ["CPMCU_DENSE_KV_MODE"] = "kernel" if a.arm == "fixk" else "mask"
    else:
        os.environ.pop("CPMCU_DENSE_KV_HEADS", None)
        os.environ.pop("CPMCU_DENSE_KV_MODE", None)

    import torch
    from transformers import AutoTokenizer
    from cpmcu.llm import LLM

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[a.dtype]
    llm = LLM(a.model_dir, memory_limit=a.memory_limit, chunk_length=a.chunk_length, dtype=dtype,
              cuda_graph=bool(a.cuda_graph), apply_sparse=(a.arm != "dense"),
              sink_window_size=a.sink_window, block_window_size=a.block_window, sparse_topk_k=a.topk,
              sparse_switch=a.sparse_switch, use_compress_lse=bool(a.use_compress_lse), temperature=0.0)
    llm.init_storage()
    llm.load_from_hf()
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    gen_cfg = json.load(open(os.path.join(a.model_dir, "generation_config.json")))
    eos = gen_cfg["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]

    rows = [json.loads(l) for l in open(a.data)]
    res = {"arm": a.arm, "mode": a.mode, "model_dir": a.model_dir, "data": a.data,
           "dense_heads": a.dense_heads if a.arm in ("fix", "fixk") else None,
           "dense_mode": {"fix": "mask", "fixk": "kernel"}.get(a.arm),
           "settings": {k: getattr(a, k) for k in ("chunk_length", "memory_limit", "cuda_graph", "dtype", "topk",
                                                   "block_window", "sink_window", "sparse_switch",
                                                   "use_compress_lse", "max_new", "time_decode", "warmup")},
           "gpu": torch.cuda.get_device_name(0), "host": platform.node(),
           "max_total_length": llm.max_total_length, "eos": eos}
    try:
        res["cpmcu_commit"] = subprocess.check_output(
            ["git", "-C", os.environ.get("CPMCU_REPO", "."), "rev-parse", "HEAD"]).decode().strip()
    except Exception as e:  # noqa: BLE001
        res["cpmcu_commit"] = f"unknown ({e})"

    def ids_of(r):
        enc = tok(r["input"] + r.get("answer_prefix", ""), return_tensors="pt")
        return enc["input_ids"][0].to(torch.int32).cuda()

    if a.mode == "eval":
        scores, preds, refs, lens, gens = [], [], [], [], []
        t0 = time.time()
        for r in rows[:a.n]:
            ids = ids_of(r)
            toks, dec_t, pre_t = llm.generate(ids, generation_length=a.max_new, terminators=eos)
            pred = tok.decode(toks, skip_special_tokens=True)
            scores.append(string_match_all(pred, r["outputs"]))
            preds.append(pred); refs.append(r["outputs"]); lens.append(int(ids.numel())); gens.append(len(toks))
            print(f"[{len(scores)}/{a.n}] len={ids.numel()} score={scores[-1]:.0f} pred={pred[:80]!r}", flush=True)
        res.update({"n": len(scores), "string_match_all": sum(scores) / len(scores), "scores": scores,
                    "predictions": preds, "references": refs, "prompt_tokens": lens, "gen_tokens": gens,
                    "wall_s": time.time() - t0})
    else:
        per = []
        k = a.warmup + a.n
        for i, r in enumerate(rows[:k]):
            ids = ids_of(r)
            toks, dec_t, pre_t = llm.generate(ids, generation_length=a.time_decode, terminators=[])
            rec = {"prompt_tokens": int(ids.numel()), "ttft_s": pre_t, "decode_s": dec_t,
                   "decode_steps": len(toks) - 1, "tpot_ms": 1000.0 * dec_t / max(len(toks) - 1, 1),
                   "warmup": i < a.warmup}
            per.append(rec)
            print(json.dumps(rec), flush=True)
        timed = [x for x in per if not x["warmup"]]
        med = lambda xs: sorted(xs)[len(xs) // 2] if len(xs) % 2 else 0.5 * sum(sorted(xs)[len(xs) // 2 - 1:len(xs) // 2 + 1])
        res.update({"n": len(timed), "per_prompt": per,
                    "ttft_s_median": med([x["ttft_s"] for x in timed]),
                    "tpot_ms_median": med([x["tpot_ms"] for x in timed]),
                    "ttft_s_mean": sum(x["ttft_s"] for x in timed) / len(timed),
                    "tpot_ms_mean": sum(x["tpot_ms"] for x in timed) / len(timed)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("WROTE", a.out, {k: res.get(k) for k in ("arm", "mode", "n", "string_match_all", "ttft_s_median", "tpot_ms_median")}, flush=True)


if __name__ == "__main__":
    sys.exit(main())
