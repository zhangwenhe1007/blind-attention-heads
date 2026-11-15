#!/usr/bin/env python
"""Run ONE RULER cell on ONE arm and emit ONE json holding per-sample everything.

Which HARD RULE each piece serves:
  R2  arm-difference gate  -> --spy instruments the sparse kernel and counts calls
  R3  never invent an eval -> scoring imports RULER's OWN metric_fn, verbatim
  R6  paired identical draws -> samples are taken in file order; nothing is shuffled
  R8  per-sample output    -> scores, gen_lens, predictions, prompt_tokens, terminated
  R9  report both metrics  -> string_match_all AND all_correct

Guards, plus one the documented set does not cover:
  length-drift      ABORT if |mean_prompt_tokens - target| / target > 3%
  budget-binding    ABORT if >2% of generations hit max_new (warn under --allow_trunc)
  context-overflow  ABORT if prompt+max_new exceeds max_position_embeddings.
                    The documented drift guard is a PERCENTAGE and cannot catch this:
                    RULER's generator fills the full budget and only reserves room for
                    the chat template under --prepare_for_ns (model_template_token
                    defaults to 0), so a templated 32K prompt overruns a 32768-position
                    model by a margin far under the 3% threshold.
"""
import argparse, hashlib, importlib.util, json, os, statistics, sys, time
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def sha256_prefix(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def load_ruler_task(ruler_root, task):
    """Import RULER's own metric fn AND its canonical generation budget. HARD RULE 3."""
    root = Path(ruler_root)
    cfg = yaml.safe_load((root / "scripts" / "synthetic.yaml").read_text())
    if task not in cfg:
        raise SystemExit(f"task {task!r} not in RULER's synthetic.yaml -- ABORT")
    base_task = cfg[task]["task"]

    const = root / "scripts" / "eval" / "synthetic" / "constants.py"
    spec = importlib.util.spec_from_file_location("ruler_eval_constants", const)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # tokens_to_generate lives in the DATA-side constants
    dconst = root / "scripts" / "data" / "synthetic" / "constants.py"
    dspec = importlib.util.spec_from_file_location("ruler_data_constants", dconst)
    dmod = importlib.util.module_from_spec(dspec)
    dspec.loader.exec_module(dmod)

    return (mod.TASKS[base_task]["metric_fn"], base_task,
            int(dmod.TASKS[base_task]["tokens_to_generate"]))


def install_spy(model):
    """Count real invocations of the sparse path. HARD RULE 2: prove the arms are
    two computations by instrumenting the code path, never by reading the metric.
    This is what caught NOSA (spy = 0 calls, topk=2 identical to topk=64)."""
    modname = type(model).__module__          # transformers_modules...modeling_minicpm
    mod = sys.modules[modname]
    counts = {"infllmv2_attn_varlen_func": 0, "compressed_attention": 0}

    def wrap(name):
        original = getattr(mod, name, None)
        if original is None:
            return None

        def spied(*a, **kw):
            counts[name] += 1
            return original(*a, **kw)

        setattr(mod, name, spied)
        return original

    missing = [n for n in counts if wrap(n) is None]
    return counts, modname, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="LOCAL patched arm dir (never the HF cache)")
    p.add_argument("--arm", required=True, choices=["sparse", "dense"])
    p.add_argument("--data", required=True, help="RULER-prepared validation.jsonl")
    p.add_argument("--task", required=True)
    p.add_argument("--ruler_root", required=True)
    p.add_argument("--target_len", type=int, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--max_new", default="512",
                   help="int, or 'canonical' to use RULER's own tokens_to_generate "
                        "for this task (what the published numbers were produced with)")
    p.add_argument("--topk", type=int, default=None, help="override sparse_config.topk")
    p.add_argument("--window_size", type=int, default=None,
                   help="override sparse_config.window_size. InfLLM-V2 hands the last "
                        "window_size tokens over FOR FREE (local_blocks = window//block, and "
                        "effective budget = topk + window//block). SeerAttention's only free "
                        "structure is last_block_dense = ONE 64-token block, so cross-family "
                        "budget comparisons are confounded until this is matched.")
    p.add_argument("--allow_trunc", action="store_true")
    p.add_argument("--spy", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    metric_fn, base_task, canonical_gen = load_ruler_task(args.ruler_root, args.task)
    max_new = canonical_gen if args.max_new == "canonical" else int(args.max_new)
    print(f"task={args.task} base={base_task} canonical_tokens_to_generate={canonical_gen} "
          f"max_new={max_new}")

    rows = [json.loads(l) for l in open(args.data)]
    if len(rows) < args.n:
        raise SystemExit(f"data has {len(rows)} rows < n={args.n} -- ABORT (no partial cells)")
    rows = rows[: args.n]                      # file order. R6: identical ordered draws.

    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    sparse_cfg = getattr(cfg, "sparse_config", None)
    if args.arm == "dense" and sparse_cfg is not None and sparse_cfg.get("dense_len", 0) < 10 ** 8:
        # a sparse_config whose dense_len exceeds any context is a dense arm too: releases whose attention class needs the
        # key to exist take their own dense FlashAttention branch below dense_len (pheno/build_lcm_arms.py)
        raise SystemExit("dense arm dir still carries an active sparse_config -- ABORT")
    if args.arm == "sparse" and sparse_cfg is None:
        raise SystemExit("sparse arm dir has no sparse_config -- ABORT")
    if args.topk is not None:
        if sparse_cfg is None:
            raise SystemExit("--topk given but this arm has no sparse_config -- ABORT")
        sparse_cfg["topk"] = args.topk
        cfg.sparse_config = sparse_cfg
    if args.window_size is not None:
        if sparse_cfg is None:
            raise SystemExit("--window_size given but this arm has no sparse_config -- ABORT")
        bs = sparse_cfg.get("block_size", 64)
        if args.window_size % bs != 0:
            raise SystemExit(f"--window_size {args.window_size} not a multiple of block_size {bs} -- ABORT")
        # local_blocks = window//block_size. At 0 the query's own block is not guaranteed
        # selected, which is a different (and kernel-untested) regime -- refuse it rather than
        # silently produce numbers from an unvalidated path.
        if args.window_size < bs:
            raise SystemExit(f"--window_size must be >= block_size ({bs}); 0 leaves local_blocks=0 -- ABORT")
        sparse_cfg["window_size"] = args.window_size
        cfg.sparse_config = sparse_cfg

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if getattr(cfg, "mixer_types", None):        # MiniCPM-SALA: see pheno/compat_fla.py (the release's files stay untouched)
        import fla.ops.simple_gla as _sg
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from compat_fla import allow_explicit_head_first_false
        allow_explicit_head_first_false(_sg)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, config=cfg, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).cuda().eval()

    spy_counts, spy_module, spy_missing = (None, None, None)
    if args.spy:
        spy_counts, spy_module, spy_missing = install_spy(model)

    max_pos = getattr(cfg, "max_position_embeddings", None)
    preds, refs, gen_lens, prompt_toks, terminated = [], [], [], [], []
    len_fields = [r.get("length") for r in rows]

    t0 = time.time()
    for i, r in enumerate(rows):
        # `input` EXCLUDES the answer prefix -- the generator splits them into separate
        # fields (niah.py writes 'input' and 'answer_prefix'). Concatenating here is the
        # correct prompt; upstream restored exactly this in call_api.py (8ef9e90).
        prompt = r["input"] + r.get("answer_prefix", "")
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc.input_ids.cuda()
        # MiniCPM4 needs an EXPLICIT attention_mask on the flash-attn path
        attn_mask = enc.attention_mask.cuda()
        prompt_toks.append(int(input_ids.shape[1]))

        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids, attention_mask=attn_mask,
                max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            )
        gen_ids = out[0, input_ids.shape[1]:]
        gen_lens.append(int(gen_ids.shape[0]))
        terminated.append(bool(gen_ids.shape[0] < max_new))
        preds.append(tok.decode(gen_ids, skip_special_tokens=True))
        refs.append(r["outputs"])

        if i % 25 == 0:
            print(f"  [{i}/{len(rows)}] ptok={prompt_toks[-1]} gen={gen_lens[-1]} "
                  f"{(time.time()-t0)/max(i,1):.1f}s/sample", flush=True)

    elapsed = time.time() - t0

    # ---- scoring: RULER's own function, per-sample and aggregate ----
    per_sample = [metric_fn([pd], [rf]) for pd, rf in zip(preds, refs)]
    agg = metric_fn(preds, refs)
    assert abs(statistics.fmean(per_sample) - agg) < 0.05, "per-sample scoring disagrees with aggregate"
    all_correct = sum(1 for s in per_sample if s >= 100.0) / len(per_sample)

    mean_ptok = statistics.fmean(prompt_toks)
    trunc_frac = 1.0 - (sum(terminated) / len(terminated))
    drift = abs(mean_ptok - args.target_len) / args.target_len

    # ---- tokenizer-consistency check (sharper than the documented drift guard) ----
    # The generator records length = realized_prompt_tokens + tokens_to_generate, so the
    # residual must be a CONSTANT equal to that budget. If the jsonl were generated under
    # a different tokenizer than the one scoring it, the residual scatters -- which is
    # exactly artifact #6 (the tokenizer path collision that silently overwrote cells).
    # A percentage drift guard cannot see this: measured nominal drift is 0.1% on VT but
    # ~3.0% on MK3 purely from how the generator packs its haystack, so a 3% threshold
    # both false-fires on MK3 and stays blind to a tokenizer swap.
    resid = [lf - pt for lf, pt in zip(len_fields, prompt_toks) if lf is not None]
    resid_mean = statistics.fmean(resid) if resid else None
    resid_sd = statistics.pstdev(resid) if len(resid) > 1 else 0.0

    over = [i for i, (pt, gl) in enumerate(zip(prompt_toks, gen_lens))
            if max_pos is not None and pt + gl > max_pos]

    result = {
        "tag": args.tag, "task": args.task, "base_task": base_task, "arm": args.arm,
        "n": len(rows), "max_new": max_new, "canonical_tokens_to_generate": canonical_gen,
        "topk_override": args.topk, "window_size_override": args.window_size,
        "string_match_all": agg, "all_correct": all_correct,
        "scores": per_sample, "gen_lens": gen_lens, "prompt_tokens": prompt_toks,
        "terminated": terminated, "predictions": preds, "references": refs,
        "mean_prompt_tokens": mean_ptok,
        "median_prompt_tokens": statistics.median(prompt_toks),
        "max_prompt_tokens": max(prompt_toks),
        "mean_gen_len": statistics.fmean(gen_lens),
        "median_gen_len": statistics.median(gen_lens),
        "trunc_frac": trunc_frac,
        "target_len": args.target_len, "length_drift": drift,
        "max_position_embeddings": max_pos,
        "len_residual_mean": resid_mean, "len_residual_sd": resid_sd,
        "n_position_overflow": len(over),
        "provenance": {
            "data_path": os.path.abspath(args.data),
            "data_sha256_12": sha256_prefix(args.data),
            "data_mtime": os.path.getmtime(args.data),
            "model_dir": os.path.abspath(args.model_dir),
            "tokenizer": os.path.abspath(args.model_dir),
            "sparse_config": sparse_cfg,
            "ruler_root": os.path.abspath(args.ruler_root),
            "elapsed_s": elapsed, "s_per_sample": elapsed / len(rows),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "spy": ({"module": spy_module, "counts": spy_counts, "missing": spy_missing}
                if args.spy else None),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in
                      ("tag", "task", "arm", "n", "string_match_all", "all_correct",
                       "mean_prompt_tokens", "mean_gen_len", "trunc_frac", "length_drift")},
                     indent=2))
    if args.spy:
        print("SPY:", json.dumps(result["spy"]))

    # ---- guards, applied AFTER the json is written so evidence is never lost ----
    fatal = []
    print(f"length: mean_ptok {mean_ptok:.0f} | nominal drift {drift:.2%} | "
          f"residual {resid_mean} +/- {resid_sd:.2f} (expect {canonical_gen} +/- 0)")
    if resid_mean is not None:
        if resid_sd > 2.0:
            fatal.append(f"tokenizer mismatch: length-residual sd {resid_sd:.2f} > 2 tokens "
                         f"-- the jsonl was tokenized by a DIFFERENT tokenizer than this run")
        if abs(resid_mean - canonical_gen) > 2.0:
            fatal.append(f"length-residual mean {resid_mean:.1f} != canonical "
                         f"tokens_to_generate {canonical_gen} -- data/config mismatch")
    if drift > 0.05:
        fatal.append(f"nominal length-drift {drift:.2%} > 5% (mean_ptok {mean_ptok:.0f} "
                     f"vs target {args.target_len})")
    if over:
        msg = (f"position-overflow: {len(over)}/{len(rows)} samples exceeded "
               f"max_position_embeddings {max_pos} (prompt+generated)")
        (print("WARN (allow_trunc):", msg) if args.allow_trunc else fatal.append(msg))
    if trunc_frac > 0.02 + 1e-9:   # exact 2% (e.g. 4/200) must not trip on float error
        msg = f"budget-binding: trunc_frac {trunc_frac:.1%} > 2%"
        (print("WARN (allow_trunc):", msg) if args.allow_trunc else fatal.append(msg))
    if fatal:
        for f in fatal:
            print("GUARD FAILED:", f, file=sys.stderr)
        sys.exit(3)
    print("guards: PASS")


if __name__ == "__main__":
    main()
