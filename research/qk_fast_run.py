"""Run a corrected-protocol RULER cell with the DEPLOYABLE low-bit key scorer (pheno/qk_fast.py) instead of the
emulation inside oracle_run.py (--oracle_query meanq --key_bits 2). Everything else -- data, generation, scoring,
provenance -- is oracle_run.main(), so the cell is comparable to every other cell row for row.

Usage: python pheno/qk_fast_run.py --qk_mode {torch,triton} [--qk_kv_heads L.H,.. --qk_prefill_tail K
       --qk_criterion {meanq,heads} --qk_bits {2,4}] <oracle_run.py args with --no_oracle --no_install [--layers a-b]>

--layers is the band in which the scorer replaces the shipped stage-1 selector at decode (recorded by the harness in
the JSON's "layers" field, which merge_chunks.py checks); without it the scorer runs in every layer.
The harness writes to a temporary file; the final JSON is written only after the hook counters have been checked, with
arm = "qk-fast", so a file that says "shipped" can never sit under this cell's name.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pop_opt(argv, flag, default=None):
    """Remove `flag VALUE` from argv and return (VALUE or default, argv)."""
    if flag not in argv:
        return default, argv
    k = argv.index(flag)
    return argv[k + 1], argv[:k] + argv[k + 2:]


def split_args(argv):
    """-> (mode, harness argv with --out redirected to a temp file, final out path)."""
    argv = list(argv)
    if "--qk_mode" not in argv:
        raise SystemExit("--qk_mode torch|triton required")
    k = argv.index("--qk_mode")
    mode = argv[k + 1]
    if mode not in ("torch", "triton"):
        raise SystemExit(f"--qk_mode {mode}: torch or triton")
    argv = argv[:k] + argv[k + 2:]
    for need in ("--no_oracle", "--no_install", "--out"):
        if need not in argv:
            raise SystemExit(f"{need} required: the deployable scorer installs its own hooks")
    for bad in ("--learned_indexer", "--quant_indexer", "--key_bits", "--layer_budget", "--topk", "--keep_only",
                "--random_sel", "--legacy_rows"):
        if bad in argv:
            raise SystemExit(f"{bad} has no meaning with the deployable scorer -- ABORT")
    o = argv.index("--out")
    out = argv[o + 1]
    argv[o + 1] = out + ".harness_tmp"
    return mode, argv, out


def finalize(tmp, out, mode, layers, st, stale, extra=None):
    """Check the hook counters and write the final JSON. Raises SystemExit on any failure (the temp file is removed,
    nothing is left under the cell's name)."""
    d = json.load(open(tmp))
    os.remove(tmp)
    if d.get("arm") != "shipped" or d.get("n_oracle_calls"):
        raise SystemExit(f"harness arm was {d.get('arm')} with {d.get('n_oracle_calls')} oracle calls: the harness hooks "
                         f"must stay uninstalled -- ABORT")
    if st["errors"]:
        raise SystemExit(f"QK_FAST HOOK ERRORS {len(st['errors'])}: {st['errors'][:3]} -- ABORT")
    if stale:
        raise SystemExit("QK_FAST used a side cache that belonged to another prompt -- ABORT")
    if st["n_calls"] == 0:
        raise SystemExit("QK_FAST never fired -- ABORT")
    d["arm"] = "qk-fast"
    d["arm_legacy_label"] = "qk-fast"
    extra = extra or {}
    # the label carries the whole configuration, so merge_chunks.py cannot mix chunks of different configurations
    d["learned_indexer"] = f"qk_fast:{int(extra.get('bits') or 2)}bit:{mode}" \
        + (":crit=heads" if extra.get("criterion") == "heads" else "") \
        + (f":heads={extra['kv_heads']}" if extra.get("kv_heads") else "") \
        + (f":tail={extra['prefill_tail']}" if extra.get("prefill_tail") else "") \
        + (f":qp={extra['stats_prefix']}" if extra.get("stats_prefix") else "") \
        + (":half" if extra.get("half_prefill") else "") + (":fused" if extra.get("fused_prefill") else "")
    d["qk_fast"] = {"wrapper": "pheno/qk_fast_run.py", "mode": mode, "layers": sorted(layers) if isinstance(layers, frozenset) else (list(layers) if layers else None),
                    "n_calls": st["n_calls"], "n_passthrough": st["n_passthrough"], "n_errors": 0, "stale": False, **extra}
    with open(out, "w") as f:
        json.dump(d, f, indent=2)
    return d


def main():
    raw = sys.argv[1:]
    qk_heads, raw = pop_opt(raw, "--qk_kv_heads")            # 'L.H,L.H': scorer on these KV heads only (replaces --layers)
    qk_tail, raw = pop_opt(raw, "--qk_prefill_tail", "0")    # K: the last K prefill rows also come from the scorer
    qk_tail = int(qk_tail)
    qk_crit, raw = pop_opt(raw, "--qk_criterion", "meanq")   # meanq: group-mean query; heads: each query head's own softmax, summed
    qk_bits, raw = pop_opt(raw, "--qk_bits", "2")            # 2 or 4 bit keys in the side cache
    qk_bits = int(qk_bits)
    qk_prefix, raw = pop_opt(raw, "--qk_stats_prefix")       # N: causal quantizer statistics from the first N keys
    qk_prefix = int(qk_prefix) if qk_prefix is not None else None
    qk_half = "--qk_half_prefill" in raw                     # float16 scoring of the rewritten prefill rows
    qk_fused = "--qk_fused_prefill" in raw                   # with --qk_half_prefill: fused Triton mask + block logsumexp
    raw = [x for x in raw if x not in ("--qk_half_prefill", "--qk_fused_prefill")]
    if qk_crit not in ("meanq", "heads") or qk_bits not in (2, 4):
        raise SystemExit(f"--qk_criterion {qk_crit} / --qk_bits {qk_bits}: meanq|heads and 2|4 -- ABORT")
    if qk_heads is not None and "--layers" in raw:
        raise SystemExit("--qk_kv_heads replaces --layers -- ABORT")
    mode, argv, out = split_args(raw)
    import oracle_run as H
    import qk_fast as Q
    box = {}
    orig_build = H.build_model_and_state

    def build(a):
        model, tok, cfg, state = orig_build(a)
        layers = H._parse_layer_set(a.layers)               # (lo, hi) inclusive, a frozenset, or None for every layer
        box["layers"] = layers
        kvh = H._parse_kv_heads(qk_heads)
        box["st"] = Q.install(model, cfg, layers=(None if kvh is not None else layers), mode=mode, kv_heads=kvh,
                              prefill_tail=qk_tail, criterion=qk_crit, bits=qk_bits, stats_prefix=qk_prefix,
                              half_prefill=qk_half, fused_prefill=qk_fused)
        print(f"qk_fast armed: mode={mode} layers={layers} kv_heads={qk_heads} prefill_tail={qk_tail} "
              f"criterion={qk_crit} bits={qk_bits}", flush=True)
        return model, tok, cfg, state
    H.build_model_and_state = build
    sys.argv = [H.__file__] + argv
    try:
        H.main()
    except SystemExit:
        if os.path.exists(out + ".harness_tmp"):
            os.remove(out + ".harness_tmp")
        raise
    st = box["st"]
    if qk_tail and not st.get("n_tail_rows"):
        if os.path.exists(out + ".harness_tmp"):
            os.remove(out + ".harness_tmp")
        raise SystemExit("--qk_prefill_tail requested but no prefill row was rewritten -- ABORT")
    d = finalize(out + ".harness_tmp", out, mode, box.get("layers"), st, Q.stale_uses(st),
                 extra={"kv_heads": qk_heads, "prefill_tail": qk_tail, "n_tail_rows": st.get("n_tail_rows"),
                        "criterion": qk_crit, "bits": qk_bits, "stats_prefix": qk_prefix, "half_prefill": qk_half, "fused_prefill": qk_fused})
    print("QK_FAST", json.dumps({"score": d["string_match_all"], **d["qk_fast"]}))


if __name__ == "__main__":
    main()
