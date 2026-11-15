# Research scripts

The `blindheads` package covers the InfLLM-V2 release. These scripts cover three things it does not cover yet. They are
the scripts the paper's runs used, kept as they were: they carry arms the paper does not use, and their comments are a
lab notebook. Paths written as `$SPARSE_ROOT` stood for a scratch folder on the cluster the runs used.

## SeerAttention-R on Qwen3

`seerr_run.py` runs the SeerAttention-R decode gates with a fix in the flagged heads, and `fixgain_path_seerr.py` is the
calibration probe for them. Both need a clone of [microsoft/SeerAttention](https://github.com/microsoft/SeerAttention),
the gate weights and the base model (`THIRD_PARTY.md`).

```bash
C="--repo SeerAttention --gates SeerAttention-Decode-Qwen3-8B-AttnGates --base Qwen3-8B --task niah_multikey_3 \
   --ruler_root RULER --target_len 32768 --token_budget 4096 --data PROMPTS --n 200"
H=$(python -c "from blindheads import load_heads; print(','.join(load_heads('qwen3-8b_seerattention-r').heads))")
python research/seerr_run.py $C --arm gate --out shipped.json                       # the shipped gates
python research/seerr_run.py $C --arm full --kv_heads $H --out fix.json             # full attention in the flagged heads
python research/seerr_run.py $C --arm lowbit --key_bits 4 --kv_heads $H --out scorer.json   # the 4-bit scorer
python research/seerr_run.py $C --arm dense --out dense.json                        # the dense model
```

Prompts for Qwen3 use RULER's `qwen3-nothink` template (`TEMPLATE=qwen3-nothink bash scripts/make_ruler_prompts.sh ...`).

## The deployable 4-bit path

`qk_fast.py` keeps the packed 4-bit keys of the flagged heads in a side cache and scans them with Triton kernels, and
`qk_fast_run.py` runs it inside InfLLM-V2.

## Full attention in the flagged heads inside CPM.cu

`cpmcu/dense_flagged_heads.patch` is a patch for OpenBMB's CPM.cu engine (the commit in `THIRD_PARTY.md`). With it,
`CPMCU_DENSE_KV_HEADS="9.0,16.0,22.0"` gives those key-value heads full attention through the engine's own block mask.
`cpmcu/run_cpmcu.py` runs and times one arm.

## What the scripts above import

`oracle_run.py` (the original harness for InfLLM-V2, which the package's hooks were lifted from), `needle_probe.py`
(the probe's forfeited attention read-out), `ruler_run.py`, `ckpt_guard.py`, `fasa_compat.py` and `free_region.py`.
