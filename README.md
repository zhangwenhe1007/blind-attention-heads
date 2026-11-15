# Blind Attention Heads

Code for **"Lost in a Few Heads: Diagnosing and Repairing Block Selection in Sparse Attention"**
(Wenhe Zhang and Boxing Chen).

Sparse attention models read a few blocks of keys for every query, at a budget fixed when the model ships, and can lose
much of their long-context retrieval at that budget. The paper shows that the loss is made in a few heads, finds those
heads with a calibration probe, and fixes them without training. This repository holds the three things you need to
use that result:

1. **The calibration.** A probe scores every head by how much of the answer fixing its selection restores, and a
   threshold rule turns the scores into a set of flagged heads.
2. **The fixes.** Full attention in the flagged heads, a 4-bit scorer that keeps the shipped budget, or a larger
   budget in the flagged layers, each installed as a hook on the release's own attention. No weight changes.
3. **The flagged heads** of six checkpoints, so you can apply a fix without running the calibration.

The worked integration is the InfLLM-V2 release (`openbmb/InfLLM-V2-Long-Sparse-Base`, the paper's InfLLM-V2-8B) and
the two checkpoints that share its selector, `openbmb/MiniCPM4-8B` and `openbmb/MiniCPM4.1-8B`.

## What is here

| Path | Contents |
|---|---|
| `blindheads/` | the package: head sets, the threshold rule, the probe's arithmetic, and the integration with InfLLM-V2 |
| `heads/` | the flagged heads of six checkpoints, with the score of every head |
| `scripts/` | `calibrate.py`, `evaluate.py` and `make_ruler_prompts.sh` |
| `tests/` | unit tests, CPU only |
| `research/` | SeerAttention-R, the deployable 4-bit path and a CPM.cu patch, as research scripts (see `research/README.md`) |
| `third_party/` | one small patch that adds chat templates to RULER |

## Install

```bash
git clone https://github.com/zhangwenhe1007/blind-attention-heads && cd blind-attention-heads
pip install -e .            # the core is pure Python
pip install -r requirements.txt
```

The InfLLM-V2 attention kernels are not on PyPI. Build them from
[OpenBMB/infllmv2_cuda_impl](https://github.com/OpenBMB/infllmv2_cuda_impl), as the model card describes.
`THIRD_PARTY.md` lists the exact versions the paper used. One GPU with 48 GB runs everything below at 32K.

```bash
pytest          # a few seconds, no GPU
```

The hooks were checked on a GPU against the scripts behind the paper's numbers. On the same prompts they generate
identical text, for every mode of `scripts/evaluate.py` on InfLLM-V2-8B and for the shipped selector and the default
fix on MiniCPM4-8B and MiniCPM4.1-8B.

## Quickstart: fix a checkpoint with its flagged heads

Make RULER multi-key prompts with RULER's own generator (see the header of `scripts/make_ruler_prompts.sh` for the
RULER commit and the patch):

```bash
bash scripts/make_ruler_prompts.sh ~/RULER openbmb/InfLLM-V2-Long-Sparse-Base niah_multikey_3 32768 200 43 data/eval
```

Run the shipped selector, the fix and the dense model on the same prompts:

```bash
M=openbmb/InfLLM-V2-Long-Sparse-Base
P=data/eval/niah_multikey_3/validation.jsonl
python scripts/evaluate.py --model $M --prompts $P --mode shipped
python scripts/evaluate.py --model $M --prompts $P --mode full-attention-heads --heads infllm-v2-8b
python scripts/evaluate.py --model $M --prompts $P --mode dense
```

The paper's numbers for this setting, multi-key retrieval at 32K, as the mean over five seeds of 200 prompts:

| Shipped selector | Full attention in the three flagged heads | Dense |
|---|---|---|
| 63.1 | 96.0 | 97.8 |

A single seed moves by a few points around these means.

**MiniCPM4-8B and MiniCPM4.1-8B.** These two ship with sparse attention switched off, and their model cards give the
`sparse_config` entry that switches it on. `infllm_v2.load` and the scripts add that entry when a checkpoint has none,
so the same commands run with `--model openbmb/MiniCPM4.1-8B --heads minicpm4.1-8b`. The three checkpoints share one
tokenizer and one chat template, so they also share prompts.

## Calibrate a checkpoint yourself

Calibration prompts must come from a seed that no evaluation uses, and they must contain look-alike competitors.
RULER's multi-key task does. A single needle among filler sentences flags no head at all.

```bash
bash scripts/make_ruler_prompts.sh ~/RULER $M niah_multikey_3 32768 60 7 data/calibration
python scripts/calibrate.py --model $M --prompts data/calibration/niah_multikey_3/validation.jsonl --out my_heads.json
python scripts/evaluate.py --model $M --prompts $P --mode full-attention-heads --heads my_heads.json
```

`calibrate.py` runs the two steps of Section 5 of the paper.

- **The probe.** Every head gets a gate that blends its output over the shipped selection with its output over every
  key. The fix gain of a head is the integrated gradient of the answer's log-probability along the path from the
  shipped selector to full attention, in absolute value per prompt.
- **The rule.** The threshold on the score is lowered by a factor of the square root of two per step. Each step that
  adds heads is run on the calibration prompts with full attention in its heads, and the first step that closes 90
  percent of the gap between the shipped selector and full attention is kept.

The probe is the expensive part: at 32K and 16 midpoint nodes it takes about four minutes per prompt on an H100.
`--probe_prompts` and `--rule_prompts` set how many prompts each step uses. Two runs of the probe on the same prompt
differ in the third decimal of a score, because the backward pass of the attention kernels is not deterministic.

## Use it from Python

```python
from blindheads import infllm_v2, load_heads, ruler

model, tok, cfg = infllm_v2.load("openbmb/InfLLM-V2-Long-Sparse-Base")
heads = load_heads("infllm-v2-8b")                     # ["9.0", "16.0", "22.0"]: layer.head, key-value heads

with infllm_v2.full_attention_heads(model, heads.heads) as fix:
    text = infllm_v2.generate(model, tok, prompt, max_new_tokens=128)
    print(fix.stats)                                   # how often the hook ran, at prefill and at decode
# outside the block the model is the shipped one again
```

The fixes are hooks on the model object, so they also install on a model you loaded yourself, as long as its
configuration carries `sparse_config`.

The rule and the head sets do not depend on a model:

```python
from blindheads import threshold_rule

kept = threshold_rule(scores, evaluate, shipped=acc_shipped, full=acc_full)   # evaluate(units) -> calibration accuracy
```

## The flagged heads

| Checkpoint | Selector | Flagged | File in `heads/` |
|---|---|---|---|
| InfLLM-V2-8B | InfLLM-V2 | 3 of 64 key-value heads | `infllm-v2-8b` |
| MiniCPM4-8B | InfLLM-V2 | 3 of 64 | `minicpm4-8b` |
| MiniCPM4.1-8B | InfLLM-V2 | 3 of 64 | `minicpm4.1-8b` |
| Qwen3-4B | SeerAttention-R | 9 of 288 | `qwen3-4b_seerattention-r` |
| Qwen3-8B | SeerAttention-R | 18 of 288 | `qwen3-8b_seerattention-r` |
| Qwen3-14B | SeerAttention-R | 5 of 320 | `qwen3-14b_seerattention-r` |

The three checkpoints of the MiniCPM4-8B lineage lose retrieval in the same three heads, the first key-value head of
layers 9, 16 and 22. Each file also carries the score of every head. For a checkpoint that is not listed, run the
calibration.

## The other fixes and the diagnostic

`scripts/evaluate.py` runs every arm of the paper on InfLLM-V2 (the same functions are in `blindheads.infllm_v2`):

| Mode | What it is |
|---|---|
| `--mode full-attention-heads --heads H` | the default fix |
| `--mode scorer --tokens decode --heads H` | the 4-bit scorer at the shipped budget, at the decode tokens. Harmless on every task the paper measured |
| `--mode scorer --tokens question --heads H` | the scorer, also at the last 128 prompt tokens |
| `--mode scorer --tokens every --heads H` | the scorer at every token |
| `--mode larger-budget --heads H` | four times the free slots in the layers of the flagged heads |
| `--mode exact-mass --tokens T [--heads H]` | the diagnostic of Section 5: selection by exact attention mass at the shipped budget. It reads every key to choose the blocks, so it measures what the choice of blocks costs and is not a way to run the model |

## Beyond InfLLM-V2

The package covers the InfLLM-V2 release. `research/` holds what it does not cover yet, as the scripts the paper used:

- **SeerAttention-R on Qwen3.** The evaluation arms and the calibration probe for the SeerAttention-R decode gates.
- **The deployable 4-bit path.** A side cache of packed 4-bit keys scanned with Triton kernels.
- **CPM.cu.** A patch that gives the flagged heads full attention inside OpenBMB's inference engine.

## Citation

```bibtex
@article{zhang2026lost,
  title  = {Lost in a Few Heads: Diagnosing and Repairing Block Selection in Sparse Attention},
  author = {Zhang, Wenhe and Chen, Boxing},
  year   = {2026}
}
```

## License

MIT. See `LICENSE`. Models and third-party code keep their own licenses (`THIRD_PARTY.md`).
