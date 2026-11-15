# Third-party code and models

This repository contains no model weights, no benchmark data and no third-party source code. It hooks into releases
that you install yourself. The versions below are the ones the paper's runs used.

## Models

| Name in the paper | Hugging Face id | Revision |
|---|---|---|
| InfLLM-V2-8B | `openbmb/InfLLM-V2-Long-Sparse-Base` | `ef54e627b7f97973fb70372eb73cce6c639cb7d1` |
| MiniCPM4-8B | `openbmb/MiniCPM4-8B` | `bb2ae14cf59d4ca769c4e42ece54cc3b82a58ef7` |
| MiniCPM4.1-8B | `openbmb/MiniCPM4.1-8B` | `3a8dfed9c79a45e07dbff95bcd49d792343fa1a3` |
| Qwen3-4B, 8B, 14B | `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B` | |
| SeerAttention-R gates | `SeerAttention/SeerAttention-Decode-Qwen3-{4B,8B,14B}-AttnGates` | |

Each model keeps its own license. Read it on the model's page before use.

## Code you install

| What | Where | Version |
|---|---|---|
| InfLLM-V2 attention kernels (`infllm_v2`) | https://github.com/OpenBMB/infllmv2_cuda_impl | commit `93cf2ec28e5a7acebe3f0bb7329b6c73a1be91f6` |
| FlashAttention | https://github.com/Dao-AILab/flash-attention | 2.7.4.post1 |
| RULER (prompt generator and metric) | https://github.com/NVIDIA/RULER | commit `dac2a4057d0fcd7d98ac18eaae72db825023bd3e` |
| CPM.cu (for `research/cpmcu/` only) | https://github.com/OpenBMB/CPM.cu | commit `23aa7b7fefc537113166691cec63d5baa5209ebe` |
| SeerAttention (for `research/seerr_run.py` only) | https://github.com/microsoft/SeerAttention | commit `aba03e3f2caefd0ccd21e576670aa830b748c84e` |

## The one patch in this repository

`third_party/ruler_templates.patch` adds three chat templates to RULER's `scripts/data/template.py`, so that RULER's own
generator writes prompts in the chat format of the paper's checkpoints. It changes nothing else in RULER. RULER is
licensed under Apache 2.0, and the patch is distributed under the same terms.

`research/cpmcu/dense_flagged_heads.patch` is a patch for CPM.cu at the commit above. It lets listed key-value heads read
every block, and is distributed under CPM.cu's license.

`blindheads/ruler.py` restates RULER's two string-match metrics (a few lines each) so that scoring needs no RULER clone.
