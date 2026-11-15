#!/bin/bash
# RULER prompts with RULER's own generator at the commit the paper pins, in the chat template of the MiniCPM4 lineage.
#
#   bash scripts/make_ruler_prompts.sh RULER_DIR TOKENIZER TASK LENGTH N SEED OUT_DIR
#   bash scripts/make_ruler_prompts.sh ~/RULER openbmb/InfLLM-V2-Long-Sparse-Base niah_multikey_3 32768 200 43 data/eval
#   bash scripts/make_ruler_prompts.sh ~/RULER openbmb/InfLLM-V2-Long-Sparse-Base niah_multikey_3 32768 60 7 data/calibration
#
# RULER_DIR is a clone of https://github.com/NVIDIA/RULER at commit dac2a4057d0fcd7d98ac18eaae72db825023bd3e with
# third_party/ruler_templates.patch applied (it adds the chat templates; RULER's generator is otherwise unchanged):
#   git clone https://github.com/NVIDIA/RULER && cd RULER && git checkout dac2a4057d0fcd7d98ac18eaae72db825023bd3e
#   git apply /path/to/blind-attention-heads/third_party/ruler_templates.patch
# The paper evaluates on seeds 43 to 47 and calibrates on seed 7. Never calibrate on a seed you evaluate on.
set -euo pipefail
RULER=${1:?RULER_DIR}; TOK=${2:?tokenizer}; TASK=${3:?task}; LEN=${4:?length}; N=${5:?prompts}; SEED=${6:?seed}; OUT=${7:?out dir}
TEMPLATE=${TEMPLATE:-minicpm4}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
cd "$RULER/scripts/data"
PYTHONPATH="$RULER/scripts/data:${PYTHONPATH:-}" python prepare.py --save_dir "$OUT" --benchmark synthetic --task "$TASK" \
  --tokenizer_path "$TOK" --tokenizer_type hf --max_seq_length "$LEN" --num_samples "$N" --random_seed "$SEED" \
  --model_template_type "$TEMPLATE"
echo "prompts: $OUT/$TASK/validation.jsonl"
