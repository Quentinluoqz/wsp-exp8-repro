#!/usr/bin/env bash
# Items 1 and 2 on a laptop. Run from the repository root after cloning the logs and downloading the model
# (see README). Paths can be overridden: LOGS=... MODEL=... bash scripts/run_local.sh
set -euo pipefail

LOGS="${LOGS:-lm_understand-logs}"
MODEL="${MODEL:-models/Qwen3.5-0.8B}"
E="$LOGS/exp8_math"
ST="$MODEL/model.safetensors-00001-of-00001.safetensors"
EDITED="${EDITED:-edited}"

# 1. rebuild the three evaluated edits (one float32 .npy per edited tensor, hash-checked)
for r in p10c p25c p50c; do
    python "$E/code/apply_edit.py" "$ST" "$E/runs/$r/edit" "$EDITED/$r"
done

# 2. item 1: released evaluation outputs
python item1_eval_analysis/analyze_exp8.py "$E/eval" item1_eval_analysis/out

# 3. item 2: weight analysis of the edits, and the outlier-dimension control
python item2_weight_analysis/analyze_edits.py "$ST" "$MODEL/tokenizer.json" "$EDITED" "$E/code" item2_weight_analysis/out
python item2_weight_analysis/control_outlier_dims.py "$ST" "$EDITED" "$E/code" | tee item2_weight_analysis/out/control_outlier_dims.txt
