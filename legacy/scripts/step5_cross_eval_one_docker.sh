#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rccn-5090-pytorch270-cu128}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
METHOD="${METHOD:?METHOD is required}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
TRAIN_DATASET="${TRAIN_DATASET:?TRAIN_DATASET is required}"
TEST_DATASET="${TEST_DATASET:?TEST_DATASET is required}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-400}"
THRESHOLD="${THRESHOLD:-0.5}"
EXPORT_EXTRA_ARGS="${EXPORT_EXTRA_ARGS:-}"
ROOT="/home/ly/AAAI/OHCM-MSHNet"
ROOT_IN_CONTAINER="/home/AAAI/OHCM-MSHNet"
RUN_ROOT="$ROOT/results/step5_experiments/$RUN_ID"
CHECKPOINT="${CHECKPOINT:-$RUN_ROOT/$METHOD/$TRAIN_DATASET/seed_$SEED/checkpoints/$TRAIN_DATASET/${MODEL_NAME}_${EPOCHS}.pth.tar}"
OUT="$RUN_ROOT/$METHOD/cross_${TRAIN_DATASET}_to_${TEST_DATASET}/seed_$SEED"
OUT_IN_CONTAINER="$ROOT_IN_CONTAINER/results/step5_experiments/$RUN_ID/$METHOD/cross_${TRAIN_DATASET}_to_${TEST_DATASET}/seed_$SEED"
CKPT_IN_CONTAINER="$CHECKPOINT"
if [[ "$CHECKPOINT" == "$ROOT/"* ]]; then
  CKPT_IN_CONTAINER="$ROOT_IN_CONTAINER/${CHECKPOINT#"$ROOT/"}"
fi

mkdir -p "$OUT"

name_pair="$(printf '%s_to_%s' "$TRAIN_DATASET" "$TEST_DATASET" | tr -c '[:alnum:]' '_')"
name_method="$(printf '%s' "$METHOD" | tr -c '[:alnum:]' '_')"
container_name="ohcm_cross_${name_method}_${name_pair}_s${SEED}_${RUN_ID}"

docker run --rm \
  --name "$container_name" \
  --gpus "\"device=${GPU}\"" \
  -v /home/ly:/home \
  -w "$ROOT_IN_CONTAINER" \
  "$IMAGE" \
  bash -lc "set -euo pipefail
    mkdir -p '$OUT_IN_CONTAINER'
    python tools/export_step0_predictions.py \
      --model_name '$MODEL_NAME' \
      --dataset_dir ./datasets \
      --dataset_name '$TEST_DATASET' \
      --train_dataset_name '$TRAIN_DATASET' \
      --checkpoint '$CKPT_IN_CONTAINER' \
      --output_dir '$OUT_IN_CONTAINER/exports' \
      --threshold '$THRESHOLD' \
      --seed '$SEED' \
      --mshnet_warm_epoch 5 \
      --mshnet_in_channels 1 \
      $EXPORT_EXTRA_ARGS \
      2>&1 | tee '$OUT_IN_CONTAINER/export_console.log'

    python tools/evaluate_prediction_exports.py \
      --dataset_dir ./datasets \
      --dataset_name '$TEST_DATASET' \
      --train_dataset_name '$TRAIN_DATASET' \
      --exports_dir '$OUT_IN_CONTAINER/exports' \
      --output_dir '$OUT_IN_CONTAINER/eval_full' \
      --method '$METHOD' \
      --seed '$SEED' \
      --threshold '$THRESHOLD' \
      2>&1 | tee '$OUT_IN_CONTAINER/eval_full_console.log'

    if [[ -f '$ROOT_IN_CONTAINER/results/step0_baseline/20260611_155232/step2_hcset/hcset_${TEST_DATASET}.txt' ]]; then
      python tools/evaluate_prediction_exports.py \
        --dataset_dir ./datasets \
        --dataset_name '$TEST_DATASET' \
        --train_dataset_name '$TRAIN_DATASET' \
        --exports_dir '$OUT_IN_CONTAINER/exports' \
        --output_dir '$OUT_IN_CONTAINER/eval_hcset' \
        --image_list '$ROOT_IN_CONTAINER/results/step0_baseline/20260611_155232/step2_hcset/hcset_${TEST_DATASET}.txt' \
        --method '$METHOD' \
        --seed '$SEED' \
        --threshold '$THRESHOLD' \
        2>&1 | tee '$OUT_IN_CONTAINER/eval_hcset_console.log'
    fi
    chown -R 1004:1004 '$OUT_IN_CONTAINER'"
