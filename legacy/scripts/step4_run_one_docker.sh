#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rccn-5090-pytorch270-cu128}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
METHOD="${METHOD:?METHOD is required}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
DATASET="${DATASET:-NUDT-SIRST}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-4}"
THRESHOLD="${THRESHOLD:-0.5}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
EXPORT_EXTRA_ARGS="${EXPORT_EXTRA_ARGS:-}"
RESUME="${RESUME:-}"
ROOT="/home/ly/AAAI/OHCM-MSHNet"
ROOT_IN_CONTAINER="/home/AAAI/OHCM-MSHNet"
RUN_ROOT="$ROOT/results/step4_ohcm_full_proto/$RUN_ID"
OUT="$RUN_ROOT/$METHOD/$DATASET/seed_$SEED"
OUT_IN_CONTAINER="$ROOT_IN_CONTAINER/results/step4_ohcm_full_proto/$RUN_ID/$METHOD/$DATASET/seed_$SEED"
HC_LIST="$ROOT_IN_CONTAINER/results/step0_baseline/20260611_155232/step2_hcset/hcset_${DATASET}.txt"
RESUME_ARG=""
if [[ -n "$RESUME" ]]; then
  resume_in_container="$RESUME"
  if [[ "$RESUME" == "$ROOT/"* ]]; then
    resume_in_container="$ROOT_IN_CONTAINER/${RESUME#"$ROOT/"}"
  fi
  RESUME_ARG="--resume '$resume_in_container'"
fi

mkdir -p "$OUT"

name_dataset="$(printf '%s' "$DATASET" | tr -c '[:alnum:]' '_')"
name_method="$(printf '%s' "$METHOD" | tr -c '[:alnum:]' '_')"
container_name="ohcm_step4_${name_method}_${name_dataset}_s${SEED}_${RUN_ID}"

docker run --rm \
  --name "$container_name" \
  --gpus "\"device=${GPU}\"" \
  -v /home/ly:/home \
  -w "$ROOT_IN_CONTAINER" \
  "$IMAGE" \
  bash -lc "set -euo pipefail
    mkdir -p '$OUT_IN_CONTAINER'
    python train.py \
      --model_names '$MODEL_NAME' \
      --dataset_names '$DATASET' \
      --batchSize '$BATCH_SIZE' \
      --patchSize 256 \
      --nEpochs '$EPOCHS' \
      --optimizer_name Adagrad \
      --threads 1 \
      --intervals 10 \
      --seed '$SEED' \
      --mshnet_warm_epoch 5 \
      --mshnet_in_channels 1 \
      --save '$OUT_IN_CONTAINER/checkpoints' \
      $RESUME_ARG \
      $TRAIN_EXTRA_ARGS \
      2>&1 | tee -a '$OUT_IN_CONTAINER/train_console.log'

    python tools/export_step0_predictions.py \
      --model_name '$MODEL_NAME' \
      --dataset_dir ./datasets \
      --dataset_name '$DATASET' \
      --train_dataset_name '$DATASET' \
      --checkpoint '$OUT_IN_CONTAINER/checkpoints/$DATASET/${MODEL_NAME}_${EPOCHS}.pth.tar' \
      --output_dir '$OUT_IN_CONTAINER/exports' \
      --threshold '$THRESHOLD' \
      --seed '$SEED' \
      --mshnet_warm_epoch 5 \
      --mshnet_in_channels 1 \
      $EXPORT_EXTRA_ARGS \
      2>&1 | tee '$OUT_IN_CONTAINER/export_console.log'

    python tools/evaluate_prediction_exports.py \
      --dataset_dir ./datasets \
      --dataset_name '$DATASET' \
      --train_dataset_name '$DATASET' \
      --exports_dir '$OUT_IN_CONTAINER/exports' \
      --output_dir '$OUT_IN_CONTAINER/eval_full' \
      --method '$METHOD' \
      --seed '$SEED' \
      --threshold '$THRESHOLD' \
      2>&1 | tee '$OUT_IN_CONTAINER/eval_full_console.log'

    if [[ -f '$HC_LIST' ]]; then
      python tools/evaluate_prediction_exports.py \
        --dataset_dir ./datasets \
        --dataset_name '$DATASET' \
        --train_dataset_name '$DATASET' \
        --exports_dir '$OUT_IN_CONTAINER/exports' \
        --output_dir '$OUT_IN_CONTAINER/eval_hcset' \
        --image_list '$HC_LIST' \
        --method '$METHOD' \
        --seed '$SEED' \
        --threshold '$THRESHOLD' \
        2>&1 | tee '$OUT_IN_CONTAINER/eval_hcset_console.log'
    fi

    python tools/analyze_step1_hard_clutter.py \
      --dataset_dir ./datasets \
      --dataset_name '$DATASET' \
      --exports_dir '$OUT_IN_CONTAINER/exports' \
      --output_dir '$OUT_IN_CONTAINER/fp_analysis' \
      --seed '$SEED' \
      --threshold '$THRESHOLD' \
      2>&1 | tee '$OUT_IN_CONTAINER/fp_analysis_console.log'

    chown -R 1004:1004 '$OUT_IN_CONTAINER'"
