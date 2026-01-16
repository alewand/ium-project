#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR" || exit 1

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

train_model() {
    local model_name=$1
    shift
    echo "========================================="
    echo "Training model: $model_name"
    echo "Parameters: $*"
    echo "========================================="
    uv run python src/train_model.py --model-name "$model_name" "$@"
    echo ""
}

MIN_REVIEWS=10
RATING_WEIGHT=0.5
N_ESTIMATORS=150
RANDOM_STATE=42

train_model "model_lr0.01" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.01 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.02" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.02 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.03" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.03 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.04" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.04 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.05" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.05 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.06" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.06 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.07" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.07 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.08" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.08 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.09" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.09 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.10" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.1 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.12" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.12 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.15" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.15 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.18" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.18 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.20" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.2 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.25" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.25 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.30" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.3 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.35" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.35 \
    --random-state "$RANDOM_STATE"

train_model "model_lr0.40" \
    --min-reviews "$MIN_REVIEWS" \
    --rating-weight "$RATING_WEIGHT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate 0.4 \
    --random-state "$RANDOM_STATE"

echo "========================================="
echo "Trenowanie wszystkich modeli zakończone!"
echo "Łącznie wytrenowano 18 modeli"
echo "========================================="
