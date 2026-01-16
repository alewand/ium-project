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

train_model "model_baseline" \
    --min-reviews 5 \
    --rating-weight 0.5 \
    --n-estimators 100 \
    --learning-rate 0.1 \
    --random-state 42

train_model "model_deep" \
    --min-reviews 5 \
    --rating-weight 0.5 \
    --n-estimators 200 \
    --learning-rate 0.1 \
    --random-state 42

train_model "model_conservative" \
    --min-reviews 5 \
    --rating-weight 0.5 \
    --n-estimators 200 \
    --learning-rate 0.05 \
    --random-state 42

train_model "model_shallow" \
    --min-reviews 5 \
    --rating-weight 0.5 \
    --n-estimators 150 \
    --max-depth 10 \
    --learning-rate 0.1 \
    --random-state 42

train_model "model_rating_focused" \
    --min-reviews 5 \
    --rating-weight 0.8 \
    --n-estimators 150 \
    --learning-rate 0.1 \
    --random-state 42

train_model "model_prediction_focused" \
    --min-reviews 5 \
    --rating-weight 0.2 \
    --n-estimators 150 \
    --learning-rate 0.1 \
    --random-state 42

train_model "model_high_reviews" \
    --min-reviews 10 \
    --rating-weight 0.5 \
    --n-estimators 150 \
    --learning-rate 0.1 \
    --random-state 42

train_model "model_complex" \
    --min-reviews 5 \
    --rating-weight 0.6 \
    --n-estimators 250 \
    --max-depth 15 \
    --learning-rate 0.08 \
    --subsample 0.8 \
    --min-samples-split 4 \
    --min-samples-leaf 2 \
    --random-state 42

train_model "model_fast" \
    --min-reviews 5 \
    --rating-weight 0.5 \
    --n-estimators 50 \
    --max-depth 8 \
    --learning-rate 0.15 \
    --random-state 42

train_model "model_balanced" \
    --min-reviews 5 \
    --rating-weight 0.5 \
    --n-estimators 150 \
    --max-depth 12 \
    --learning-rate 0.1 \
    --subsample 0.9 \
    --random-state 42

echo "========================================="
echo "Trenowanie wszystkich modeli zakończone!"
echo "========================================="
