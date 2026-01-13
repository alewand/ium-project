import argparse

from constants import (
    DEFAULT_BOOTSTRAP,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_REVIEWS,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_MIN_SAMPLES_SPLIT,
    DEFAULT_MODEL_NAME,
    DEFAULT_N_ESTIMATORS,
)
from data import get_listings
from model import train_model


def parse_none_int(value: str) -> int | None:
    if value.lower() == "none":
        return None
    return int(value)


def parse_none_float(value: str) -> float | None:
    if value.lower() == "none":
        return None
    return float(value)


def parse_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes", "on")


def get_arguments() -> tuple[
    str, int, float, int, int | None, int, int, str, bool, float | None
]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
    )

    parser.add_argument(
        "--min-reviews",
        type=int,
        default=DEFAULT_MIN_REVIEWS,
    )

    parser.add_argument(
        "--rating-weight",
        type=float,
        default=DEFAULT_MIN_REVIEWS,
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
    )

    parser.add_argument(
        "--max-depth",
        type=parse_none_int,
        default=DEFAULT_MAX_DEPTH,
    )

    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=DEFAULT_MIN_SAMPLES_SPLIT,
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=DEFAULT_MIN_SAMPLES_LEAF,
    )

    parser.add_argument(
        "--max-features",
        type=str,
        default=DEFAULT_MAX_FEATURES,
    )

    parser.add_argument(
        "--bootstrap",
        type=parse_bool,
        default=DEFAULT_BOOTSTRAP,
    )

    parser.add_argument(
        "--max-samples",
        type=parse_none_float,
        default=DEFAULT_MAX_SAMPLES,
    )

    arguments = parser.parse_args()

    return (
        arguments.model_name,
        arguments.min_reviews,
        arguments.rating_weight,
        arguments.n_estimators,
        arguments.max_depth,
        arguments.min_samples_split,
        arguments.min_samples_leaf,
        arguments.max_features,
        arguments.bootstrap,
        arguments.max_samples,
    )


if __name__ == "__main__":
    print("Loading data...")
    listings = get_listings()
    (
        model_name,
        min_reviews,
        rating_weight,
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap,
        max_samples,
    ) = get_arguments()

    print("Training model...")
    _, metrics, _, _ = train_model(
        listings,
        model_name=model_name,
        min_reviews=min_reviews,
        rating_weight=rating_weight,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        max_samples=max_samples,
    )

    print(f"\nModel training {model_name} completed!")
    print(f"Validation MAE: {metrics['mae']:.4f}")
    print(f"Validation RMSE: {metrics['rmse']:.4f}")
    print(f"Validation R²: {metrics['r2']:.4f}")
