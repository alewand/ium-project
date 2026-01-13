import argparse

from constants import DEFAULT_MIN_REVIEWS, DEFAULT_MODEL_NAME, DEFAULT_TRANSFORMER_NAME
from data import get_listings
from model import train_model


def get_arguments() -> tuple[str, str, int, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--transformer-name",
        type=str,
        default=DEFAULT_TRANSFORMER_NAME,
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

    arguments = parser.parse_args()

    return (
        arguments.model_name,
        arguments.transformer_name,
        arguments.min_reviews,
        arguments.rating_weight,
    )


if __name__ == "__main__":
    print("Loading data...")
    listings = get_listings()
    model_name, transformer_name, min_reviews, rating_weight = get_arguments()
    print("Training model...")
    model, metrics, test_features, test_target = train_model(
        listings,
        model_name=model_name,
        transformer_name=transformer_name,
        min_reviews=min_reviews,
        rating_weight=rating_weight,
    )
    print("\nModel training completed!")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
