import argparse

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from constants import (
    DEFAULT_DATASET_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_RANDOM_STATE,
    REVIEW_SCORES_RATING_COLUMN,
)
from data import get_listings, get_listings_without_small_amount_of_reviews
from model import load_model
from model.preprocessing import prepare_data
from model.train import split_data


def get_arguments() -> tuple[str, str, int]:
    parser = argparse.ArgumentParser(description="Test trained model on test dataset")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help=f"Dataset file name (default: {DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random state for data splitting (default: {DEFAULT_RANDOM_STATE})",
    )

    arguments = parser.parse_args()

    return arguments.model_name, arguments.dataset, arguments.random_state


def test_model(model_name: str, dataset_name: str, random_state: int) -> None:
    print(f"Loading model '{model_name}'...")
    model, transformer, min_reviews, _ = load_model(model_name)

    print(f"Loading data '{dataset_name}'...")
    listings = get_listings(dataset_name)

    filtered_listings = get_listings_without_small_amount_of_reviews(
        listings, min_reviews
    ).copy()
    _, _, test_listings = split_data(filtered_listings, random_state)

    test_processed_listings, _ = prepare_data(
        test_listings, fit=False, transformer=transformer
    )

    test_target = test_listings[REVIEW_SCORES_RATING_COLUMN]

    print("Making predictions...")

    test_predictions = model.predict(test_processed_listings)

    print("Calculating metrics...")
    mae = mean_absolute_error(test_target, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_target, test_predictions))

    print(f"Model testing {model_name} completed!")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")


if __name__ == "__main__":
    model_name, dataset_name, random_state = get_arguments()
    test_model(model_name, dataset_name, random_state)
