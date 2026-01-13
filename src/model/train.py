import json

import joblib
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

from constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_REVIEWS,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_MIN_SAMPLES_SPLIT,
    DEFAULT_MODEL_CONFIG_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SUBSAMPLE,
    DEFAULT_TRANSFORMER_NAME,
    MODEL_DIR,
    REVIEW_SCORES_RATING_COLUMN,
)
from data import get_listings_without_small_amount_of_reviews
from schemas import ListingSchema

from .preprocessing import prepare_data


def split_data(
    listings: DataFrame[ListingSchema],
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[
    DataFrame[ListingSchema], DataFrame[ListingSchema], DataFrame[ListingSchema]
]:
    train_listings, validation_and_test_listings = train_test_split(
        listings,
        test_size=0.3,
        random_state=random_state,
    )

    validation_listings, test_listings = train_test_split(
        validation_and_test_listings,
        test_size=0.5,
        random_state=random_state,
    )

    return train_listings, validation_listings, test_listings


def train_model(
    listings: DataFrame[ListingSchema],
    min_reviews: int = DEFAULT_MIN_REVIEWS,
    rating_weight: float = DEFAULT_MIN_REVIEWS,
    model_name: str = DEFAULT_MODEL_NAME,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int | None = DEFAULT_MAX_DEPTH,
    min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    max_features: str = DEFAULT_MAX_FEATURES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    subsample: float = DEFAULT_SUBSAMPLE,
) -> tuple[GradientBoostingRegressor, dict[str, float], pd.DataFrame, pd.Series]:
    filtered_listings = get_listings_without_small_amount_of_reviews(
        listings, min_reviews
    ).copy()

    train_listings, validation_listings, test_listings = split_data(
        filtered_listings, random_state
    )

    train_features, transformer = prepare_data(train_listings, fit=True)

    train_target = train_listings[REVIEW_SCORES_RATING_COLUMN]

    validation_processed_listings, _ = prepare_data(
        validation_listings, fit=False, transformer=transformer
    )

    validation_target = validation_listings[REVIEW_SCORES_RATING_COLUMN]

    test_processed_listings, _ = prepare_data(
        test_listings, fit=False, transformer=transformer
    )

    test_target = test_listings[REVIEW_SCORES_RATING_COLUMN]

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=random_state,
    )
    model.fit(train_features, train_target)

    validation_predictions = model.predict(validation_processed_listings)

    metrics = {
        "mae": mean_absolute_error(validation_target, validation_predictions),
        "rmse": np.sqrt(mean_squared_error(validation_target, validation_predictions)),
    }

    model_folder = MODEL_DIR / model_name
    model_folder.mkdir(parents=True, exist_ok=True)

    model_path = model_folder / DEFAULT_MODEL_NAME
    transformer_path = model_folder / DEFAULT_TRANSFORMER_NAME
    config_path = model_folder / DEFAULT_MODEL_CONFIG_NAME

    joblib.dump(model, model_path)
    joblib.dump(transformer, transformer_path)

    config = {
        "min_reviews": min_reviews,
        "rating_weight": rating_weight,
    }

    with config_path.open("w") as f:
        json.dump(config, f)

    return model, metrics, test_processed_listings, test_target
