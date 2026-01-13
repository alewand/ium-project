import json
from typing import Any

import joblib
import pandas as pd
from pandera.typing import DataFrame
from sklearn.compose import ColumnTransformer

from constants import (
    DEFAULT_MIN_REVIEWS,
    DEFAULT_MODEL_CONFIG_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRANSFORMER_NAME,
    MIN_REVIEWS_KEY,
    MODEL_DIR,
    PREDICTED_RATING_COLUMN,
    RATING_WEIGHT_KEY,
    REVIEW_SCORES_RATING_COLUMN,
    REVIEWS_AMOUNT_COLUMN,
)
from schemas import ListingSchema

from .preprocessing import prepare_data


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[Any, ColumnTransformer, int, float]:
    model_folder = MODEL_DIR / model_name

    model_path = model_folder / DEFAULT_MODEL_NAME
    transformer_path = model_folder / DEFAULT_TRANSFORMER_NAME
    config_path = model_folder / DEFAULT_MODEL_CONFIG_NAME

    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)

    with config_path.open() as f:
        config = json.load(f)

    min_reviews = config[MIN_REVIEWS_KEY]
    rating_weight = config[RATING_WEIGHT_KEY]

    return model, transformer, min_reviews, rating_weight


def predict_ratings(
    listings: DataFrame[ListingSchema],
    model: Any,
    transformer: ColumnTransformer,
) -> pd.Series:
    processed_listings, _ = prepare_data(listings, fit=False, transformer=transformer)
    predictions = model.predict(processed_listings)
    predictions_rounded = pd.Series(
        [round(float(p), 3) for p in predictions], index=listings.index
    )
    return predictions_rounded


def calculate_bayesian_rating(
    actual_rating: float,
    predicted_rating: float,
    num_reviews: int,
    rating_weight: float,
) -> float:
    weight_sum = num_reviews + rating_weight
    return (num_reviews / weight_sum * actual_rating) + (
        rating_weight / weight_sum * predicted_rating
    )


def predict(
    listings: DataFrame[ListingSchema],
    model: Any,
    transformer: ColumnTransformer,
    min_reviews: int = DEFAULT_MIN_REVIEWS,
    rating_weight: float = DEFAULT_MIN_REVIEWS,
) -> tuple[DataFrame[ListingSchema], list[float]]:
    listings_copy = listings.copy()
    listings_to_predict = listings_copy[
        listings_copy[REVIEWS_AMOUNT_COLUMN] < min_reviews
    ]

    predicted_ratings_series = pd.Series(
        dtype=float, index=listings_copy.index, name=PREDICTED_RATING_COLUMN
    )

    if len(listings_to_predict) > 0:
        predicted_ratings = predict_ratings(listings_to_predict, model, transformer)
        predicted_ratings_series.loc[listings_to_predict.index] = (
            predicted_ratings.to_numpy()
        )

    final_ratings = []

    for index in listings_copy.index:
        row = listings_copy.loc[index]
        num_reviews = int(row[REVIEWS_AMOUNT_COLUMN] or 0)
        actual_rating = row[REVIEW_SCORES_RATING_COLUMN]
        actual_rating = None if pd.isna(actual_rating) else float(actual_rating)
        predicted = (
            float(predicted_ratings_series.loc[index])
            if not pd.isna(predicted_ratings_series.loc[index])
            else None
        )

        if num_reviews == 0 or actual_rating is None:
            final_rating = predicted if predicted is not None else 0.0
        elif num_reviews >= min_reviews or predicted is None:
            final_rating = actual_rating
        else:
            final_rating = calculate_bayesian_rating(
                actual_rating, predicted, num_reviews, rating_weight
            )

        final_ratings.append(final_rating)

    sorted_indices = sorted(
        range(len(final_ratings)), key=lambda i: final_ratings[i], reverse=True
    )

    sorted_listings = listings_copy.iloc[sorted_indices].reset_index(drop=True)
    sorted_ratings = [final_ratings[i] for i in sorted_indices]

    return ListingSchema.validate(sorted_listings), sorted_ratings
