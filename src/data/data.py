import pandas as pd
from pandera.typing import DataFrame

from constants import (
    BOOLEAN_COLUMNS,
    COLUMNS_TO_DROP,
    DATASET_DIR,
    DEFAULT_DATASET_NAME,
    DEFAULT_MIN_REVIEWS,
    PERCENTAGE_COLUMNS,
    PRICE_COLUMN,
    REVIEW_SCORES_RATING_COLUMN,
    REVIEWS_AMOUNT_COLUMN,
)
from schemas import ListingSchema

from .helpers import (
    format_boolean,
    format_percentage,
    format_price,
)


def get_listings(
    data_set_name: str = DEFAULT_DATASET_NAME,
) -> DataFrame[ListingSchema]:
    data_set_path = DATASET_DIR / data_set_name

    data = pd.read_csv(data_set_path)
    data = data.drop(columns=COLUMNS_TO_DROP)

    data[PRICE_COLUMN] = data[PRICE_COLUMN].apply(format_price)

    for column in BOOLEAN_COLUMNS:
        data[column] = data[column].apply(format_boolean)

    for column in PERCENTAGE_COLUMNS:
        data[column] = data[column].apply(format_percentage)

    return ListingSchema.validate(data)


def save_listings(listings: DataFrame[ListingSchema], output_path: str) -> None:
    listings.to_csv(output_path, index=False)


def get_listings_without_small_amount_of_reviews(
    listings: DataFrame[ListingSchema], min_reviews: int = DEFAULT_MIN_REVIEWS
) -> DataFrame[ListingSchema]:
    return listings[
        (listings[REVIEWS_AMOUNT_COLUMN] >= min_reviews)
        & (listings[REVIEW_SCORES_RATING_COLUMN].notna())
    ]
