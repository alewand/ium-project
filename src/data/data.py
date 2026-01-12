import pandas as pd
from pandera.typing import DataFrame

from constants import (
    BOOLEAN_COLUMNS,
    COLUMNS_TO_DROP,
    DEFAULT_MIN_REVIEWS,
    PRICE_COLUMN,
    REVIEWS_AMOUNT_COLUMN,
)
from schemas import ListingSchema

from .helpers import (
    format_boolean,
    format_price,
)


def get_listings(data_set_path: str) -> DataFrame[ListingSchema]:
    data = pd.read_csv(data_set_path)
    data = data.drop(columns=COLUMNS_TO_DROP)

    data[PRICE_COLUMN] = data[PRICE_COLUMN].apply(format_price)

    for column in BOOLEAN_COLUMNS:
        data[column] = data[column].apply(format_boolean)

    return ListingSchema.validate(data)


def save_listings(listings: DataFrame[ListingSchema], output_path: str) -> None:
    listings.to_csv(output_path, index=False)


def get_listings_without_small_amount_of_reviews(
    listings: DataFrame[ListingSchema], min_reviews: int = DEFAULT_MIN_REVIEWS
) -> DataFrame[ListingSchema]:
    return listings[listings[REVIEWS_AMOUNT_COLUMN] >= min_reviews]
