import warnings
from typing import Any

import pandas as pd
from pandera.typing import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

from constants import (
    AMENITIES_COLUMN,
    CATEGORICAL_COLUMNS,
    IMPUTER_STRATEGY,
    NUMERIC_COLUMNS,
)
from data.helpers import parse_amenities
from schemas import ListingSchema


class AmenitiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.mlb = MultiLabelBinarizer()

    def fit(self, X: pd.DataFrame, _y: Any = None) -> "AmenitiesTransformer":
        amenities_list = X[AMENITIES_COLUMN].apply(parse_amenities).tolist()
        self.mlb.fit(amenities_list)
        return self

    def transform(self, X: pd.DataFrame) -> Any:
        amenities_list = X[AMENITIES_COLUMN].apply(parse_amenities).tolist()
        known_classes = set(self.mlb.classes_)
        filtered_amenities = [
            [amenity for amenity in amenities if amenity in known_classes]
            for amenities in amenities_list
        ]
        return self.mlb.transform(filtered_amenities)

    def get_feature_names_out(self, _input_features: Any = None) -> list[str]:
        return [f"amenity_{amenity}" for amenity in self.mlb.classes_]


def get_transformer() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=IMPUTER_STRATEGY)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, drop="first"
    )

    amenities_transformer = AmenitiesTransformer()

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
            ("amenities", amenities_transformer, [AMENITIES_COLUMN]),
        ],
        remainder="drop",
    )


def prepare_data(
    listings: DataFrame[ListingSchema],
    fit: bool = True,
    transformer: ColumnTransformer | None = None,
) -> tuple[pd.DataFrame, ColumnTransformer]:
    if transformer is None:
        transformer = get_transformer()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories.*",
            category=UserWarning,
            module="sklearn.preprocessing._encoders",
        )
        if fit:
            processed_listings = transformer.fit_transform(listings)
        else:
            processed_listings = transformer.transform(listings)

    return (
        pd.DataFrame(processed_listings, columns=transformer.get_feature_names_out()),
        transformer,
    )
