import json
from typing import TypedDict

import pandas as pd
from fastapi import APIRouter, HTTPException
from pandera.typing import DataFrame

from constants import REVIEW_SCORES_RATING_COLUMN, SERVICE_CONFIG_PATH
from model import load_model
from model.predict import predict
from schemas import Listing, ListingSchema

from .schema import (
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    RankListingsRequest,
    RankListingsResponse,
)

router = APIRouter(prefix="/api/v1", tags=["api"])


class Config(TypedDict):
    model_name: str
    transformer_name: str


def get_config() -> Config:
    with SERVICE_CONFIG_PATH.open() as f:
        config: Config = json.load(f)
        return config


def get_model() -> tuple:
    config = get_config()
    return load_model(
        model_name=config["model_name"],
        transformer_name=config["transformer_name"],
    )


def listings_to_dataframe(listings: list[Listing]) -> DataFrame[ListingSchema]:
    dataframe_listings_list = []
    for listing in listings:
        listing_dict = listing.model_dump()
        int_columns = [
            "accommodates",
            "minimum_nights",
            "maximum_nights",
            "minimum_minimum_nights",
            "maximum_minimum_nights",
            "minimum_maximum_nights",
            "maximum_maximum_nights",
            "number_of_reviews",
            "availability_30",
            "availability_60",
            "availability_90",
            "availability_365",
        ]
        for col in int_columns:
            if col in listing_dict and listing_dict[col] is None:
                listing_dict[col] = pd.NA
        dataframe_listings_list.append(listing_dict)
    dataframe_listings = pd.DataFrame(dataframe_listings_list)
    for col in int_columns:
        if col in dataframe_listings.columns:
            dataframe_listings[col] = dataframe_listings[col].astype("Float64")

    return ListingSchema.validate(dataframe_listings)


def dataframe_to_listings(
    dataframe_listings: DataFrame[ListingSchema],
) -> list[Listing]:
    if REVIEW_SCORES_RATING_COLUMN in dataframe_listings.columns:
        dataframe_listings = dataframe_listings.sort_values(
            REVIEW_SCORES_RATING_COLUMN, ascending=False
        )

    dataframe_listings = dataframe_listings.copy()

    listings = []
    for _, row in dataframe_listings.iterrows():
        row_dict = row.to_dict()
        for key, value in row_dict.items():
            if pd.isna(value):
                row_dict[key] = None
        listings.append(Listing.model_validate(row_dict))

    return listings


def update_config(config_data: ConfigUpdateRequest) -> None:
    config_dict: Config = {
        "model_name": config_data.model_name,
        "transformer_name": config_data.transformer_name,
    }

    with SERVICE_CONFIG_PATH.open("w") as f:
        json.dump(config_dict, f, indent=2)


@router.post("/rank-listings")
def rank_listings(request: RankListingsRequest) -> RankListingsResponse:
    try:
        listings_dataframe = listings_to_dataframe(request.listings)
        model, transformer, min_reviews, rating_weight = get_model()
        ranked_dataframe = predict(
            listings_dataframe,
            model,
            transformer,
            min_reviews=min_reviews,
            rating_weight=rating_weight,
        )
        ranked_listings = dataframe_to_listings(ranked_dataframe)

        return RankListingsResponse(listings=ranked_listings)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rank listings: {e}",
        ) from e


@router.post("/config", response_model=ConfigUpdateResponse)
def update_server_config(request: ConfigUpdateRequest) -> ConfigUpdateResponse:
    try:
        update_config(request)

        return ConfigUpdateResponse(
            message="Server configuration updated successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update server configuration: {e}",
        ) from e
