import pandas as pd
from pandera.typing import DataFrame

from constants import NULLABLE_INT_COLUMNS
from schemas import Listing, ListingSchema


def listings_to_dataframe(listings: list[Listing]) -> DataFrame[ListingSchema]:
    dataframe_listings_list = []
    for listing in listings:
        listing_dict = listing.model_dump()
        for col in NULLABLE_INT_COLUMNS:
            if col in listing_dict and listing_dict[col] is None:
                listing_dict[col] = pd.NA
        dataframe_listings_list.append(listing_dict)

    dataframe_listings = pd.DataFrame(dataframe_listings_list)

    for col in NULLABLE_INT_COLUMNS:
        if col in dataframe_listings.columns:
            dataframe_listings[col] = dataframe_listings[col].astype("Float64")

    return ListingSchema.validate(dataframe_listings)


def dataframe_to_listings(
    dataframe_listings: DataFrame[ListingSchema],
) -> list[Listing]:
    dataframe_listings = dataframe_listings.copy()

    listings = []
    for _, row in dataframe_listings.iterrows():
        row_dict = row.to_dict()
        for key, value in row_dict.items():
            if pd.isna(value):
                row_dict[key] = None
        listings.append(Listing.model_validate(row_dict))

    return listings
