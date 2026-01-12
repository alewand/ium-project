import pandera.pandas as pa
from pandera.typing import Series


class ListingSchema(pa.DataFrameModel):
    id: Series[int] = pa.Field(nullable=False, unique=True, coerce=True)
    host_response_time: Series[str] = pa.Field(nullable=True, coerce=True)
    host_is_superhost: Series[bool] = pa.Field(nullable=True, coerce=True)
    host_listings_count: Series[float] = pa.Field(nullable=True, ge=0, coerce=True)
    host_total_listings_count: Series[float] = pa.Field(
        nullable=True, ge=0, coerce=True
    )
    host_identity_verified: Series[bool] = pa.Field(nullable=True, coerce=True)
    neighbourhood_group_cleansed: Series[str] = pa.Field(nullable=True, coerce=True)
    latitude: Series[float] = pa.Field(nullable=True, ge=-90, le=90, coerce=True)
    longitude: Series[float] = pa.Field(nullable=True, ge=-180, le=180, coerce=True)
    room_type: Series[str] = pa.Field(nullable=True, coerce=True)
    accommodates: Series[int] = pa.Field(nullable=True, ge=1, coerce=True)
    bathrooms: Series[float] = pa.Field(nullable=True, ge=0, coerce=True)
    bedrooms: Series[float] = pa.Field(nullable=True, ge=0, coerce=True)
    beds: Series[float] = pa.Field(nullable=True, ge=0, coerce=True)
    price: Series[float] = pa.Field(nullable=True, gt=0, coerce=True)
    minimum_nights: Series[int] = pa.Field(nullable=True, ge=1, coerce=True)
    maximum_nights: Series[int] = pa.Field(nullable=True, ge=1, coerce=True)
    minimum_minimum_nights: Series[int] = pa.Field(nullable=True, ge=1, coerce=True)
    maximum_minimum_nights: Series[int] = pa.Field(nullable=True, ge=1, coerce=True)
    minimum_maximum_nights: Series[int] = pa.Field(nullable=True, ge=1, coerce=True)
    maximum_maximum_nights: Series[int] = pa.Field(nullable=True, ge=1, coerce=True)
    minimum_nights_avg_ntm: Series[float] = pa.Field(nullable=True, ge=1, coerce=True)
    maximum_nights_avg_ntm: Series[float] = pa.Field(nullable=True, ge=1, coerce=True)
    number_of_reviews: Series[int] = pa.Field(nullable=True, ge=0, coerce=True)
    review_scores_rating: Series[float] = pa.Field(
        nullable=True, ge=0, le=5, coerce=True
    )
    has_availability: Series[bool] = pa.Field(nullable=True, coerce=True)
    availability_30: Series[int] = pa.Field(nullable=True, ge=0, le=30, coerce=True)
    availability_60: Series[int] = pa.Field(nullable=True, ge=0, le=60, coerce=True)
    availability_90: Series[int] = pa.Field(nullable=True, ge=0, le=90, coerce=True)
    availability_365: Series[int] = pa.Field(nullable=True, ge=0, le=365, coerce=True)
    host_acceptance_rate: Series[float] = pa.Field(
        nullable=True, ge=0, le=100, coerce=True
    )
    host_response_rate: Series[float] = pa.Field(
        nullable=True, ge=0, le=100, coerce=True
    )

    class Config:
        strict = False
