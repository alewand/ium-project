from pydantic import BaseModel, ConfigDict, Field


class Listing(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    host_response_time: str | None = None
    host_is_superhost: bool | None = None
    host_listings_count: float | None = Field(None, ge=0)
    host_total_listings_count: float | None = Field(None, ge=0)
    host_identity_verified: bool | None = None
    neighbourhood_group_cleansed: str | None = None
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    room_type: str | None = None
    accommodates: int | None = Field(None, ge=1)
    bathrooms: float | None = Field(None, ge=0)
    bedrooms: float | None = Field(None, ge=0)
    beds: float | None = Field(None, ge=0)
    price: float | None = Field(None, gt=0)
    minimum_nights: int | None = Field(None, ge=1)
    maximum_nights: int | None = Field(None, ge=1)
    minimum_minimum_nights: int | None = Field(None, ge=1)
    maximum_minimum_nights: int | None = Field(None, ge=1)
    minimum_maximum_nights: int | None = Field(None, ge=1)
    maximum_maximum_nights: int | None = Field(None, ge=1)
    minimum_nights_avg_ntm: float | None = Field(None, ge=1)
    maximum_nights_avg_ntm: float | None = Field(None, ge=1)
    number_of_reviews: int | None = Field(None, ge=0)
    review_scores_rating: float | None = Field(None, ge=0, le=5)
    has_availability: bool | None = None
    availability_30: int | None = Field(None, ge=0, le=30)
    availability_60: int | None = Field(None, ge=0, le=60)
    availability_90: int | None = Field(None, ge=0, le=90)
    availability_365: int | None = Field(None, ge=0, le=365)
