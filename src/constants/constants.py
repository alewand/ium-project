from pathlib import Path

_MODEL_DIR = Path(__file__).parent.parent / "model" / "models"
_DATASET_DIR = Path(__file__).parent.parent / "data" / "datasets"

MODEL_DIR = _MODEL_DIR
DEFAULT_MODEL_NAME = "model.pkl"
DEFAULT_TRANSFORMER_NAME = "transformer.pkl"

DATASET_DIR = _DATASET_DIR
DEFAULT_DATASET_NAME = "listings.csv"

SERVICE_CONFIG_PATH = Path("service_config.json")

PRICE_COLUMN = "price"
REVIEWS_AMOUNT_COLUMN = "number_of_reviews"
REVIEW_SCORES_RATING_COLUMN = "review_scores_rating"
PREDICTED_RATING_COLUMN = "predicted_rating"
FINAL_RATING_COLUMN = "final_rating"
AMENITIES_COLUMN = "amenities"
DEFAULT_MIN_REVIEWS = 5

DEFAULT_RANDOM_STATE = 42

BOOLEAN_COLUMNS = ["host_is_superhost", "host_identity_verified"]

PERCENTAGE_COLUMNS = ["host_acceptance_rate", "host_response_rate"]

NUMERIC_COLUMNS = [
    "price",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "latitude",
    "longitude",
    "host_listings_count",
    "host_total_listings_count",
    "host_acceptance_rate",
    "host_response_rate",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
]

CATEGORICAL_COLUMNS = [
    "host_response_time",
    "host_is_superhost",
    "host_identity_verified",
    "room_type",
    "neighbourhood_group_cleansed",
    "has_availability",
]

IMPUTER_STRATEGY = "mean"

COLUMNS_TO_DROP = [
    "listing_url",
    "scrape_id",
    "last_scraped",
    "source",
    "name",
    "description",
    "neighborhood_overview",
    "picture_url",
    "host_url",
    "host_name",
    "host_about",
    "host_thumbnail_url",
    "host_picture_url",
    "bathrooms_text",
    "license",
    "calendar_last_scraped",
    "first_review",
    "last_review",
    "host_verifications",
    "host_location",
    "host_neighbourhood",
    "neighbourhood",
    "neighbourhood_cleansed",
    "property_type",
    "calendar_updated",
    "host_has_profile_pic",
    "host_id",
]
