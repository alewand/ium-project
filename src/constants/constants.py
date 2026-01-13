from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "data" / "datasets"
MODEL_DIR = Path(__file__).parent.parent / "model" / "models"
SERVICE_MODEL_DIR = Path(__file__).parent.parent / "service" / "models"

DEFAULT_DATASET_NAME = "listings.csv"

DEFAULT_MODEL_NAME = "model.pkl"
DEFAULT_TRANSFORMER_NAME = "transformer.pkl"
DEFAULT_MODEL_CONFIG_NAME = "model_config.json"

DEFAULT_MIN_REVIEWS = 5
DEFAULT_RANDOM_STATE = 42
IMPUTER_STRATEGY = "mean"

HTTP_OK = 200

DEFAULT_SERVICE_URL = "http://localhost:8000"

MAX_MODELS = 2

MIN_PAIRS_FOR_CORRELATION = 2

DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = None
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MIN_SAMPLES_LEAF = 1
DEFAULT_MAX_FEATURES = "sqrt"
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_SUBSAMPLE = 1.0

MIN_REVIEWS_KEY = "min_reviews"
RATING_WEIGHT_KEY = "rating_weight"

PRICE_COLUMN = "price"
REVIEWS_AMOUNT_COLUMN = "number_of_reviews"
REVIEW_SCORES_RATING_COLUMN = "review_scores_rating"
PREDICTED_RATING_COLUMN = "predicted_rating"
FINAL_RATING_COLUMN = "final_rating"
AMENITIES_COLUMN = "amenities"

BOOLEAN_COLUMNS = ["host_is_superhost", "host_identity_verified"]

PERCENTAGE_COLUMNS = ["host_acceptance_rate", "host_response_rate"]

NULLABLE_INT_COLUMNS = [
    "accommodates",
    "minimum_nights",
    "maximum_nights",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    REVIEWS_AMOUNT_COLUMN,
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
]

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
