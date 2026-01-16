import json
import logging
from datetime import datetime
from pathlib import Path

from constants import PREDICTION_LOG_DIR
from schemas import Listing

logger = logging.getLogger(__name__)


def log_prediction(
    user_id: str,
    model_name: str,
    input_listings: list[Listing],
    predictions: list[float],
) -> None:
    try:
        logger.info(f"Logging prediction for user {user_id}, model {model_name}, {len(input_listings)} listings")
        PREDICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)

        log_file = PREDICTION_LOG_DIR / "predictions.log"

        timestamp = datetime.now().isoformat()

        for listing, rating in zip(input_listings, predictions, strict=True):
            log_entry = {
                "timestamp": timestamp,
                "user_id": user_id,
                "model_name": model_name,
                "listing_id": listing.id,
                "input_data": listing.model_dump(),
                "prediction": rating,
            }

            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        logger.info(f"Successfully logged {len(input_listings)} predictions for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to log prediction for user {user_id}: {e}", exc_info=True)
