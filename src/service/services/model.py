import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
from fastapi import HTTPException
from sklearn.compose import ColumnTransformer

from constants import (
    DEFAULT_MODEL_CONFIG_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRANSFORMER_NAME,
    MIN_REVIEWS_KEY,
    RATING_WEIGHT_KEY,
    SERVICE_MODEL_DIR,
)


def validate_model_config(config_content: bytes) -> dict:
    try:
        config = json.loads(config_content.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in config file: {e}.",
        ) from e

    if not isinstance(config, dict):
        raise HTTPException(
            status_code=400,
            detail="Config file must contain a JSON object.",
        )

    if MIN_REVIEWS_KEY not in config:
        raise HTTPException(
            status_code=400,
            detail=f"Config file must contain '{MIN_REVIEWS_KEY}' field.",
        )

    if RATING_WEIGHT_KEY not in config:
        raise HTTPException(
            status_code=400,
            detail=f"Config file must contain '{RATING_WEIGHT_KEY}' field.",
        )

    if not isinstance(config[MIN_REVIEWS_KEY], int) or config[MIN_REVIEWS_KEY] < 0:
        raise HTTPException(
            status_code=400,
            detail=f"'{MIN_REVIEWS_KEY}' must be a non-negative integer.",
        )

    if (
        not isinstance(config[RATING_WEIGHT_KEY], (int, float))
        or config[RATING_WEIGHT_KEY] < 0
    ):
        raise HTTPException(
            status_code=400,
            detail=f"'{RATING_WEIGHT_KEY}' must be a non-negative number.",
        )

    return config


def get_models() -> list[Path]:
    SERVICE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    available_models = []

    for model_folder in SERVICE_MODEL_DIR.iterdir():
        if model_folder.is_dir():
            model_path = model_folder / DEFAULT_MODEL_NAME
            transformer_path = model_folder / DEFAULT_TRANSFORMER_NAME
            config_path = model_folder / DEFAULT_MODEL_CONFIG_NAME

            if (
                model_path.exists()
                and transformer_path.exists()
                and config_path.exists()
            ):
                available_models.append(model_folder)

    return sorted(available_models)


def get_model_for_user(user_id: str, available_models: list[Path]) -> Path:
    hash_value = int(hashlib.md5(user_id.encode("utf-8")).hexdigest(), 16)
    model_index = hash_value % len(available_models)
    return available_models[model_index]


def load_model(user_id: str) -> tuple[Any, ColumnTransformer, int, float, str]:
    available_models = get_models()

    if len(available_models) == 0:
        raise HTTPException(
            status_code=404,
            detail="Models not found. Please upload at least one model first.",
        )

    model_folder = get_model_for_user(user_id, available_models)

    model_path = model_folder / DEFAULT_MODEL_NAME
    transformer_path = model_folder / DEFAULT_TRANSFORMER_NAME
    config_path = model_folder / DEFAULT_MODEL_CONFIG_NAME

    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)

    with config_path.open() as f:
        config = json.load(f)

    return (
        model,
        transformer,
        config[MIN_REVIEWS_KEY],
        config[RATING_WEIGHT_KEY],
        model_folder.name,
    )
