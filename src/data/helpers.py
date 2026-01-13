import json
import re
from typing import Any

import pandas as pd


def format_price(price_str: str) -> float | None:
    if pd.isna(price_str) or price_str == "":
        return None
    formatted = re.sub(r"[^\d.]", "", str(price_str))
    try:
        return float(formatted)
    except ValueError:
        return None


def format_boolean(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    formatted = str(value).lower().strip()
    if formatted in ("t", "true", "1", "yes"):
        return True
    if formatted in ("f", "false", "0", "no"):
        return False
    return None


def format_percentage(percentage_str: Any) -> float | None:
    if pd.isna(percentage_str) or percentage_str == "":
        return None
    formatted = str(percentage_str).replace("%", "").strip()
    try:
        return float(formatted)
    except ValueError:
        return None


def parse_amenities(amenities_str: str) -> list[str]:
    if pd.isna(amenities_str) or amenities_str == "":
        return []
    try:
        amenities_list = json.loads(amenities_str)
        return [str(item).strip() for item in amenities_list if item]
    except (ValueError, TypeError, json.JSONDecodeError):
        return []
