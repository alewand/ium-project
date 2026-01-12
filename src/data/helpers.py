import re

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
