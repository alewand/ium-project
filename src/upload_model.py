import argparse

import requests

from constants import (
    DEFAULT_MODEL_CONFIG_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_SERVICE_URL,
    DEFAULT_TRANSFORMER_NAME,
    HTTP_OK,
    MODEL_DIR,
)


def get_arguments() -> tuple[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--service-url",
        type=str,
        default=DEFAULT_SERVICE_URL,
    )

    arguments = parser.parse_args()

    return arguments.model_name, arguments.service_url


def upload_model(model_name: str, service_url: str) -> None:
    model_folder = MODEL_DIR / model_name

    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}.")

    model_path = model_folder / DEFAULT_MODEL_NAME
    transformer_path = model_folder / DEFAULT_TRANSFORMER_NAME
    config_path = model_folder / DEFAULT_MODEL_CONFIG_NAME

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}.")
    if not transformer_path.exists():
        raise FileNotFoundError(f"Transformer file not found: {transformer_path}.")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}.")

    upload_url = f"{service_url}/api/v1/admin/models"

    with (
        model_path.open("rb") as model_file,
        transformer_path.open("rb") as transformer_file,
        config_path.open("rb") as config_file,
    ):
        files = {
            "model_file": ("model.pkl", model_file, "application/octet-stream"),
            "transformer_file": (
                "transformer.pkl",
                transformer_file,
                "application/octet-stream",
            ),
            "model_config_file": ("model_config.json", config_file, "application/json"),
        }
        params = {"model_name": model_name}

        response = requests.post(upload_url, files=files, params=params, timeout=300.0)

        if response.status_code == HTTP_OK:
            result = response.json()
            print(f"Model uploaded successfully: {result.get('message', '')}.")
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text
            print(f"Error occurred: {error_detail}")


if __name__ == "__main__":
    model_name, service_url = get_arguments()
    print(f"Uploading model '{model_name}' to {service_url}...")
    try:
        upload_model(model_name, service_url)
    except Exception as e:
        print(e)
