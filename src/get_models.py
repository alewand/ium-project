import argparse

import requests

from constants import DEFAULT_SERVICE_URL, HTTP_OK


def get_arguments() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service-url",
        type=str,
        default=DEFAULT_SERVICE_URL,
    )

    arguments = parser.parse_args()

    return str(arguments.service_url)


def get_models(service_url: str) -> None:
    url = f"{service_url}/api/v1/admin/models"

    response = requests.get(url, timeout=300.0)

    if response.status_code == HTTP_OK:
        result = response.json()
        models = result.get("models", [])

        if len(models) == 0:
            print("No models found.")
        else:
            print(f"Uploaded models ({len(models)}):")
            for model in models:
                print(f"  - {model}")
    else:
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text
        print(f"Error occurred: {error_detail}")


if __name__ == "__main__":
    service_url = get_arguments()
    get_models(service_url)
