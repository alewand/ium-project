import argparse

import requests

from constants import DEFAULT_SERVICE_URL, HTTP_OK


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


def delete_model(model_name: str, service_url: str) -> None:
    url = f"{service_url}/api/v1/admin/models/{model_name}"

    response = requests.delete(url, timeout=300.0)

    if response.status_code == HTTP_OK:
        result = response.json()
        print(f"{result.get('message', '')}.")
    else:
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text
        print(f"Error occurred: {error_detail}")


if __name__ == "__main__":
    model_name, service_url = get_arguments()
    print(f"Deleting model '{model_name}' from {service_url}...")
    try:
        delete_model(model_name, service_url)
    except Exception as e:
        print(e)
