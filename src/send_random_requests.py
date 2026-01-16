import argparse
import random
import string

import requests

from constants import DEFAULT_SERVICE_URL, HTTP_OK
from data import get_listings
from schemas import Listing
from service.services.listings import dataframe_to_listings


def generate_random_user_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=10))


def load_listings_from_csv() -> list[Listing]:
    listings_dataframe = get_listings()
    return dataframe_to_listings(listings_dataframe)


def get_random_listings_from_csv(
    all_listings: list[Listing], num_listings: int
) -> list[Listing]:
    selected_listings = random.sample(all_listings, min(num_listings, len(all_listings)))

    modified_listings = []
    for listing in selected_listings:
        listing_dict = listing.model_dump()
        listing_dict["number_of_reviews"] = random.randint(0, 500)
        modified_listing = Listing.model_validate(listing_dict)
        modified_listings.append(modified_listing)

    return modified_listings


def send_request(
    service_url: str, user_id: str, all_listings: list[Listing], num_listings: int
) -> bool:
    url = f"{service_url}/api/v1/rank-listings"

    listings = get_random_listings_from_csv(all_listings, num_listings)

    payload = {"user_id": user_id, "listings": [l.model_dump() for l in listings]}

    try:
        response = requests.post(url, json=payload, timeout=30.0)

        if response.status_code == HTTP_OK:
            result = response.json()
            print(
                f"✓ Request for user {user_id}: {len(result.get('listings', []))} listings ranked, "
                f"correlation: {result.get('spearman_correlation', 'N/A')}"
            )
            return True
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text
            print(f"✗ Request for user {user_id} failed: {error_detail}")
            return False
    except Exception as e:
        print(f"✗ Request for user {user_id} failed with exception: {e}")
        return False


def get_arguments() -> tuple[int, str]:
    parser = argparse.ArgumentParser(
        description="Send random requests to the ranking service"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of requests to send (default: 10)",
    )
    parser.add_argument(
        "--service-url",
        type=str,
        default=DEFAULT_SERVICE_URL,
        help=f"Service URL (default: {DEFAULT_SERVICE_URL})",
    )

    arguments = parser.parse_args()

    return arguments.num_requests, arguments.service_url


if __name__ == "__main__":
    num_requests, service_url = get_arguments()

    print("Loading listings from CSV...")
    all_listings = load_listings_from_csv()
    print(f"Loaded {len(all_listings)} listings from CSV")

    print(f"Sending {num_requests} requests to {service_url}")
    print("Number of listings per request will be random (1-100)")
    print("=" * 60)

    successful = 0
    failed = 0

    for i in range(num_requests):
        user_id = generate_random_user_id()
        num_listings = random.randint(1, 100)
        print(
            f"[{i+1}/{num_requests}] Sending request for user {user_id} with {num_listings} listings"
        )
        if send_request(service_url, user_id, all_listings, num_listings):
            successful += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"Summary: {successful} successful, {failed} failed")
