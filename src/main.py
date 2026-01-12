from data import get_listings

if __name__ == "__main__":
    listings = get_listings("src/data/datasets/listings.csv")
    print(len(listings))

    high_reviewed = listings[listings["number_of_reviews"] >= 5]
    print(f"Listings with 5 or more reviews: {len(high_reviewed)}")
