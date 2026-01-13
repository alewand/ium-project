from fastapi import APIRouter, HTTPException

from model.predict import predict
from service.schemas.schema import (
    RankListingsRequest,
    RankListingsResponse,
)
from service.services.listings import dataframe_to_listings, listings_to_dataframe
from service.services.model import load_model

router = APIRouter(prefix="/api/v1", tags=["listings"])


@router.post("/rank-listings")
def rank_listings(request: RankListingsRequest) -> RankListingsResponse:
    try:
        listings_dataframe = listings_to_dataframe(request.listings)
        model, transformer, min_reviews, rating_weight = load_model()

        ranked_listings_dataframe = predict(
            listings_dataframe,
            model,
            transformer,
            min_reviews=min_reviews,
            rating_weight=rating_weight,
        )

        ranked_listings = dataframe_to_listings(ranked_listings_dataframe)

        return RankListingsResponse(listings=ranked_listings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
