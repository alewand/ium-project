import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from scipy.stats import spearmanr

from constants import (
    MIN_PAIRS_FOR_CORRELATION,
    REVIEW_SCORES_RATING_COLUMN,
)
from model.predict import predict
from service.schemas.schema import (
    RankListingsRequest,
    RankListingsResponse,
)
from service.services.listings import dataframe_to_listings, listings_to_dataframe
from service.services.model import load_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["listings"])


@router.post("/rank-listings")
def rank_listings(request: RankListingsRequest) -> RankListingsResponse:
    try:
        listings_dataframe = listings_to_dataframe(request.listings)
        model, transformer, min_reviews, rating_weight, model_name = load_model(
            request.user_id
        )
        logger.info(f"Model chosen for {request.user_id}: {model_name}")

        original_ratings = listings_dataframe[REVIEW_SCORES_RATING_COLUMN].tolist()
        original_ids = listings_dataframe["id"].tolist()

        ranked_listings_dataframe, sorted_ratings = predict(
            listings_dataframe,
            model,
            transformer,
            min_reviews=min_reviews,
            rating_weight=rating_weight,
        )

        ranked_ids = ranked_listings_dataframe["id"].tolist()

        id_to_final_rating = dict(zip(ranked_ids, sorted_ratings, strict=True))

        final_ratings_original_order = [
            id_to_final_rating[listing_id] for listing_id in original_ids
        ]

        valid_pairs = [
            (original, final)
            for original, final in zip(
                original_ratings, final_ratings_original_order, strict=True
            )
            if original is not None
            and not pd.isna(original)
            and final is not None
            and not pd.isna(final)
        ]

        if len(valid_pairs) < MIN_PAIRS_FOR_CORRELATION:
            spearman_correlation = None
        else:
            valid_original, valid_final = zip(*valid_pairs, strict=True)
            spearman_correlation, _ = spearmanr(valid_original, valid_final)

        ranked_listings = dataframe_to_listings(ranked_listings_dataframe)

        return RankListingsResponse(
            ratings=sorted_ratings,
            listings=ranked_listings,
            spearman_correlation=spearman_correlation,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
