from fastapi import APIRouter, FastAPI

from .schema import SortListingsRequest, SortListingsResponse

router = APIRouter(prefix="/api/v1", tags=["api"])


@router.get("/sort-listings")
def sort_listings(request: SortListingsRequest) -> SortListingsResponse:
    return SortListingsResponse(
        listings=sorted(
            request.listings,
            key=lambda x: (
                x.review_scores_rating is None,
                x.review_scores_rating or 0.0,
            ),
        )
    )


app = FastAPI()

app.include_router(router)
