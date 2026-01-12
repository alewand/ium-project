from fastapi import APIRouter, FastAPI

router = APIRouter(prefix="/api/v1", tags=["api"])


@router.get("/sort-listings")
def sort_listings() -> dict[str, str]:
    return {"message": "hello world"}


app = FastAPI()

app.include_router(router)
