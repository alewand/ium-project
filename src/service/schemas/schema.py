from pydantic import BaseModel

from schemas import Listing


class RankListingsRequest(BaseModel):
    user_id: str
    listings: list[Listing]


class RankListingsResponse(BaseModel):
    listings: list[Listing]
    ratings: list[float]
    spearman_correlation: float | None


class AvailableModelsResponse(BaseModel):
    models: list[str]


class UploadModelResponse(BaseModel):
    message: str


class DeleteModelResponse(BaseModel):
    message: str
