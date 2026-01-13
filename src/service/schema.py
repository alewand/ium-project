from pydantic import BaseModel

from schemas import Listing


class RankListingsRequest(BaseModel):
    user_id: str
    listings: list[Listing]


class RankListingsResponse(BaseModel):
    listings: list[Listing]


class ConfigUpdateRequest(BaseModel):
    model_name: str
    transformer_name: str


class ConfigUpdateResponse(BaseModel):
    message: str
