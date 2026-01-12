from pydantic import BaseModel

from schemas import Listing


class SortListingsRequest(BaseModel):
    user_id: str
    listings: list[Listing]


class SortListingsResponse(BaseModel):
    listings: list[Listing]
