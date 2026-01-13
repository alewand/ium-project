from fastapi import APIRouter

from .admin import router as admin_router
from .listings import router as listings_router

router = APIRouter()

router.include_router(admin_router)
router.include_router(listings_router)

__all__ = ["router"]
