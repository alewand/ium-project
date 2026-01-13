import shutil

from fastapi import APIRouter, HTTPException, UploadFile

from constants import (
    DEFAULT_MODEL_CONFIG_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRANSFORMER_NAME,
    SERVICE_MODEL_DIR,
)
from service.schemas.schema import (
    AvailableModelsResponse,
    DeleteModelResponse,
    UploadModelResponse,
)
from service.services.model import get_models, validate_model_config

MAX_MODELS = 2

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get("/models", response_model=AvailableModelsResponse)
async def get_available_models() -> AvailableModelsResponse:
    return AvailableModelsResponse(models=[model.name for model in get_models()])


@router.post("/models", response_model=UploadModelResponse)
async def upload_model(
    model_name: str,
    model_file: UploadFile,
    transformer_file: UploadFile,
    model_config_file: UploadFile,
) -> UploadModelResponse:
    if len(get_models()) >= MAX_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Service can handle only {MAX_MODELS} models, remove one.",
        )

    try:
        if not model_file or not transformer_file or not model_config_file:
            raise HTTPException(
                status_code=400,
                detail="Missing required files.",
            )

        if model_file.filename != DEFAULT_MODEL_NAME:
            raise HTTPException(
                status_code=400,
                detail=f"Model file must be named '{DEFAULT_MODEL_NAME}'.",
            )

        if transformer_file.filename != DEFAULT_TRANSFORMER_NAME:
            raise HTTPException(
                status_code=400,
                detail=f"Transformer file must be named '{DEFAULT_TRANSFORMER_NAME}'.",
            )

        if model_config_file.filename != DEFAULT_MODEL_CONFIG_NAME:
            raise HTTPException(
                status_code=400,
                detail=f"Config file must be named '{DEFAULT_MODEL_CONFIG_NAME}'.",
            )

        config_content = await model_config_file.read()
        validate_model_config(config_content)

        model_folder = SERVICE_MODEL_DIR / model_name
        model_folder.mkdir(parents=True, exist_ok=True)

        model_path = model_folder / DEFAULT_MODEL_NAME
        transformer_path = model_folder / DEFAULT_TRANSFORMER_NAME
        config_path = model_folder / DEFAULT_MODEL_CONFIG_NAME

        with model_path.open("wb") as f:
            shutil.copyfileobj(model_file.file, f)

        with transformer_path.open("wb") as f:
            shutil.copyfileobj(transformer_file.file, f)

        with config_path.open("wb") as f:
            f.write(config_content)

        return UploadModelResponse(
            message=f"Model {model_name} uploaded successfully.",
        )

    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/models/{model_name}", response_model=DeleteModelResponse)
async def delete_model(model_name: str) -> DeleteModelResponse:
    SERVICE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        model_folder = SERVICE_MODEL_DIR / model_name

        if not model_folder.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found.",
            )

        shutil.rmtree(model_folder)

        return DeleteModelResponse(
            message=f"Model {model_name} deleted successfully.",
        )

    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
