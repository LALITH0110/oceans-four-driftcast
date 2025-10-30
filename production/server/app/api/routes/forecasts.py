"""
Forecast generation and retrieval API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
from datetime import datetime

from app.config.database import get_async_session
from app.models.database import Forecast
from app.auth.manager import auth_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class ForecastRequest(BaseModel):
    forecast_type: str = "plastic_drift"
    time_horizon: int = 24  # hours
    spatial_bounds: Dict[str, Any]  # GeoJSON polygon
    parameters: Dict[str, Any] = {}

class ForecastResponse(BaseModel):
    forecast_id: str
    forecast_type: str
    time_horizon: int
    spatial_bounds: Dict[str, Any]
    confidence_level: float
    created_at: datetime
    metadata: Dict[str, Any]

class ForecastListResponse(BaseModel):
    forecasts: List[ForecastResponse]
    total: int
    page: int
    page_size: int

@router.post("/generate", response_model=ForecastResponse)
async def generate_forecast(
    request: ForecastRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """Generate a new forecast"""
    try:
        # This would trigger the forecast generation process
        # For now, create a placeholder forecast
        
        forecast = Forecast(
            forecast_type=request.forecast_type,
            time_horizon=request.time_horizon,
            spatial_bounds=request.spatial_bounds,
            confidence_level=0.85,  # Would be calculated
            result_data=b"placeholder_forecast_data",  # Would contain actual forecast
            metadata={
                "parameters": request.parameters,
                "generation_method": "distributed_simulation",
                "status": "generating"
            }
        )
        
        session.add(forecast)
        await session.commit()
        await session.refresh(forecast)
        
        logger.info(f"Forecast generation started: {forecast.id}")
        
        return ForecastResponse(
            forecast_id=str(forecast.id),
            forecast_type=forecast.forecast_type,
            time_horizon=forecast.time_horizon,
            spatial_bounds=forecast.spatial_bounds,
            confidence_level=forecast.confidence_level,
            created_at=forecast.created_at,
            metadata=forecast.metadata
        )
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate forecast"
        )

@router.get("/latest", response_model=List[ForecastResponse])
async def get_latest_forecasts(
    forecast_type: Optional[str] = None,
    limit: int = 10,
    session: AsyncSession = Depends(get_async_session)
):
    """Get latest forecasts"""
    try:
        query = select(Forecast).order_by(desc(Forecast.created_at)).limit(limit)
        
        if forecast_type:
            query = query.where(Forecast.forecast_type == forecast_type)
        
        result = await session.execute(query)
        forecasts = result.scalars().all()
        
        return [
            ForecastResponse(
                forecast_id=str(f.id),
                forecast_type=f.forecast_type,
                time_horizon=f.time_horizon,
                spatial_bounds=f.spatial_bounds,
                confidence_level=f.confidence_level,
                created_at=f.created_at,
                metadata=f.metadata or {}
            )
            for f in forecasts
        ]
        
    except Exception as e:
        logger.error(f"Error getting latest forecasts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get latest forecasts"
        )

@router.get("/{forecast_id}", response_model=ForecastResponse)
async def get_forecast(
    forecast_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """Get specific forecast by ID"""
    try:
        result = await session.execute(
            select(Forecast).where(Forecast.id == forecast_id)
        )
        forecast = result.scalar_one_or_none()
        
        if not forecast:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Forecast not found"
            )
        
        return ForecastResponse(
            forecast_id=str(forecast.id),
            forecast_type=forecast.forecast_type,
            time_horizon=forecast.time_horizon,
            spatial_bounds=forecast.spatial_bounds,
            confidence_level=forecast.confidence_level,
            created_at=forecast.created_at,
            metadata=forecast.metadata or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting forecast {forecast_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get forecast"
        )

@router.get("/{forecast_id}/download")
async def download_forecast(
    forecast_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """Download forecast data"""
    try:
        result = await session.execute(
            select(Forecast).where(Forecast.id == forecast_id)
        )
        forecast = result.scalar_one_or_none()
        
        if not forecast:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Forecast not found"
            )
        
        # Return forecast data as binary response
        return Response(
            content=forecast.result_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=forecast_{forecast_id}.bin"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading forecast {forecast_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download forecast"
        )

@router.get("/")
async def list_forecasts(
    page: int = 1,
    page_size: int = 20,
    forecast_type: Optional[str] = None,
    session: AsyncSession = Depends(get_async_session)
) -> ForecastListResponse:
    """List forecasts with pagination"""
    try:
        offset = (page - 1) * page_size
        
        query = select(Forecast).order_by(desc(Forecast.created_at))
        
        if forecast_type:
            query = query.where(Forecast.forecast_type == forecast_type)
        
        # Get total count
        count_result = await session.execute(
            select(Forecast).where(
                Forecast.forecast_type == forecast_type if forecast_type else True
            )
        )
        total = len(count_result.scalars().all())
        
        # Get paginated results
        result = await session.execute(
            query.offset(offset).limit(page_size)
        )
        forecasts = result.scalars().all()
        
        forecast_responses = [
            ForecastResponse(
                forecast_id=str(f.id),
                forecast_type=f.forecast_type,
                time_horizon=f.time_horizon,
                spatial_bounds=f.spatial_bounds,
                confidence_level=f.confidence_level,
                created_at=f.created_at,
                metadata=f.metadata or {}
            )
            for f in forecasts
        ]
        
        return ForecastListResponse(
            forecasts=forecast_responses,
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing forecasts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list forecasts"
        )
