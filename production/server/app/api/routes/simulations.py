"""
Simulation batch management API endpoints
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any
import logging
from datetime import datetime

from app.scheduler.task_manager import task_scheduler

logger = logging.getLogger(__name__)
router = APIRouter()

class SimulationBatchRequest(BaseModel):
    name: str
    particle_count: int
    time_horizon: int
    spatial_bounds: Dict[str, float]
    parameters: Dict[str, Any] = {}

class SimulationBatchResponse(BaseModel):
    batch_id: str
    name: str
    total_tasks: int
    created_at: str
    status: str

@router.post("/batch", response_model=SimulationBatchResponse)
async def create_simulation_batch(request: SimulationBatchRequest):
    """Create a new simulation batch"""
    try:
        batch_id = await task_scheduler.create_simulation_batch(
            name=request.name,
            particle_count=request.particle_count,
            time_horizon=request.time_horizon,
            spatial_bounds=request.spatial_bounds,
            parameters=request.parameters
        )
        
        # Calculate total tasks (same logic as in task_scheduler)
        particles_per_task = 1000
        total_tasks = max(1, request.particle_count // particles_per_task)
        
        return SimulationBatchResponse(
            batch_id=batch_id,
            name=request.name,
            total_tasks=total_tasks,
            created_at=datetime.utcnow().isoformat(),
            status="created"
        )
        
    except Exception as e:
        logger.error(f"Error creating simulation batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create simulation batch: {str(e)}"
        )

@router.get("/batch/{batch_id}")
async def get_simulation_batch(batch_id: str):
    """Get simulation batch status"""
    # This would query the database for batch info
    # For now, return basic info
    return {
        "batch_id": batch_id,
        "status": "active",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/test-batch")
async def create_test_batch():
    """Create a test simulation batch for development"""
    try:
        batch_id = await task_scheduler.create_simulation_batch(
            name="Test RL Simulation Batch",
            particle_count=5,  # Small number for testing
            time_horizon=2,
            spatial_bounds={
                "min_lat": 25.0,
                "max_lat": 30.0,
                "min_lon": -95.0,
                "max_lon": -85.0
            },
            parameters={
                "current_strength": 0.5,
                "wind_speed": 10.0,
                "priority": 1,
                "test_mode": True
            }
        )
        
        # Get current queue status
        queue_status = await task_scheduler.get_queue_status()
        
        return {
            "batch_id": batch_id,
            "message": "Test batch created successfully",
            "tasks_created": 1,  # 5 particles = 1 task (since particles_per_task = 10)
            "queue_status": queue_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating test batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create test batch: {str(e)}"
        )
