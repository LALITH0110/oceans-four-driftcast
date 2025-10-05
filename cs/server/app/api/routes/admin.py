"""
Admin API endpoints for system management
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Dict, Any, List
from pydantic import BaseModel
import logging
from datetime import datetime, timedelta

from app.config.database import get_async_session
from app.models.database import Client, Task, TaskResult, Forecast, SimulationBatch
from app.scheduler.task_manager import task_scheduler

logger = logging.getLogger(__name__)
router = APIRouter()

class SystemStats(BaseModel):
    total_clients: int
    active_clients: int
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    pending_tasks: int
    total_forecasts: int
    queue_status: Dict[str, Any]

class ClientStats(BaseModel):
    client_id: str
    name: str
    tasks_completed: int
    tasks_failed: int
    success_rate: float
    avg_execution_time: float
    last_seen: datetime

class BatchCreationRequest(BaseModel):
    name: str
    description: str = ""
    particle_count: int
    time_horizon: int
    spatial_bounds: Dict[str, Any]
    parameters: Dict[str, Any] = {}

@router.get("/stats", response_model=SystemStats)
async def get_system_stats(
    session: AsyncSession = Depends(get_async_session)
):
    """Get overall system statistics"""
    try:
        # Get client counts
        total_clients_result = await session.execute(
            select(func.count(Client.id))
        )
        total_clients = total_clients_result.scalar()
        
        active_clients_result = await session.execute(
            select(func.count(Client.id)).where(Client.is_active == True)
        )
        active_clients = active_clients_result.scalar()
        
        # Get task counts
        total_tasks_result = await session.execute(
            select(func.count(Task.id))
        )
        total_tasks = total_tasks_result.scalar()
        
        completed_tasks_result = await session.execute(
            select(func.count(Task.id)).where(Task.status == 'completed')
        )
        completed_tasks = completed_tasks_result.scalar()
        
        failed_tasks_result = await session.execute(
            select(func.count(Task.id)).where(Task.status == 'failed')
        )
        failed_tasks = failed_tasks_result.scalar()
        
        pending_tasks_result = await session.execute(
            select(func.count(Task.id)).where(Task.status == 'pending')
        )
        pending_tasks = pending_tasks_result.scalar()
        
        # Get forecast count
        total_forecasts_result = await session.execute(
            select(func.count(Forecast.id))
        )
        total_forecasts = total_forecasts_result.scalar()
        
        # Get queue status
        queue_status = await task_scheduler.get_queue_status()
        
        return SystemStats(
            total_clients=total_clients or 0,
            active_clients=active_clients or 0,
            total_tasks=total_tasks or 0,
            completed_tasks=completed_tasks or 0,
            failed_tasks=failed_tasks or 0,
            pending_tasks=pending_tasks or 0,
            total_forecasts=total_forecasts or 0,
            queue_status=queue_status
        )
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system statistics"
        )

@router.get("/clients", response_model=List[ClientStats])
async def get_client_stats(
    session: AsyncSession = Depends(get_async_session)
):
    """Get statistics for all clients"""
    try:
        # Get all clients with their task statistics
        clients_result = await session.execute(
            select(Client).where(Client.is_active == True)
        )
        clients = clients_result.scalars().all()
        
        client_stats = []
        
        for client in clients:
            # Get task completion stats for this client
            completed_result = await session.execute(
                select(func.count(TaskResult.id)).where(
                    TaskResult.client_id == client.id
                )
            )
            completed_count = completed_result.scalar() or 0
            
            failed_result = await session.execute(
                select(func.count(Task.id)).where(
                    Task.assigned_client_id == client.id,
                    Task.status == 'failed'
                )
            )
            failed_count = failed_result.scalar() or 0
            
            # Calculate success rate
            total_tasks = completed_count + failed_count
            success_rate = completed_count / total_tasks if total_tasks > 0 else 0.0
            
            # Get average execution time
            avg_time_result = await session.execute(
                select(func.avg(TaskResult.execution_time)).where(
                    TaskResult.client_id == client.id
                )
            )
            avg_execution_time = avg_time_result.scalar() or 0.0
            
            client_stats.append(ClientStats(
                client_id=str(client.id),
                name=client.name,
                tasks_completed=completed_count,
                tasks_failed=failed_count,
                success_rate=success_rate,
                avg_execution_time=avg_execution_time,
                last_seen=client.last_seen
            ))
        
        return client_stats
        
    except Exception as e:
        logger.error(f"Error getting client stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get client statistics"
        )

@router.post("/batch/create")
async def create_simulation_batch(
    request: BatchCreationRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """Create a new simulation batch"""
    try:
        batch_id = await task_scheduler.create_simulation_batch(
            name=request.name,
            particle_count=request.particle_count,
            time_horizon=request.time_horizon,
            spatial_bounds=request.spatial_bounds,
            parameters=request.parameters
        )
        
        return {
            "batch_id": batch_id,
            "status": "created",
            "message": f"Simulation batch '{request.name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating simulation batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create simulation batch"
        )

@router.get("/batches")
async def list_simulation_batches(
    session: AsyncSession = Depends(get_async_session)
):
    """List all simulation batches"""
    try:
        result = await session.execute(
            select(SimulationBatch).order_by(SimulationBatch.created_at.desc())
        )
        batches = result.scalars().all()
        
        return [
            {
                "batch_id": str(batch.id),
                "name": batch.name,
                "description": batch.description,
                "total_tasks": batch.total_tasks,
                "completed_tasks": batch.completed_tasks,
                "status": batch.status,
                "created_at": batch.created_at,
                "completed_at": batch.completed_at
            }
            for batch in batches
        ]
        
    except Exception as e:
        logger.error(f"Error listing simulation batches: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list simulation batches"
        )

@router.get("/health")
async def health_check():
    """Detailed health check for admin"""
    try:
        queue_status = await task_scheduler.get_queue_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "services": {
                "task_scheduler": "running",
                "database": "connected",
                "queue": queue_status
            }
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow(),
            "error": str(e)
        }
