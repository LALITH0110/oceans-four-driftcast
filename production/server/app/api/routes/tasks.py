"""
Task management API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
from datetime import datetime

from app.config.database import get_async_session
from app.models.database import Task, TaskResult, Client
from app.auth.manager import auth_manager, verify_client_token_dependency
from app.scheduler.task_manager import task_scheduler

logger = logging.getLogger(__name__)
router = APIRouter()

class TaskRequest(BaseModel):
    client_capabilities: Dict[str, Any] = {}

class TaskResponse(BaseModel):
    task_id: str
    simulation_id: str
    particle_count: int
    parameters: Dict[str, Any]
    input_data: str  # hex encoded
    deadline: str
    priority: int

class TaskCompletion(BaseModel):
    task_id: str
    result_data: str  # hex encoded
    execution_time: float
    metadata: Dict[str, Any] = {}

class TaskFailure(BaseModel):
    task_id: str
    error_message: str
    metadata: Dict[str, Any] = {}

@router.get("/available")
async def get_available_task(
    client_id: str = Depends(verify_client_token_dependency),
    session: AsyncSession = Depends(get_async_session)
) -> Optional[TaskResponse]:
    """Get an available task for the client"""
    try:
        # Ensure client is registered with the in-memory scheduler (can be lost after restarts)
        if client_id not in task_scheduler.client_pool:
            result = await session.execute(select(Client).where(Client.id == client_id))
            client = result.scalar_one_or_none()
            if client:
                await task_scheduler.register_client(client)
                logger.info(f"Re-registered client {client_id} with scheduler for task request")

        # Get task from scheduler
        task_data = await task_scheduler.assign_task(client_id)
        
        if not task_data:
            return None
        
        return TaskResponse(**task_data)
        
    except Exception as e:
        logger.error(f"Error getting available task for client {client_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available task"
        )

@router.post("/request")
async def request_task(
    request: TaskRequest,
    client_id: str = Depends(auth_manager.verify_client_token),
    session: AsyncSession = Depends(get_async_session)
) -> Optional[TaskResponse]:
    """Request a new task (alternative to WebSocket)"""
    try:
        # Ensure client is registered with the scheduler
        if client_id not in task_scheduler.client_pool:
            result = await session.execute(select(Client).where(Client.id == client_id))
            client = result.scalar_one_or_none()
            if client:
                await task_scheduler.register_client(client)
                logger.info(f"Re-registered client {client_id} with scheduler for task request")

        # Get task from scheduler
        task_data = await task_scheduler.assign_task(client_id)
        
        if not task_data:
            return None
        
        return TaskResponse(**task_data)
        
    except Exception as e:
        logger.error(f"Error requesting task for client {client_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to request task"
        )

@router.post("/complete")
async def complete_task(
    completion: TaskCompletion,
    client_id: str = Depends(auth_manager.verify_client_token),
    session: AsyncSession = Depends(get_async_session)
):
    """Mark task as completed and submit results"""
    try:
        # Decode result data
        result_data = bytes.fromhex(completion.result_data)
        
        # Complete task through scheduler
        success = await task_scheduler.complete_task(
            completion.task_id,
            client_id,
            result_data
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to complete task"
            )
        
        # Store result in database
        task_result = TaskResult(
            task_id=completion.task_id,
            client_id=client_id,
            result_data=result_data,
            execution_time=completion.execution_time,
            quality_score=1.0  # Would be calculated based on validation
        )
        
        session.add(task_result)
        
        # Update task status
        result = await session.execute(
            select(Task).where(Task.id == completion.task_id)
        )
        task = result.scalar_one_or_none()
        
        if task:
            task.status = 'completed'
            task.completed_at = datetime.utcnow()
        
        await session.commit()
        
        logger.info(f"Task {completion.task_id} completed by client {client_id}")
        
        return {
            "status": "completed",
            "task_id": completion.task_id,
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing task {completion.task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete task"
        )

@router.post("/fail")
async def fail_task(
    failure: TaskFailure,
    client_id: str = Depends(auth_manager.verify_client_token),
    session: AsyncSession = Depends(get_async_session)
):
    """Report task failure"""
    try:
        # Handle failure through scheduler
        success = await task_scheduler.fail_task(
            failure.task_id,
            client_id,
            failure.error_message
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to report task failure"
            )
        
        # Update task status in database
        result = await session.execute(
            select(Task).where(Task.id == failure.task_id)
        )
        task = result.scalar_one_or_none()
        
        if task:
            task.status = 'failed'
            task.retry_count += 1
        
        await session.commit()
        
        logger.info(f"Task {failure.task_id} failed by client {client_id}: {failure.error_message}")
        
        return {
            "status": "failed",
            "task_id": failure.task_id,
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reporting task failure {failure.task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to report task failure"
        )

@router.get("/status/{task_id}")
async def get_task_status(
    task_id: str,
    client_id: str = Depends(auth_manager.verify_client_token),
    session: AsyncSession = Depends(get_async_session)
):
    """Get status of a specific task"""
    try:
        result = await session.execute(
            select(Task).where(Task.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
        
        return {
            "task_id": str(task.id),
            "status": task.status,
            "created_at": task.created_at,
            "assigned_at": task.assigned_at,
            "completed_at": task.completed_at,
            "deadline": task.deadline,
            "retry_count": task.retry_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get task status"
        )

@router.get("/queue/status")
async def get_queue_status():
    """Get current queue status"""
    try:
        status = await task_scheduler.get_queue_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get queue status"
        )

@router.get("/results")
async def get_task_results(
    limit: int = 10,
    session: AsyncSession = Depends(get_async_session)
):
    """Get recent task results"""
    try:
        result = await session.execute(
            select(TaskResult, Task, Client)
            .join(Task, TaskResult.task_id == Task.id)
            .join(Client, TaskResult.client_id == Client.id)
            .order_by(TaskResult.created_at.desc())
            .limit(limit)
        )
        
        results = []
        for task_result, task, client in result:
            # Decode result data from bytes
            try:
                import json
                result_json = json.loads(task_result.result_data.decode('utf-8'))
            except:
                result_json = {"error": "Could not decode result data"}
            
            results.append({
                "result_id": str(task_result.id),
                "task_id": str(task_result.task_id),
                "client_name": client.name,
                "client_id": str(task_result.client_id),
                "execution_time": task_result.execution_time,
                "quality_score": task_result.quality_score,
                "created_at": task_result.created_at,
                "task_status": task.status,
                "result_data": result_json,
                "particle_count": result_json.get("particle_count", 0),
                "user_id": result_json.get("user_id", "unknown"),
                "simulation_type": result_json.get("metadata", {}).get("simulation_type", "unknown")
            })
        
        return {
            "results": results,
            "total_count": len(results),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting task results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get task results"
        )
