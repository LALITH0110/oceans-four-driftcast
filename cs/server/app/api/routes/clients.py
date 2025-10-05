"""
Client registration and management API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, Any, List
from pydantic import BaseModel
import logging
from datetime import datetime

from app.config.database import get_async_session
from app.models.database import Client
from app.auth.manager import auth_manager
from app.scheduler.task_manager import task_scheduler

logger = logging.getLogger(__name__)
router = APIRouter()

class ClientRegistration(BaseModel):
    name: str
    public_key: str
    capabilities: Dict[str, Any]
    system_info: Dict[str, Any] = {}

class ClientResponse(BaseModel):
    client_id: str
    name: str
    capabilities: Dict[str, Any]
    created_at: datetime
    is_active: bool

class TokenResponse(BaseModel):
    client_id: str
    token: str
    expires_in: int

@router.post("/register", response_model=TokenResponse)
async def register_client(
    registration: ClientRegistration,
    session: AsyncSession = Depends(get_async_session)
):
    """Register a new volunteer client"""
    try:
        # Create new client record
        client = Client(
            name=registration.name,
            public_key=registration.public_key,
            capabilities=registration.capabilities,
            is_active=True,
            trust_level=1
        )
        
        session.add(client)
        await session.commit()
        await session.refresh(client)
        
        # Register with task scheduler
        await task_scheduler.register_client(client)
        
        # Generate access token
        token = auth_manager.generate_client_token(
            str(client.id), 
            registration.capabilities
        )
        
        logger.info(f"Client {registration.name} registered successfully")
        
        return TokenResponse(
            client_id=str(client.id),
            token=token,
            expires_in=auth_manager.expire_hours * 3600
        )
        
    except Exception as e:
        logger.error(f"Error registering client: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register client"
        )

@router.get("/me", response_model=ClientResponse)
async def get_client_info(
    client_id: str = Depends(auth_manager.verify_client_token),
    session: AsyncSession = Depends(get_async_session)
):
    """Get current client information"""
    try:
        result = await session.execute(
            select(Client).where(Client.id == client_id)
        )
        client = result.scalar_one_or_none()
        
        if not client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found"
            )
        
        return ClientResponse(
            client_id=str(client.id),
            name=client.name,
            capabilities=client.capabilities,
            created_at=client.created_at,
            is_active=client.is_active
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting client info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get client information"
        )

@router.put("/heartbeat")
async def client_heartbeat(
    client_id: str = Depends(auth_manager.verify_client_token),
    session: AsyncSession = Depends(get_async_session)
):
    """Update client last seen timestamp"""
    try:
        result = await session.execute(
            select(Client).where(Client.id == client_id)
        )
        client = result.scalar_one_or_none()
        
        if not client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found"
            )
        
        client.last_seen = datetime.utcnow()
        await session.commit()
        
        return {"status": "ok", "timestamp": client.last_seen}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating client heartbeat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update heartbeat"
        )

@router.delete("/unregister")
async def unregister_client(
    client_id: str = Depends(auth_manager.verify_client_token),
    session: AsyncSession = Depends(get_async_session)
):
    """Unregister client"""
    try:
        result = await session.execute(
            select(Client).where(Client.id == client_id)
        )
        client = result.scalar_one_or_none()
        
        if not client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found"
            )
        
        # Mark as inactive instead of deleting
        client.is_active = False
        await session.commit()
        
        # Unregister from task scheduler
        await task_scheduler.unregister_client(client_id)
        
        logger.info(f"Client {client.name} unregistered")
        
        return {"status": "unregistered"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering client: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unregister client"
        )

@router.get("/stats")
async def get_client_stats(
    client_id: str = Depends(auth_manager.verify_client_token),
    session: AsyncSession = Depends(get_async_session)
):
    """Get client statistics"""
    try:
        # This would query task completion stats
        # For now, return placeholder data
        return {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0,
            "average_execution_time": 0,
            "success_rate": 0.0
        }
        
    except Exception as e:
        logger.error(f"Error getting client stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get client statistics"
        )
