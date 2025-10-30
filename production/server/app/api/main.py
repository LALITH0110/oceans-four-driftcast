"""
FastAPI main application
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn

from app.config.settings import settings
from app.config.database import init_database, close_database
from app.api.routes import clients, tasks, forecasts, admin, simulations
from app.scheduler.task_manager import task_scheduler
from app.monitoring.metrics import setup_metrics
from app.websocket.connection_manager import connection_manager

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Ocean Plastic Forecast API...")
    
    try:
        # Initialize database
        await init_database()
        
        # Start task scheduler
        await task_scheduler.start()
        
        # Create a default test batch for development
        try:
            batch_id = await task_scheduler.create_simulation_batch(
                name="Default Test Batch",
                particle_count=5,
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
                    "auto_created": True
                }
            )
            logger.info(f"Created default test batch: {batch_id}")
        except Exception as e:
            logger.warning(f"Could not create default test batch: {e}")
        
        # Setup metrics
        setup_metrics()
        
        logger.info("Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down application...")
        await close_database()
        logger.info("Application shutdown completed")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Distributed computing platform for ocean plastic drift forecasting",
    lifespan=lifespan
)

# Add CORS middleware - PRODUCTION: Restricted to IIT domain only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://system80.rice.iit.edu",
        "https://www.system80.rice.iit.edu",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(clients.router, prefix="/api/v1/clients", tags=["clients"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["tasks"])
app.include_router(forecasts.router, prefix="/api/v1/forecasts", tags=["forecasts"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(simulations.router, prefix="/api/v1/simulations", tags=["simulations"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Ocean Plastic Drift Forecasting API",
        "version": settings.app_version,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    queue_status = await task_scheduler.get_queue_status()
    return {
        "status": "healthy",
        "timestamp": queue_status["timestamp"],
        "queue_status": queue_status
    }

@app.websocket("/ws/client/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time client communication"""
    await connection_manager.connect(websocket, client_id)
    
    # Look up the real registered client from database
    from app.models.database import Client
    from app.config.database import get_async_session
    from sqlalchemy import select
    from datetime import datetime
    
    # Get the real client from database
    async for session in get_async_session():
        result = await session.execute(
            select(Client).where(Client.id == client_id)
        )
        real_client = result.scalar_one_or_none()
        break
    
    if not real_client:
        logger.error(f"Client {client_id} not found in database")
        await websocket.close(code=4004, reason="Client not registered")
        return
    
    # Update last seen and register with task scheduler
    real_client.last_seen = datetime.utcnow()
    await task_scheduler.register_client(real_client)
    logger.info(f"Registered real client {real_client.name} ({client_id}) with task scheduler")
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get("type")
            
            if message_type == "heartbeat":
                # Refresh in-memory client last_seen to keep scheduler registration alive
                try:
                    from datetime import datetime as _dt
                    real_client.last_seen = _dt.utcnow()
                except Exception:
                    pass
                await connection_manager.send_personal_message(
                    {"type": "heartbeat_ack", "timestamp": data.get("timestamp")},
                    client_id
                )
            
            elif message_type == "task_request":
                # Client requesting new task
                logger.info(f"Received task_request from client {client_id}")
                task_data = await task_scheduler.assign_task(client_id)
                if task_data:
                    logger.info(f"Assigned task {task_data['task_id']} to client {client_id}")
                    await connection_manager.send_personal_message(
                        {"type": "task_assignment", "task": task_data},
                        client_id
                    )
                else:
                    logger.info(f"No tasks available for client {client_id}")
                    await connection_manager.send_personal_message(
                        {"type": "no_tasks_available"},
                        client_id
                    )
            
            elif message_type == "task_completed":
                # Client completed a task
                task_id = data.get("task_id")
                result_data_hex = data.get("result_data", "")
                execution_time = data.get("execution_time", 0)
                
                logger.info(f"Received task_completed for task {task_id} from client {client_id} (execution time: {execution_time}s)")
                
                # Convert hex back to bytes
                try:
                    result_data = bytes.fromhex(result_data_hex)
                except ValueError:
                    logger.warning(f"Could not decode hex result data, using raw encoding")
                    result_data = result_data_hex.encode()
                
                success = await task_scheduler.complete_task(task_id, client_id, result_data)
                
                if success:
                    logger.info(f"✓ Task {task_id} marked as completed successfully")
                else:
                    logger.error(f"✗ Failed to mark task {task_id} as completed")
                
                # Store result in database
                if success:
                    from app.models.database import TaskResult
                    from app.config.database import get_async_session
                    
                    async for session in get_async_session():
                        task_result = TaskResult(
                            task_id=task_id,
                            client_id=client_id,
                            result_data=result_data,
                            execution_time=execution_time,
                            quality_score=1.0
                        )
                        session.add(task_result)
                        await session.commit()
                        logger.info(f"Stored result for task {task_id} in database")
                        break
                
                await connection_manager.send_personal_message(
                    {"type": "task_completion_ack", "task_id": task_id, "success": success},
                    client_id
                )
                logger.debug(f"Sent task_completion_ack to client {client_id}")
            
            elif message_type == "task_failed":
                # Client failed to complete task
                task_id = data.get("task_id")
                error_message = data.get("error", "Unknown error")
                
                await task_scheduler.fail_task(task_id, client_id, error_message)
                
                await connection_manager.send_personal_message(
                    {"type": "task_failure_ack", "task_id": task_id},
                    client_id
                )
            
            else:
                logger.warning(f"Unknown message type from client {client_id}: {message_type}")
                
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        await task_scheduler.unregister_client(client_id)
        logger.info(f"Client {client_id} disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        connection_manager.disconnect(client_id)

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers if not settings.debug else 1,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
