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
from app.api.routes import clients, tasks, forecasts, admin
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(clients.router, prefix="/api/v1/clients", tags=["clients"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["tasks"])
app.include_router(forecasts.router, prefix="/api/v1/forecasts", tags=["forecasts"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])

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
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get("type")
            
            if message_type == "heartbeat":
                await connection_manager.send_personal_message(
                    {"type": "heartbeat_ack", "timestamp": data.get("timestamp")},
                    client_id
                )
            
            elif message_type == "task_request":
                # Client requesting new task
                task_data = await task_scheduler.assign_task(client_id)
                if task_data:
                    await connection_manager.send_personal_message(
                        {"type": "task_assignment", "task": task_data},
                        client_id
                    )
                else:
                    await connection_manager.send_personal_message(
                        {"type": "no_tasks_available"},
                        client_id
                    )
            
            elif message_type == "task_completed":
                # Client completed a task
                task_id = data.get("task_id")
                result_data_hex = data.get("result_data", "")
                execution_time = data.get("execution_time", 0)
                
                # Convert hex back to bytes
                try:
                    result_data = bytes.fromhex(result_data_hex)
                except ValueError:
                    result_data = result_data_hex.encode()
                
                success = await task_scheduler.complete_task(task_id, client_id, result_data)
                
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
                        break
                
                await connection_manager.send_personal_message(
                    {"type": "task_completion_ack", "task_id": task_id, "success": success},
                    client_id
                )
            
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
