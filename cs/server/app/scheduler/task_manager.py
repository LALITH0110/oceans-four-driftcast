"""
Task scheduling and management using Celery
"""
import asyncio
from typing import Dict, List, Optional, Any
from celery import Celery
from datetime import datetime, timedelta
import logging
import json
import uuid
from dataclasses import dataclass, asdict

from app.config.settings import settings
from app.models.database import Task, Client, SimulationBatch
from app.services.load_balancer import LoadBalancer

logger = logging.getLogger(__name__)

# Celery app configuration
celery_app = Celery(
    'ocean_forecast',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=['app.workers.simulation_worker']
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_compression='gzip',
    result_compression='gzip',
)

@dataclass
class WorkUnit:
    id: str
    simulation_id: str
    particle_count: int
    parameters: Dict[str, Any]
    input_data: bytes
    priority: int
    deadline: datetime
    retry_count: int = 0
    assigned_client: Optional[str] = None
    assigned_at: Optional[datetime] = None

class TaskScheduler:
    def __init__(self):
        self.client_pool: Dict[str, Client] = {}
        self.work_queue = asyncio.Queue()
        self.load_balancer = LoadBalancer()
        self.active_tasks: Dict[str, WorkUnit] = {}
        
    async def start(self):
        """Start the task scheduler service"""
        logger.info("Starting task scheduler...")
        
        # Start background tasks
        asyncio.create_task(self._process_work_queue())
        asyncio.create_task(self._monitor_task_timeouts())
        asyncio.create_task(self._update_client_status())
        
        logger.info("Task scheduler started successfully")
    
    async def register_client(self, client: Client) -> bool:
        """Register a new volunteer client"""
        try:
            self.client_pool[str(client.id)] = client
            logger.info(f"Client {client.name} registered successfully")
            return True
        except Exception as e:
            logger.error(f"Error registering client {client.name}: {e}")
            return False
    
    async def unregister_client(self, client_id: str) -> bool:
        """Unregister a volunteer client"""
        try:
            if client_id in self.client_pool:
                client = self.client_pool.pop(client_id)
                logger.info(f"Client {client.name} unregistered")
                return True
            return False
        except Exception as e:
            logger.error(f"Error unregistering client {client_id}: {e}")
            return False
    
    async def create_simulation_batch(self, 
                                    name: str,
                                    particle_count: int,
                                    time_horizon: int,
                                    spatial_bounds: Dict[str, Any],
                                    parameters: Dict[str, Any]) -> str:
        """Create a new simulation batch"""
        try:
            batch_id = str(uuid.uuid4())
            
            # Calculate number of tasks needed
            particles_per_task = 1000  # Configurable
            total_tasks = max(1, particle_count // particles_per_task)
            
            # Create simulation batch record
            batch = SimulationBatch(
                id=batch_id,
                name=name,
                total_tasks=total_tasks,
                parameters={
                    'particle_count': particle_count,
                    'time_horizon': time_horizon,
                    'spatial_bounds': spatial_bounds,
                    **parameters
                }
            )
            
            # Generate individual work units
            for i in range(total_tasks):
                work_unit = WorkUnit(
                    id=str(uuid.uuid4()),
                    simulation_id=batch_id,
                    particle_count=particles_per_task,
                    parameters=parameters,
                    input_data=b"",  # Will be populated with ocean data
                    priority=parameters.get('priority', 0),
                    deadline=datetime.utcnow() + timedelta(hours=time_horizon)
                )
                
                await self.work_queue.put(work_unit)
                self.active_tasks[work_unit.id] = work_unit
            
            logger.info(f"Created simulation batch {name} with {total_tasks} tasks")
            return batch_id
            
        except Exception as e:
            logger.error(f"Error creating simulation batch: {e}")
            raise
    
    async def assign_task(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Assign a task to a specific client"""
        try:
            if client_id not in self.client_pool:
                logger.warning(f"Client {client_id} not found in pool")
                return None
            
            client = self.client_pool[client_id]
            
            # Check if client can handle more tasks
            client_task_count = await self._get_client_active_tasks(client_id)
            if client_task_count >= settings.max_tasks_per_client:
                logger.info(f"Client {client_id} has reached max task limit")
                return None
            
            # Get next available task
            if self.work_queue.empty():
                logger.info("No tasks available in queue")
                return None
            
            work_unit = await self.work_queue.get()
            
            # Prepare task data for client
            task_data = {
                "task_id": work_unit.id,
                "simulation_id": work_unit.simulation_id,
                "particle_count": work_unit.particle_count,
                "parameters": work_unit.parameters,
                "input_data": work_unit.input_data.hex(),  # Convert bytes to hex
                "deadline": work_unit.deadline.isoformat(),
                "priority": work_unit.priority
            }
            
            # Mark task as assigned to this client (don't use Celery for direct client communication)
            work_unit.assigned_client = client_id
            work_unit.assigned_at = datetime.utcnow()
            
            logger.info(f"Assigned task {work_unit.id} to client {client_id}")
            return task_data
            
        except Exception as e:
            logger.error(f"Error assigning task to client {client_id}: {e}")
            return None
    
    async def complete_task(self, task_id: str, client_id: str, result_data: bytes) -> bool:
        """Mark task as completed and process results"""
        try:
            if task_id not in self.active_tasks:
                logger.warning(f"Task {task_id} not found in active tasks")
                return False
            
            work_unit = self.active_tasks.pop(task_id)
            
            # Process and validate results
            await self._process_task_results(work_unit, client_id, result_data)
            
            logger.info(f"Task {task_id} completed by client {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error completing task {task_id}: {e}")
            return False
    
    async def fail_task(self, task_id: str, client_id: str, error_message: str) -> bool:
        """Handle task failure and retry if needed"""
        try:
            if task_id not in self.active_tasks:
                logger.warning(f"Task {task_id} not found in active tasks")
                return False
            
            work_unit = self.active_tasks[task_id]
            work_unit.retry_count += 1
            
            if work_unit.retry_count < settings.max_retry_count:
                # Retry task
                await self.work_queue.put(work_unit)
                logger.info(f"Retrying task {task_id} (attempt {work_unit.retry_count})")
            else:
                # Max retries reached, mark as failed
                self.active_tasks.pop(task_id)
                logger.error(f"Task {task_id} failed after {work_unit.retry_count} attempts")
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling task failure {task_id}: {e}")
            return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "pending_tasks": self.work_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "active_clients": len(self.client_pool),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _process_work_queue(self):
        """Background task to process work queue"""
        while True:
            try:
                # Process queue every 5 seconds
                await asyncio.sleep(5)
                
                # Check for available clients and tasks
                if not self.work_queue.empty() and self.client_pool:
                    available_clients = [
                        client_id for client_id in self.client_pool.keys()
                        if await self._get_client_active_tasks(client_id) < settings.max_tasks_per_client
                    ]
                    
                    if available_clients:
                        # Use load balancer to select client
                        selected_client = self.load_balancer.select_client(
                            available_clients, {}
                        )
                        await self.assign_task(selected_client)
                        
            except Exception as e:
                logger.error(f"Error in work queue processor: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_task_timeouts(self):
        """Monitor for task timeouts and handle them"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow()
                timed_out_tasks = []
                
                for task_id, work_unit in self.active_tasks.items():
                    if current_time > work_unit.deadline:
                        timed_out_tasks.append(task_id)
                
                for task_id in timed_out_tasks:
                    await self.fail_task(task_id, "system", "Task timeout")
                    
            except Exception as e:
                logger.error(f"Error monitoring task timeouts: {e}")
                await asyncio.sleep(60)
    
    async def _update_client_status(self):
        """Update client last seen timestamps"""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                # Update client last seen times
                current_time = datetime.utcnow()
                inactive_clients = []
                
                for client_id, client in self.client_pool.items():
                    if (current_time - client.last_seen).seconds > 300:  # 5 minutes
                        inactive_clients.append(client_id)
                
                for client_id in inactive_clients:
                    await self.unregister_client(client_id)
                    
            except Exception as e:
                logger.error(f"Error updating client status: {e}")
                await asyncio.sleep(60)
    
    async def _get_client_active_tasks(self, client_id: str) -> int:
        """Get number of active tasks for a client"""
        # This would query the database for active tasks
        # For now, return 0 as placeholder
        return 0
    
    async def _process_task_results(self, work_unit: WorkUnit, client_id: str, result_data: bytes):
        """Process and store task results"""
        # This would validate and store results in database
        # For now, just log
        logger.info(f"Processing results for task {work_unit.id} from client {client_id}")

# Global task scheduler instance
task_scheduler = TaskScheduler()
