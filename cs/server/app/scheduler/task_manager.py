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
        
        # Task management configuration
        self.target_pending_tasks = 10  # Target number of pending tasks
        self.min_pending_tasks = 5     # Minimum before creating new batch
        self.max_tasks_per_client = 5  # Maximum concurrent tasks per client
        
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
                
                # Return any active tasks back to the queue
                tasks_to_requeue = []
                for task_id, work_unit in list(self.active_tasks.items()):
                    if work_unit.assigned_client == client_id:
                        tasks_to_requeue.append(task_id)
                
                for task_id in tasks_to_requeue:
                    work_unit = self.active_tasks.pop(task_id)
                    work_unit.assigned_client = None
                    work_unit.assigned_at = None
                    await self.work_queue.put(work_unit)
                    logger.info(f"Requeued task {task_id} from disconnected client {client_id}")
                
                logger.info(f"Client {client.name} unregistered, requeued {len(tasks_to_requeue)} tasks")
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
            particles_per_task = 10  # Smaller for testing, configurable
            total_tasks = max(1, (particle_count + particles_per_task - 1) // particles_per_task)  # Ceiling division
            
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
                # Calculate particles for this task
                remaining_particles = particle_count - (i * particles_per_task)
                task_particles = min(particles_per_task, remaining_particles)
                
                work_unit = WorkUnit(
                    id=str(uuid.uuid4()),
                    simulation_id=batch_id,
                    particle_count=task_particles,
                    parameters=parameters,
                    input_data=b"",  # Will be populated with ocean data
                    priority=parameters.get('priority', 0),
                    deadline=datetime.utcnow() + timedelta(hours=time_horizon)
                )
                
                await self.work_queue.put(work_unit)
                # Note: Don't add to active_tasks here - only add when assigned to client
            
            logger.info(f"Created simulation batch {name} with {total_tasks} tasks")
            return batch_id
            
        except Exception as e:
            logger.error(f"Error creating simulation batch: {e}")
            raise
    
    async def assign_task(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Assign a task to a specific client"""
        try:
            logger.info(f"Attempting to assign task to client {client_id}")
            
            if client_id not in self.client_pool:
                logger.warning(f"Client {client_id} not found in pool. Available clients: {list(self.client_pool.keys())}")
                return None
            
            client = self.client_pool[client_id]
            
            # Check if client can handle more tasks
            client_task_count = await self._get_client_active_tasks(client_id)
            
            logger.info(f"Client {client_id} has {client_task_count}/{self.max_tasks_per_client} active tasks (total active: {len(self.active_tasks)})")
            
            if client_task_count >= self.max_tasks_per_client:
                # Attempt to free up stale tasks assigned to this client
                freed = 0
                now = datetime.utcnow()
                stale_threshold_seconds = 120  # 2 minutes without completion => requeue
                for task_id, work_unit in list(self.active_tasks.items()):
                    if work_unit.assigned_client == client_id and work_unit.assigned_at:
                        age = (now - work_unit.assigned_at).total_seconds()
                        if age > stale_threshold_seconds:
                            # Requeue as stale
                            work_unit.assigned_client = None
                            work_unit.assigned_at = None
                            await self.work_queue.put(work_unit)
                            self.active_tasks.pop(task_id, None)
                            freed += 1
                            logger.warning(f"Requeued stale task {task_id} for client {client_id} (age: {int(age)}s)")
                if freed > 0:
                    logger.info(f"Freed {freed} stale tasks for client {client_id}")
                    client_task_count = await self._get_client_active_tasks(client_id)
                if client_task_count >= self.max_tasks_per_client:
                    logger.warning(f"Client {client_id} at max task limit after cleanup ({client_task_count}/{self.max_tasks_per_client})")
                    logger.info(f"Active tasks for this client: {[tid for tid, wu in self.active_tasks.items() if wu.assigned_client == client_id]}")
                    return None
            
            # Get next available task
            queue_size = self.work_queue.qsize()
            logger.info(f"Queue has {queue_size} pending tasks available")
            
            if self.work_queue.empty():
                logger.info("No tasks available in queue, triggering batch creation")
                # Prefer synchronous creation to reduce client idle time
                await self._check_and_create_tasks()
                # If tasks were created, retrieve one immediately
                if self.work_queue.empty():
                    logger.info("Task queue still empty after creation attempt")
                    return None
            
            work_unit = await self.work_queue.get()
            logger.info(f"Retrieved task {work_unit.id} from queue (remaining in queue: {self.work_queue.qsize()})")
            
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
            
            # Add to active tasks tracking
            self.active_tasks[work_unit.id] = work_unit
            
            logger.info(f"Successfully assigned task {work_unit.id} to client {client_id}. Active tasks now: {len(self.active_tasks)} (queue: {self.work_queue.qsize()})")
            return task_data
            
        except Exception as e:
            logger.error(f"Error assigning task to client {client_id}: {e}")
            return None
    
    async def complete_task(self, task_id: str, client_id: str, result_data: bytes) -> bool:
        """Mark task as completed and process results"""
        try:
            if task_id not in self.active_tasks:
                logger.warning(f"Task {task_id} not found in active tasks (client {client_id})")
                logger.info(f"Active tasks: {list(self.active_tasks.keys())}")
                return False
            
            work_unit = self.active_tasks.pop(task_id)
            
            # Verify the task was assigned to this client
            if work_unit.assigned_client != client_id:
                logger.warning(f"Task {task_id} was assigned to {work_unit.assigned_client}, not {client_id}")
                # Still mark as complete since we got a result
            
            # Process and validate results
            await self._process_task_results(work_unit, client_id, result_data)
            
            logger.info(f"Task {task_id} completed by client {client_id}. Active tasks now: {len(self.active_tasks)}")
            
            # Trigger immediate queue check to create new tasks if needed
            asyncio.create_task(self._check_and_create_tasks())
            
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
            
            if work_unit.retry_count < getattr(settings, 'max_retry_count', 3):
                # Retry task - reset assignment info
                work_unit.assigned_client = None
                work_unit.assigned_at = None
                await self.work_queue.put(work_unit)
                self.active_tasks.pop(task_id)  # Remove from active tasks
                logger.info(f"Retrying task {task_id} (attempt {work_unit.retry_count})")
            else:
                # Max retries reached, mark as failed
                self.active_tasks.pop(task_id)
                logger.error(f"Task {task_id} failed after {work_unit.retry_count} attempts")
            
            # Trigger immediate queue check to create new tasks if needed
            asyncio.create_task(self._check_and_create_tasks())
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling task failure {task_id}: {e}")
            return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        queue_size = self.work_queue.qsize()
        active_tasks = len(self.active_tasks)
        active_clients = len(self.client_pool)
        total_client_capacity = active_clients * self.max_tasks_per_client
        
        return {
            "pending_tasks": queue_size,
            "active_tasks": active_tasks,
            "active_clients": active_clients,
            "total_client_capacity": total_client_capacity,
            "target_pending_tasks": self.target_pending_tasks,
            "min_pending_tasks": self.min_pending_tasks,
            "max_tasks_per_client": self.max_tasks_per_client,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def clear_stuck_tasks(self) -> int:
        """Clear all stuck tasks and return them to queue (for debugging)"""
        stuck_count = len(self.active_tasks)
        for work_unit in list(self.active_tasks.values()):
            work_unit.assigned_client = None
            work_unit.assigned_at = None
            await self.work_queue.put(work_unit)
        
        self.active_tasks.clear()
        logger.info(f"Cleared {stuck_count} stuck tasks back to queue")
        return stuck_count
    
    def update_task_config(self, target_pending: int = None, min_pending: int = None, max_per_client: int = None):
        """Update task management configuration"""
        if target_pending is not None:
            self.target_pending_tasks = max(1, target_pending)
            logger.info(f"Updated target_pending_tasks to {self.target_pending_tasks}")
        
        if min_pending is not None:
            self.min_pending_tasks = max(1, min_pending)
            logger.info(f"Updated min_pending_tasks to {self.min_pending_tasks}")
        
        if max_per_client is not None:
            self.max_tasks_per_client = max(1, max_per_client)
            logger.info(f"Updated max_tasks_per_client to {self.max_tasks_per_client}")
        
        # Trigger immediate check after config change
        asyncio.create_task(self._check_and_create_tasks())
    
    async def _process_work_queue(self):
        """Background task to process work queue"""
        while True:
            try:
                # Process queue every 5 seconds
                await asyncio.sleep(5)
                
                # Check and create tasks if needed
                await self._check_and_create_tasks()
                        
            except Exception as e:
                logger.error(f"Error in work queue processor: {e}")
                await asyncio.sleep(10)
    
    async def _check_and_create_tasks(self):
        """Check if we need to create new tasks and create them"""
        try:
            queue_size = self.work_queue.qsize()
            active_tasks = len(self.active_tasks)
            active_clients = len(self.client_pool)
            
            # Calculate total client capacity
            total_client_capacity = active_clients * self.max_tasks_per_client
            
            logger.info(f"Queue check - Pending: {queue_size}, Active: {active_tasks}, Clients: {active_clients}, Capacity: {total_client_capacity}")
            
            # Create new tasks if we're below the minimum pending tasks and have clients
            if queue_size < self.min_pending_tasks and active_clients > 0:
                # Calculate how many tasks to create to reach target
                tasks_to_create = self.target_pending_tasks - queue_size
                
                if tasks_to_create > 0:
                    logger.info(f"Queue below minimum ({queue_size} < {self.min_pending_tasks}). Creating {tasks_to_create} new tasks to reach target {self.target_pending_tasks}")
                    
                    # Create batch - particles_per_task is 10, so we multiply by 10
                    await self.create_simulation_batch(
                        name=f"Auto Batch {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
                        particle_count=tasks_to_create * 10,  # 10 particles per task
                        time_horizon=1,
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
                    logger.info(f"✓ Auto-created batch with {tasks_to_create} tasks. New queue size: {self.work_queue.qsize()}")
                else:
                    logger.debug(f"Queue size OK: {queue_size} >= {self.min_pending_tasks}")
            else:
                if active_clients == 0:
                    logger.debug("No active clients, skipping task creation")
                else:
                    logger.debug(f"Queue size sufficient: {queue_size} >= {self.min_pending_tasks}")
                    
        except Exception as e:
            logger.error(f"Error checking and creating tasks: {e}")
    
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
        count = 0
        for work_unit in self.active_tasks.values():
            if work_unit.assigned_client == client_id:
                count += 1
        return count
    
    async def _process_task_results(self, work_unit: WorkUnit, client_id: str, result_data: bytes):
        """Process and store task results"""
        # This would validate and store results in database
        # For now, just log
        logger.info(f"Processing results for task {work_unit.id} from client {client_id}")

# Global task scheduler instance
task_scheduler = TaskScheduler()
