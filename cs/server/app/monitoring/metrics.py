"""
Prometheus metrics collection for monitoring
"""
from prometheus_client import Counter, Gauge, Histogram, start_http_server, CollectorRegistry
import time
import logging
from typing import Dict, Any
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Create custom registry
registry = CollectorRegistry()

# Define metrics
tasks_completed = Counter(
    'ocean_forecast_tasks_completed_total',
    'Total number of tasks completed',
    ['client_id', 'status'],
    registry=registry
)

active_clients = Gauge(
    'ocean_forecast_active_clients',
    'Number of active clients',
    registry=registry
)

task_execution_time = Histogram(
    'ocean_forecast_task_execution_seconds',
    'Time spent executing tasks',
    ['client_id'],
    registry=registry
)

websocket_connections = Gauge(
    'ocean_forecast_websocket_connections',
    'Number of active WebSocket connections',
    registry=registry
)

queue_size = Gauge(
    'ocean_forecast_queue_size',
    'Number of tasks in queue',
    ['queue_type'],
    registry=registry
)

forecast_generation_time = Histogram(
    'ocean_forecast_generation_seconds',
    'Time spent generating forecasts',
    ['forecast_type'],
    registry=registry
)

api_requests = Counter(
    'ocean_forecast_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

database_operations = Counter(
    'ocean_forecast_database_operations_total',
    'Total number of database operations',
    ['operation', 'table'],
    registry=registry
)

class MetricsCollector:
    def __init__(self, port: int = None):
        self.port = port or settings.metrics_port
        self.server_started = False
    
    def start_server(self):
        """Start Prometheus metrics server"""
        if not self.server_started:
            try:
                start_http_server(self.port, registry=registry)
                self.server_started = True
                logger.info(f"Metrics server started on port {self.port}")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
    
    def record_task_completion(self, client_id: str, status: str):
        """Record task completion"""
        tasks_completed.labels(client_id=client_id, status=status).inc()
    
    def update_active_clients(self, count: int):
        """Update active clients count"""
        active_clients.set(count)
    
    def record_execution_time(self, client_id: str, duration: float):
        """Record task execution time"""
        task_execution_time.labels(client_id=client_id).observe(duration)
    
    def update_websocket_connections(self, count: int):
        """Update WebSocket connections count"""
        websocket_connections.set(count)
    
    def update_queue_size(self, queue_type: str, size: int):
        """Update queue size"""
        queue_size.labels(queue_type=queue_type).set(size)
    
    def record_forecast_generation_time(self, forecast_type: str, duration: float):
        """Record forecast generation time"""
        forecast_generation_time.labels(forecast_type=forecast_type).observe(duration)
    
    def record_api_request(self, method: str, endpoint: str, status_code: int):
        """Record API request"""
        api_requests.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
    
    def record_database_operation(self, operation: str, table: str):
        """Record database operation"""
        database_operations.labels(operation=operation, table=table).inc()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        return {
            "active_clients": active_clients._value._value,
            "websocket_connections": websocket_connections._value._value,
            "tasks_completed": sum(
                sample.value for sample in tasks_completed.collect()[0].samples
            ),
            "timestamp": time.time()
        }

# Global metrics collector instance
metrics_collector = MetricsCollector()

def setup_metrics():
    """Initialize metrics collection"""
    metrics_collector.start_server()
    logger.info("Metrics collection initialized")

# Middleware for automatic API request tracking
class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Record API request
                    method = scope["method"]
                    path = scope["path"]
                    status_code = message["status"]
                    
                    metrics_collector.record_api_request(method, path, status_code)
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)
