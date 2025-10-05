"""
Configuration settings for Ocean Plastic Forecast application
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Application
    app_name: str = "Ocean Plastic Forecast API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: str = "postgresql://user:password@localhost/ocean_forecast"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Monitoring
    metrics_port: int = 8001
    log_level: str = "INFO"
    
    # Task Management
    max_tasks_per_client: int = 5
    task_timeout_minutes: int = 30
    max_retry_count: int = 3
    
    # Data Sources
    noaa_api_key: Optional[str] = None
    copernicus_username: Optional[str] = None
    copernicus_password: Optional[str] = None
    
    # Docker
    docker_image_prefix: str = "ocean-forecast"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
