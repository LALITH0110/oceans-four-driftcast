"""
Database models for Ocean Plastic Drift Forecasting system
"""
from sqlalchemy import Column, String, DateTime, JSON, LargeBinary, Float, Integer, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()

class Client(Base):
    """Volunteer client registration and management"""
    __tablename__ = "clients"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    public_key = Column(String, nullable=False)
    capabilities = Column(JSON)  # CPU cores, memory, etc.
    last_seen = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    trust_level = Column(Integer, default=1)
    
    # Relationships
    tasks = relationship("Task", back_populates="assigned_client")
    results = relationship("TaskResult", back_populates="client")

class Task(Base):
    """Individual simulation tasks"""
    __tablename__ = "tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    simulation_id = Column(UUID(as_uuid=True), nullable=False)
    parameters = Column(JSON, nullable=False)  # Simulation parameters
    input_data = Column(LargeBinary, nullable=False)  # Compressed ocean data
    status = Column(String(20), default='pending')  # pending, assigned, completed, failed
    assigned_client_id = Column(UUID(as_uuid=True), ForeignKey('clients.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    assigned_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    deadline = Column(DateTime)
    priority = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)
    
    # Relationships
    assigned_client = relationship("Client", back_populates="tasks")
    results = relationship("TaskResult", back_populates="task")

class TaskResult(Base):
    """Results from completed tasks"""
    __tablename__ = "task_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey('tasks.id'), nullable=False)
    client_id = Column(UUID(as_uuid=True), ForeignKey('clients.id'), nullable=False)
    result_data = Column(LargeBinary, nullable=False)  # Trajectory results
    execution_time = Column(Float)  # seconds
    quality_score = Column(Float)  # 0.0 to 1.0
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    task = relationship("Task", back_populates="results")
    client = relationship("Client", back_populates="results")

class Forecast(Base):
    """Generated forecast outputs"""
    __tablename__ = "forecasts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    forecast_type = Column(String(50), nullable=False)  # plastic_drift, concentration
    time_horizon = Column(Integer, nullable=False)  # hours
    spatial_bounds = Column(JSON)  # GeoJSON polygon
    confidence_level = Column(Float)
    result_data = Column(LargeBinary, nullable=False)  # Forecast map data
    forecast_metadata = Column(JSON)  # Additional forecast metadata
    created_at = Column(DateTime, default=datetime.utcnow)

class SimulationBatch(Base):
    """Batch of related simulation tasks"""
    __tablename__ = "simulation_batches"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(String(1000))
    total_tasks = Column(Integer, nullable=False)
    completed_tasks = Column(Integer, default=0)
    status = Column(String(20), default='pending')  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    parameters = Column(JSON)  # Global simulation parameters

class OceanData(Base):
    """Ocean data from external sources"""
    __tablename__ = "ocean_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    data_source = Column(String(50), nullable=False)  # NOAA, Copernicus, etc.
    data_type = Column(String(50), nullable=False)  # currents, winds, waves
    timestamp = Column(DateTime, nullable=False)
    spatial_bounds = Column(JSON)  # GeoJSON polygon
    data_payload = Column(LargeBinary, nullable=False)  # Compressed data
    quality_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
