# 🌊 Ocean Plastic Drift Forecasting

A distributed computing platform for predicting ocean plastic drift patterns using volunteer computing resources.

## Overview

This system enables real-time forecasting of ocean plastic drift by distributing computational tasks across volunteer clients worldwide. The platform combines ocean current data, wind patterns, and particle physics simulations to predict where plastic debris will accumulate, helping cleanup organizations optimize their efforts.

## Architecture

### Backend (Python)
- **FastAPI**: High-performance async web framework
- **Celery**: Distributed task queue for background processing
- **SQLAlchemy**: Async ORM for database operations
- **Redis**: Caching and message broker
- **PostgreSQL**: Primary database with time-series support
- **WebSocket**: Real-time client communication

### Client (Electron)
- **Cross-platform desktop application**
- **Background task processing**
- **System resource monitoring**
- **Secure sandboxed execution**

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ocean-plastic-forecast
   ```

2. **Start the services**
   ```bash
   docker-compose up -d
   ```

3. **Verify services are running**
   ```bash
   docker-compose ps
   ```

4. **Access the API**
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Prometheus Metrics: http://localhost:9090
   - Grafana Dashboard: http://localhost:3000 (admin/admin)

### Manual Setup

#### Server Setup

1. **Install Python dependencies**
   ```bash
   cd server
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Initialize database**
   ```bash
   alembic upgrade head
   ```

4. **Start the server**
   ```bash
   uvicorn app.api.main:app --reload
   ```

5. **Start Celery worker**
   ```bash
   celery -A app.scheduler.task_manager.celery_app worker --loglevel=info
   ```

#### Client Setup

1. **Install Node.js dependencies**
   ```bash
   cd client
   npm install
   ```

2. **Start the client**
   ```bash
   npm start
   ```

## API Endpoints

### Client Management
- `POST /api/v1/clients/register` - Register new volunteer client
- `GET /api/v1/clients/me` - Get client information
- `PUT /api/v1/clients/heartbeat` - Update client status

### Task Management
- `GET /api/v1/tasks/available` - Get available tasks
- `POST /api/v1/tasks/complete` - Submit task results
- `POST /api/v1/tasks/fail` - Report task failure

### Forecasts
- `POST /api/v1/forecasts/generate` - Generate new forecast
- `GET /api/v1/forecasts/latest` - Get latest forecasts
- `GET /api/v1/forecasts/{id}` - Get specific forecast
- `GET /api/v1/forecasts/{id}/download` - Download forecast data

### Administration
- `GET /api/v1/admin/stats` - System statistics
- `GET /api/v1/admin/clients` - Client statistics
- `POST /api/v1/admin/batch/create` - Create simulation batch

## WebSocket Communication

Real-time communication between server and clients:

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/client/{client_id}');

// Message types
{
  "type": "heartbeat",
  "timestamp": "2024-01-15T10:30:00Z"
}

{
  "type": "task_request"
}

{
  "type": "task_completed",
  "task_id": "uuid",
  "result_data": "hex_encoded_results"
}
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/ocean_forecast
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Security
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256

# API
API_HOST=0.0.0.0
API_PORT=8000

# External APIs
NOAA_API_KEY=your-noaa-api-key
COPERNICUS_USERNAME=your-username
COPERNICUS_PASSWORD=your-password
```

## Deployment

### Docker Deployment

1. **Build images**
   ```bash
   docker-compose build
   ```

2. **Deploy with custom configuration**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

### Kubernetes Deployment

1. **Create namespace**
   ```bash
   kubectl apply -f k8s/namespace.yaml
   ```

2. **Create secrets**
   ```bash
   kubectl create secret generic ocean-forecast-secrets \
     --from-literal=database-url="postgresql://..." \
     --from-literal=redis-url="redis://..." \
     --from-literal=secret-key="..." \
     -n ocean-forecast
   ```

3. **Deploy services**
   ```bash
   kubectl apply -f k8s/
   ```

## Monitoring

### Metrics

The system exposes Prometheus metrics at `/metrics`:

- `ocean_forecast_tasks_completed_total` - Total completed tasks
- `ocean_forecast_active_clients` - Number of active clients
- `ocean_forecast_task_execution_seconds` - Task execution time
- `ocean_forecast_websocket_connections` - Active WebSocket connections

### Logging

Structured logging with different levels:
- Application logs: `/app/logs/ocean_forecast.log`
- Access logs: Nginx access logs
- Error logs: Application and system error logs

### Health Checks

- API Health: `GET /health`
- Database connectivity
- Redis connectivity
- Celery worker status

## Development

### Running Tests

```bash
# Server tests
cd server
pytest

# Client tests
cd client
npm test
```

### Code Quality

```bash
# Python linting
flake8 server/
black server/

# JavaScript linting
cd client
npm run lint
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Security

- All client communications are authenticated with JWT tokens
- Task execution runs in sandboxed environments
- Resource limits prevent abuse
- Rate limiting on API endpoints
- Input validation and sanitization

## Performance

### Scalability
- Horizontal scaling with multiple API instances
- Auto-scaling based on queue depth
- Geographic distribution support
- Load balancing across regions

### Optimization
- Redis caching for frequently accessed data
- Database connection pooling
- Async processing for I/O operations
- Efficient task distribution algorithms

## Troubleshooting

### Common Issues

1. **Database connection errors**
   - Check DATABASE_URL configuration
   - Verify PostgreSQL is running
   - Check network connectivity

2. **Celery worker not processing tasks**
   - Verify Redis is running
   - Check CELERY_BROKER_URL configuration
   - Monitor worker logs

3. **Client registration failures**
   - Check server URL configuration
   - Verify API is accessible
   - Check authentication tokens

### Logs

```bash
# View API logs
docker-compose logs -f api

# View worker logs
docker-compose logs -f celery-worker

# View all logs
docker-compose logs -f
```

## License

MIT License - see LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API documentation at `/docs`

---

**Ocean Plastic Drift Forecasting** - Democratizing ocean cleanup through distributed computing.
