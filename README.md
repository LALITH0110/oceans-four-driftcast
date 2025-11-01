# vArI - Ocean Plastic Drift Forecasting Platform

> Distributed computing platform for predicting ocean plastic drift patterns using volunteer computational resources

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

## Overview

DriftCast is a volunteer distributed computing platform that harnesses idle computational resources to predict ocean plastic drift patterns. The system combines oceanographic data, numerical simulation algorithms, and reinforcement learning to model how plastic waste moves through ocean currents and winds.

## Problem Statement

Approximately 8 million tons of plastic waste enter the oceans annually. Understanding plastic drift patterns is critical for:

- **Targeted cleanup operations** - Cleanup organizations need to know where plastic accumulates to deploy resources effectively
- **Source identification** - Tracing plastic back to pollution sources helps inform policy and intervention strategies
- **Impact assessment** - Predicting where plastic will travel helps protect sensitive marine ecosystems and wildlife habitats
- **Research advancement** - Improved models contribute to oceanographic and environmental science

Traditional approaches rely on expensive supercomputer time or limited satellite observations. DriftCast democratizes this computational power by distributing simulations across volunteer machines, similar to BOINC projects like SETI@home or Folding@home.

## Use Cases

**Marine cleanup organizations** can use drift predictions to:
- Plan cleanup vessel routes and deployment schedules
- Estimate the volume and location of plastic accumulation zones
- Optimize resource allocation across multiple cleanup sites

**Environmental researchers** benefit from:
- Large-scale ensemble simulations that were previously computationally infeasible
- Validation data for oceanographic models
- Long-term drift pattern analysis across different seasons and conditions

**Policymakers and regulators** gain:
- Evidence-based insights for plastic pollution regulations
- Identification of high-risk zones requiring intervention
- Impact assessments for proposed cleanup or prevention measures

## Technology Stack

### Client Application

**Electron Desktop App** (Windows, macOS, Linux)
- **Electron** - Cross-platform desktop framework providing native OS integration
- **Node.js v18+** - JavaScript runtime for main process logic
- **Worker Threads** - Offload CPU-intensive drift simulations to separate threads without blocking UI
- **WebSocket (ws)** - Persistent bidirectional connection for real-time task assignment and result submission
- **systeminformation** - Hardware monitoring for CPU, memory, battery, and GPU stats
- **electron-store** - Persistent key-value storage for user settings and credentials
- **axios** - HTTP client for REST API communication

**Key client-side modules:**
- `main.js` - Application lifecycle, IPC handlers, tray integration
- `server-communicator.js` - WebSocket and REST API client with auto-reconnection
- `task-manager.js` - Task queue management, worker thread pool, concurrent task execution
- `system-monitor.js` - Real-time resource monitoring with configurable sampling intervals
- `renderer.js` - UI state management, live metrics display, achievement tracking

### Server Infrastructure

**FastAPI Backend**
- **FastAPI** - Modern async Python web framework with automatic OpenAPI documentation
- **Uvicorn** - ASGI server for handling async HTTP and WebSocket connections
- **SQLAlchemy** - ORM for database operations with async support
- **Pydantic** - Request/response validation and serialization
- **python-jose** - JWT token generation and validation for client authentication
- **WebSockets** - Real-time client connection management and task distribution

**Data Layer**
- **PostgreSQL 15** - Primary relational database for users, tasks, results, and leaderboards
- **Redis 7** - In-memory cache for session data, task queues, and rate limiting
- **Celery** - Distributed task queue for background jobs (result processing, ML training, aggregation)

**Infrastructure**
- **Docker & Docker Compose** - Container orchestration for all services
- **Nginx** - Reverse proxy for TLS termination, rate limiting, and WebSocket upgrade handling
- **Prometheus** - Time-series metrics collection from all services
- **Grafana** - Visualization dashboards for system monitoring and analytics

## Implementation Architecture

### System Design

```
┌─────────────────────────────────────────────────────┐
│                 Volunteer Clients                    │
│  • Task processing in isolated worker threads       │
│  • Resource monitoring with battery/CPU awareness   │
│  • Persistent WebSocket connection for task stream  │
└───────────────┬─────────────────────────────────────┘
                │ HTTPS/WSS (Port 443)
                │ JWT Authentication
                ▼
┌─────────────────────────────────────────────────────┐
│              Nginx Reverse Proxy                     │
│  • TLS 1.2/1.3 termination with SSL certificates    │
│  • Rate limiting (10 req/s per IP)                  │
│  • WebSocket protocol upgrade (HTTP → WS)           │
│  • Security headers (HSTS, X-Frame-Options)         │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│               FastAPI Application                    │
│  • REST endpoints: /api/v1/register, /tasks, etc   │
│  • WebSocket endpoint: /ws/client/{client_id}       │
│  • JWT-based authentication and session management  │
│  • Task distribution algorithm (priority queue)     │
│  • Result validation and database persistence       │
└───────────────┬─────────────────────────────────────┘
                │
    ┌───────────┴────────────┬─────────────┐
    ▼                        ▼             ▼
┌──────────┐         ┌──────────────┐  ┌────────────┐
│PostgreSQL│         │Celery Workers│  │  Redis     │
│          │         │              │  │            │
│ • clients│         │ • Task mgmt  │  │ • Cache    │
│ • tasks  │         │ • Result agg │  │ • Sessions │
│ • results│         │ • ML train   │  │ • Celery Q │
│ • users  │         │ • Stats comp │  └────────────┘
└──────────┘         └──────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Monitoring    │
                   │  • Prometheus   │
                   │  • Grafana      │
                   └─────────────────┘
```

### Client Implementation

**Registration and Authentication**
- Client auto-registers on first launch with system specifications
- Server issues JWT token with client_id claim
- Token stored locally and included in Authorization header for API calls
- WebSocket connection established using client_id in URL path

**Task Processing Pipeline**
1. Client connects via WebSocket and subscribes to task stream
2. Server pushes tasks based on client capabilities (CPU cores, memory)
3. Task manager spawns worker threads from pool (default: 2 concurrent, 4 in turbo mode)
4. Each worker runs drift simulation with given parameters (initial position, timesteps, environmental data)
5. Results serialized and submitted back to server via WebSocket
6. Server validates, stores in PostgreSQL, updates leaderboard

**Resource Management**
- System monitor polls CPU, memory, battery every 5 seconds
- Battery-saver mode: Auto-pause when on battery or reduce CPU to 20%
- Smart scheduling: 7x24 grid for allowed computing hours
- Auto-pause: Monitors system CPU; pauses if threshold exceeded
- User-configurable limits for CPU percentage and memory usage

**Drift Simulation Algorithm**
Each task contains:
- Initial particle coordinates (lat, lon)
- Simulation duration (timesteps)
- Environmental data references (ocean currents, wind fields)

Worker thread performs:
1. Load environmental data from embedded datasets or API
2. Iterative position update using Runge-Kutta 4th order integration
3. Apply ocean current velocity field at current position
4. Apply wind-driven surface drift (percentage of wind speed)
5. Store trajectory waypoints at configured intervals
6. Return final trajectory and statistics

### Server Implementation

**Task Distribution Strategy**
- Priority queue based on simulation urgency and spatial coverage needs
- Tasks generated by dividing ocean regions into grid cells
- Each cell spawned with multiple initial particle positions for ensemble prediction
- Server tracks which tasks assigned to which clients for fault tolerance
- Timeout mechanism: Reassign task if client doesn't respond in 10 minutes

**Database Schema**
```
clients: id, name, public_key, system_info, registered_at, last_seen
tasks: id, status, priority, parameters, assigned_to, created_at
results: id, task_id, client_id, trajectory_data, statistics, submitted_at
users: id, username, email (for researchers accessing aggregated data)
leaderboard: client_id, total_time, total_tasks, points, rank
```

**WebSocket Message Protocol**
```
Client → Server:
- {"type": "heartbeat"} (every 30s)
- {"type": "result", "task_id": ..., "data": ...}

Server → Client:
- {"type": "task", "task_id": ..., "parameters": ...}
- {"type": "ack", "task_id": ...}
- {"type": "leaderboard_update", "rank": ..., "points": ...}
```

**Celery Background Jobs**
- Result aggregation: Combine individual trajectories into probability heatmaps
- ML model training: Reinforcement learning to improve trajectory prediction
- Statistics computation: Daily/weekly aggregated metrics
- Cleanup: Remove old completed tasks, archive results

### Security Implementation

**Network Security**
- Only port 443 (HTTPS/WSS) exposed externally
- Port 22 (SSH) for server administration only
- All internal services (PostgreSQL, Redis, Prometheus, Grafana) on private Docker network
- No direct external access to databases or internal APIs

**Application Security**
- CORS restricted to production domain (system76.rice.iit.edu)
- Rate limiting: 10 requests/second per IP for API endpoints
- Rate limiting: 5 requests/second per IP for WebSocket connections
- JWT tokens with 7-day expiration
- HTTPS enforced with TLS 1.2+ and strong cipher suites
- Security headers: HSTS, X-Frame-Options, X-Content-Type-Options

**Data Privacy**
- No personally identifiable information collected
- System specs are anonymized (only hardware capabilities stored)
- No keystroke logging, screen capture, or network monitoring
- All computation happens in sandboxed worker threads

## Development Setup

### Client Development

```bash
cd cs/client
npm install
npm run dev
```

**Configuration:**
- Server URL: `cs/client/src/main/server-communicator.js:14`
- Default settings: `cs/client/src/main/main.js:176`

**Building for distribution:**
```bash
npm run build        # Builds for current platform
npm run build:all    # Builds for Windows, macOS, Linux
```

### Server Development

```bash
cd cs/server
cp .env.example .env
# Edit .env with database credentials, secret key

docker-compose up -d
```

**Services start on:**
- API: http://localhost:8000
- WebSocket: ws://localhost:8000/ws/client/{client_id}
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

**Running without Docker:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload
```

## Production Deployment

Production configuration is in `/production` directory with security hardening:

**Key differences from development:**
- CORS restricted to production domain
- All internal ports unexposed (only nginx on 443)
- SSL/TLS certificates required
- Debug mode disabled
- Secure password requirements for databases

**Deployment steps:**

```bash
cd production
cp .env.example .env
# Edit .env with secure credentials

# Generate SSL certificates (self-signed for testing)
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem

# Deploy
./deploy.sh
```

**Verify deployment:**
```bash
docker ps  # All services should be healthy
sudo lsof -i -P -n | grep LISTEN  # Only ports 22 and 443
curl -k https://localhost/health  # Should return {"status": "healthy"}
```

See `/production/README.md` for detailed deployment guide including:
- IIT Cloud deployment instructions
- Let's Encrypt certificate setup
- Firewall configuration
- Client configuration for production server

## Project Structure

```
oceans-four-driftcast/
├── cs/
│   ├── client/                      # Electron desktop application
│   │   ├── src/
│   │   │   ├── main/               # Main process (Node.js)
│   │   │   │   ├── main.js                   # App entry, IPC, tray
│   │   │   │   ├── server-communicator.js    # WebSocket/REST client
│   │   │   │   ├── task-manager.js           # Worker thread pool
│   │   │   │   ├── system-monitor.js         # Resource monitoring
│   │   │   │   └── drift-simulator.js        # Simulation algorithm
│   │   │   └── renderer/           # Renderer process (Chromium)
│   │   │       ├── index.html                # Main UI
│   │   │       ├── renderer.js               # UI logic
│   │   │       └── styles.css                # Styling
│   │   └── package.json
│   └── server/                      # FastAPI backend
│       ├── app/
│       │   ├── api/
│       │   │   ├── main.py                   # FastAPI app, routes
│       │   │   └── endpoints/                # REST endpoints
│       │   ├── models/
│       │   │   ├── client.py                 # SQLAlchemy models
│       │   │   ├── task.py
│       │   │   └── result.py
│       │   ├── scheduler/
│       │   │   ├── task_manager.py           # Celery tasks
│       │   │   └── distribution.py           # Task distribution logic
│       │   ├── websocket/
│       │   │   └── connection_manager.py     # WebSocket handler
│       │   └── core/
│       │       ├── config.py                 # Environment config
│       │       ├── security.py               # JWT, auth
│       │       └── database.py               # DB connection
│       ├── Dockerfile
│       ├── docker-compose.yml
│       └── requirements.txt
├── production/                      # Production deployment
│   ├── server/                     # Production server code
│   ├── docker-compose.yml          # Production container config
│   ├── nginx/
│   │   ├── nginx.conf              # Nginx reverse proxy config
│   │   └── ssl/                    # SSL certificates
│   ├── monitoring/
│   │   ├── prometheus.yml          # Metrics collection config
│   │   └── grafana/                # Dashboard definitions
│   ├── .env.example                # Environment variables template
│   ├── deploy.sh                   # Automated deployment script
│   └── README.md                   # Deployment guide
└── README.md                        # This file
```

## Testing

**Client tests:**
```bash
cd cs/client
npm test
```

**Server tests:**
```bash
cd cs/server
pytest
pytest --cov=app tests/  # With coverage
```

**Integration testing:**
1. Start local server: `cd cs/server && docker-compose up -d`
2. Start client: `cd cs/client && npm run dev`
3. Verify registration, task assignment, result submission
4. Check database: `docker exec -it ocean-forecast-db psql -U ocean_user -d ocean_forecast`

## Monitoring

**Prometheus metrics available at** `/metrics` endpoint:
- `client_connections_total` - Active WebSocket connections
- `tasks_assigned_total` - Total tasks distributed
- `tasks_completed_total` - Successfully processed tasks
- `api_request_duration_seconds` - Request latency histogram
- `system_cpu_usage_percent` - Server CPU utilization

**Grafana dashboards:**
- System overview: CPU, memory, disk, network
- Client activity: Active clients, task throughput, success rate
- API performance: Request rate, latency, error rate
- Database metrics: Query performance, connection pool

Access Grafana at `https://your-domain/grafana` (default user: admin)

## Academic Context

Developed for the **Grainger Innovation Prize** at Illinois Institute of Technology.

**Focus areas:**
- Social impact through environmental computing
- Technical innovation in distributed systems
- User experience with resource-aware design
- Scalability for global volunteer network

**Team:** Oceans Four
**Institution:** Illinois Institute of Technology
**Infrastructure:** IIT Cloud (system76.rice.iit.edu)

## Contributing

Contributions welcome via pull requests:

1. Fork repository
2. Create feature branch: `git checkout -b feature/name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push branch: `git push origin feature/name`
5. Open pull request

Code review process:
- All tests must pass
- Code coverage should not decrease
- Follow existing style conventions
- Update documentation for API changes

## License

MIT License - see LICENSE file for details

**Bug reports and feature requests:**
Use GitHub Issues: https://github.com/your-org/oceans-four-driftcast/issues

---

**Illinois Institute of Technology - Grainger Innovation Prize Project**
