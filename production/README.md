# Ocean Forecast Production Deployment Guide

**For IIT Cloud Deployment (Port 443 Only)**

---

## 📋 Prerequisites

Before deploying, ensure you have:

- [x] SSH access to IIT cloud server (port 22)
- [x] Docker and Docker Compose installed on the server
- [x] SSL certificates from IIT IT department (`cert.pem` and `key.pem`)
- [x] Domain name configured: `system80.rice.iit.edu`
- [x] Firewall configured to allow only ports 443 (HTTPS) and 22 (SSH)

---

## 📁 Directory Structure

```
production/
├── server/              # FastAPI application (modified for production)
├── nginx/               # Nginx reverse proxy configuration
│   ├── nginx.conf       # Production-ready config
│   └── ssl/             # SSL certificates go here
│       ├── cert.pem     # ← Add your IIT SSL certificate
│       └── key.pem      # ← Add your IIT private key
├── k8s/                 # Kubernetes manifests (optional)
├── monitoring/          # Prometheus & Grafana config
├── docker-compose.yml   # Production Docker Compose
├── .env.example         # Environment variables template
├── deploy.sh            # Automated deployment script
└── README.md            # This file
```

---

## 🚀 Deployment Steps

### Step 1: Transfer Files to IIT Server

```bash
# On your local machine
cd /path/to/oceans-four-driftcast

# Option A: Using scp
scp -r production/ your-username@system80.rice.iit.edu:~/ocean-forecast-prod

# Option B: Using git (recommended)
git clone <your-repo-url>
cd oceans-four-driftcast
```

### Step 2: SSH into IIT Server

```bash
ssh your-username@system80.rice.iit.edu
cd ocean-forecast-prod  # or ~/oceans-four-driftcast/production
```

### Step 3: Install SSL Certificates

```bash
# Create SSL directory if not exists
mkdir -p nginx/ssl

# Copy your IIT-provided certificates
# Option A: If you have them locally, scp them:
# scp cert.pem system80.rice.iit.edu:~/ocean-forecast-prod/nginx/ssl/
# scp key.pem system80.rice.iit.edu:~/ocean-forecast-prod/nginx/ssl/

# Option B: For testing only - generate self-signed
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem \
  -subj "/C=US/ST=Illinois/L=Chicago/O=IIT/OU=Ocean Forecast/CN=system80.rice.iit.edu"

# ⚠️ IMPORTANT: Replace self-signed with real IIT certificates before production!
```

### Step 4: Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file
nano .env
```

**Required Configuration:**

```bash
# Database
POSTGRES_PASSWORD=<generate-strong-password>

# API Security
SECRET_KEY=<generate-32-char-random-string>
DEBUG=false

# Server
SERVER_HOST=system80.rice.iit.edu

# Monitoring
GRAFANA_ADMIN_PASSWORD=<generate-strong-password>
```

**Generate secure credentials:**

```bash
# Generate SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate passwords
openssl rand -base64 32
```

### Step 5: Run Deployment Script

```bash
# Make script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

The script will:
1. ✅ Check prerequisites (Docker, env vars, certificates)
2. ✅ Stop existing containers
3. ✅ Build Docker images
4. ✅ Start all services
5. ✅ Verify only port 443 is exposed
6. ✅ Test health endpoint

### Step 6: Verify Deployment

```bash
# Check running containers
docker ps

# Should show ONLY these port mappings:
# 0.0.0.0:443->443/tcp
# 0.0.0.0:80->80/tcp

# Test health endpoint
curl -k https://localhost/health

# Expected response:
# {"status":"healthy","timestamp":"..."}

# View logs
docker-compose logs -f api
```

---

## 🔒 Security Verification Checklist

After deployment, verify:

- [ ] **Only ports 443 and 80 exposed** (run: `docker ps`)
- [ ] **HTTPS working** (run: `curl -k https://localhost/health`)
- [ ] **HTTP redirects to HTTPS** (run: `curl -I http://localhost`)
- [ ] **Database not accessible externally** (should fail: `nc -zv localhost 5432`)
- [ ] **Redis not accessible externally** (should fail: `nc -zv localhost 6379`)
- [ ] **Metrics only accessible internally** (should fail: `curl http://localhost:8001/metrics`)
- [ ] **Strong passwords in .env**
- [ ] **CORS restricted to IIT domain**

---

## 📱 Update Client Configuration

After successful server deployment, update the client app:

### File: `cs/client/src/main/server-communicator.js`

```javascript
// Change line 14 from:
this.serverUrl = 'http://localhost:8000';

// To:
this.serverUrl = 'https://system80.rice.iit.edu';
```

### File: `cs/client/src/main/main.js`

```javascript
// Change line 178 from:
serverUrl: store.get('serverUrl', 'http://localhost:8000'),

// To:
serverUrl: store.get('serverUrl', 'https://system80.rice.iit.edu'),
```

### Rebuild Client

```bash
cd cs/client
npm run build
```

Then distribute the updated client to volunteers.

---

## 🐛 Troubleshooting

### Problem: SSL Certificate Errors

```bash
# Check certificate validity
openssl x509 -in nginx/ssl/cert.pem -text -noout

# Check if cert matches key
openssl x509 -in nginx/ssl/cert.pem -noout -modulus | md5sum
openssl rsa -in nginx/ssl/key.pem -noout -modulus | md5sum
# Should match ^
```

### Problem: Port 443 Already in Use

```bash
# Find what's using port 443
sudo lsof -i :443

# Stop the service or change nginx port temporarily
```

### Problem: WebSocket Connections Failing

```bash
# Check nginx logs
docker-compose logs nginx

# Test WebSocket upgrade
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: test" \
  https://localhost/ws/client/test-id
```

### Problem: Database Connection Errors

```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Verify environment variables
docker-compose exec api env | grep DATABASE_URL

# Test database connection
docker-compose exec postgres psql -U ocean_user -d ocean_forecast -c "SELECT 1;"
```

### Problem: Services Not Healthy

```bash
# Check individual service health
docker-compose ps

# Restart unhealthy service
docker-compose restart api

# Check logs for errors
docker-compose logs --tail=100 api
```

---

## 📊 Monitoring & Maintenance

### Access Monitoring Dashboards

- **Prometheus**: `https://system80.rice.iit.edu/prometheus/`
- **Grafana**: `https://system80.rice.iit.edu/grafana/`
  - Default login: `admin` / `<your-GRAFANA_ADMIN_PASSWORD>`

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f nginx
docker-compose logs -f celery-worker

# Last 100 lines
docker-compose logs --tail=100 api
```

### Backup Database

```bash
# Create backup
docker-compose exec postgres pg_dump -U ocean_user ocean_forecast > backup_$(date +%Y%m%d).sql

# Restore backup
docker-compose exec -T postgres psql -U ocean_user ocean_forecast < backup_20241030.sql
```

### Update Deployment

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Or use the deploy script
./deploy.sh
```

---

## 🔄 Kubernetes Deployment (Alternative)

If using Kubernetes instead of Docker Compose:

### Step 1: Create TLS Secret

```bash
kubectl create secret tls ocean-forecast-tls-secret \
  --cert=nginx/ssl/cert.pem \
  --key=nginx/ssl/key.pem \
  --namespace=default
```

### Step 2: Create ConfigMap for Environment

```bash
kubectl create configmap ocean-forecast-env \
  --from-env-file=.env \
  --namespace=default
```

### Step 3: Apply Manifests

```bash
cd k8s
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
```

### Step 4: Verify

```bash
# Check pods
kubectl get pods

# Check ingress
kubectl get ingress

# Check services
kubectl get services

# View logs
kubectl logs -f deployment/ocean-forecast-api
```

---

## 📝 Architecture Overview

### Production Network Flow

```
Internet
    │
    └─→ Port 443 (HTTPS Only)
            │
            ▼
    Nginx Reverse Proxy (TLS Termination)
            │
            ├─→ /api/* ────────→ FastAPI:8000
            ├─→ /ws/* ─────────→ FastAPI:8000 (WebSocket)
            ├─→ /health ───────→ FastAPI:8000
            ├─→ /prometheus/* ─→ Prometheus:9090
            └─→ /grafana/* ────→ Grafana:3000

Internal Docker Network (Not accessible from internet):
    ├─→ PostgreSQL:5432
    ├─→ Redis:6379
    ├─→ Celery Worker
    └─→ Celery Beat
```

### Port Exposure Summary

| Service | Internal Port | Externally Accessible | Access Method |
|---------|--------------|----------------------|---------------|
| Nginx | 443, 80 | ✅ YES | Direct (internet) |
| FastAPI | 8000, 8001 | ❌ NO | Via Nginx only |
| PostgreSQL | 5432 | ❌ NO | Docker network only |
| Redis | 6379 | ❌ NO | Docker network only |
| Prometheus | 9090 | ❌ NO | Via Nginx only |
| Grafana | 3000 | ❌ NO | Via Nginx only |

---

## ⚠️ Important Notes

1. **SSL Certificates**: Self-signed certificates are for testing only. Get proper certificates from IIT IT department before going live.

2. **Secrets Management**: Never commit `.env` file to git. Keep it secure on the server only.

3. **Firewall**: Ensure IIT cloud firewall allows only:
   - Port 443 (HTTPS) - for client connections
   - Port 22 (SSH) - for server administration

4. **CORS**: Production server is configured to accept connections only from `https://system80.rice.iit.edu`. Update if your domain changes.

5. **Client Updates**: All volunteer clients must be updated to use `https://system80.rice.iit.edu` instead of `localhost:8000`.

6. **Database Backups**: Set up automated backups for PostgreSQL data.

7. **Monitoring**: Check Grafana dashboards regularly for system health.

---

## 🆘 Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Verify environment variables: `cat .env`
3. Check port exposure: `docker ps`
4. Review nginx config: `cat nginx/nginx.conf`

---

## ✅ Deployment Checklist

Before going live:

- [ ] SSL certificates installed from IIT
- [ ] Strong passwords in `.env`
- [ ] `DEBUG=false` in `.env`
- [ ] Only port 443 exposed (`docker ps` verification)
- [ ] Health endpoint responds: `curl https://localhost/health`
- [ ] Database backups configured
- [ ] Client apps updated to production URL
- [ ] Test WebSocket connections
- [ ] Test full task submission flow
- [ ] Monitoring dashboards accessible
- [ ] Firewall rules configured

---

**Deployment prepared by:** Claude Code
**Last updated:** October 30, 2024
