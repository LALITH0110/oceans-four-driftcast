#!/bin/bash

#################################################################
# Ocean Forecast Production Deployment Script
# For IIT Cloud deployment (port 443 only)
#################################################################

set -e  # Exit on error

echo "===================================="
echo "Ocean Forecast Production Deployment"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "ℹ $1"
}

#################################################################
# Step 1: Pre-deployment Checks
#################################################################

print_info "Step 1: Running pre-deployment checks..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_success "Docker is installed"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
print_success "Docker Compose is installed"

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_error ".env file not found!"
    print_info "Please copy .env.example to .env and fill in the values:"
    print_info "  cp .env.example .env"
    print_info "  nano .env"
    exit 1
fi
print_success ".env file exists"

# Load environment variables
source .env

# Check if SECRET_KEY is changed from default
if [ "$SECRET_KEY" == "CHANGE_THIS_TO_A_RANDOM_32_CHAR_STRING" ] || [ -z "$SECRET_KEY" ]; then
    print_error "SECRET_KEY not set in .env file!"
    print_info "Generate a secure key with:"
    print_info "  python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""
    exit 1
fi
print_success "SECRET_KEY is configured"

# Check if POSTGRES_PASSWORD is changed from default
if [ "$POSTGRES_PASSWORD" == "CHANGE_THIS_SECURE_PASSWORD" ] || [ -z "$POSTGRES_PASSWORD" ]; then
    print_error "POSTGRES_PASSWORD not set in .env file!"
    exit 1
fi
print_success "POSTGRES_PASSWORD is configured"

# Check if SSL certificates exist
if [ ! -f "nginx/ssl/cert.pem" ] || [ ! -f "nginx/ssl/key.pem" ]; then
    print_warning "SSL certificates not found in nginx/ssl/"
    read -p "Generate self-signed certificates for testing? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Generating self-signed SSL certificates..."
        mkdir -p nginx/ssl
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/key.pem \
            -out nginx/ssl/cert.pem \
            -subj "/C=US/ST=Illinois/L=Chicago/O=IIT/OU=Ocean Forecast/CN=${SERVER_HOST:-system80.rice.iit.edu}"
        print_success "Self-signed certificates generated"
        print_warning "⚠ For production, replace with proper certificates from IIT!"
    else
        print_error "Deployment cancelled. Please add SSL certificates first."
        exit 1
    fi
else
    print_success "SSL certificates found"
fi

# Check if monitoring directory exists
if [ ! -d "monitoring" ]; then
    print_warning "Monitoring directory not found. Creating..."
    mkdir -p monitoring
    # Copy from cs directory if available
    if [ -d "../cs/monitoring" ]; then
        cp -r ../cs/monitoring/* monitoring/
        print_success "Copied monitoring configuration from cs/"
    fi
fi

echo ""

#################################################################
# Step 2: Stop existing containers
#################################################################

print_info "Step 2: Stopping existing containers..."

if [ "$(docker ps -q -f name=ocean-forecast)" ]; then
    docker-compose down
    print_success "Stopped existing containers"
else
    print_info "No existing containers to stop"
fi

echo ""

#################################################################
# Step 3: Build and start services
#################################################################

print_info "Step 3: Building and starting services..."

print_info "Building Docker images..."
docker-compose build --no-cache

print_info "Starting services..."
docker-compose up -d

print_success "Services started"

echo ""

#################################################################
# Step 4: Wait for services to be healthy
#################################################################

print_info "Step 4: Waiting for services to be healthy..."

max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if docker-compose ps | grep -q "healthy"; then
        sleep 2
        attempt=$((attempt + 1))
        if [ $((attempt % 5)) -eq 0 ]; then
            print_info "Still waiting... ($attempt/$max_attempts)"
        fi
    else
        break
    fi
done

sleep 5  # Give services extra time to stabilize

print_success "Services are healthy"

echo ""

#################################################################
# Step 5: Verify deployment
#################################################################

print_info "Step 5: Verifying deployment..."

# Check only port 443 is exposed
exposed_ports=$(docker ps --format '{{.Ports}}' | grep -o '0.0.0.0:[0-9]*' | cut -d':' -f2 | sort -u)
if echo "$exposed_ports" | grep -v '^443$' | grep -v '^80$' | grep -q .; then
    print_error "Unexpected ports exposed!"
    echo "Exposed ports: $exposed_ports"
    print_warning "Only ports 80 and 443 should be exposed"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_success "Only ports 80 and 443 are exposed"
fi

# Test health endpoint
print_info "Testing health endpoint..."
sleep 3  # Wait a bit more

if curl -k -f https://localhost/health &> /dev/null; then
    print_success "Health endpoint is responding"
elif curl -f http://localhost/health &> /dev/null; then
    print_success "Health endpoint is responding (HTTP)"
    print_warning "HTTPS might not be configured yet"
else
    print_warning "Health endpoint not responding yet (this is normal, may take a minute)"
fi

echo ""

#################################################################
# Step 6: Display deployment information
#################################################################

print_success "==================================="
print_success "Deployment Complete!"
print_success "==================================="
echo ""

print_info "Services Status:"
docker-compose ps

echo ""
print_info "Access Points:"
print_info "  • API: https://${SERVER_HOST:-system80.rice.iit.edu}/api/v1/"
print_info "  • Health Check: https://${SERVER_HOST:-system80.rice.iit.edu}/health"
print_info "  • WebSocket: wss://${SERVER_HOST:-system80.rice.iit.edu}/ws/client/{client_id}"
print_info "  • Prometheus: https://${SERVER_HOST:-system80.rice.iit.edu}/prometheus/"
print_info "  • Grafana: https://${SERVER_HOST:-system80.rice.iit.edu}/grafana/"

echo ""
print_info "Container Logs:"
print_info "  View all logs: docker-compose logs -f"
print_info "  View API logs: docker-compose logs -f api"
print_info "  View Nginx logs: docker-compose logs -f nginx"

echo ""
print_warning "Important Next Steps:"
print_warning "1. Test the API: curl -k https://localhost/health"
print_warning "2. Update client configuration to: https://${SERVER_HOST:-system80.rice.iit.edu}"
print_warning "3. If using self-signed certificates, replace with IIT certificates"
print_warning "4. Configure firewall to allow only port 443"
print_warning "5. Test client connections before distributing to volunteers"

echo ""
print_success "Deployment script completed successfully!"
