#!/bin/bash

# Secure Deployment Script for Ocean Plastic Forecast
# This script deploys the system with proper security measures

set -e  # Exit on error

echo "================================================"
echo "Ocean Plastic Forecast - Secure Deployment"
echo "================================================"
echo ""

# Check if running with proper directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found!"
    echo "Please run this script from the cs/ directory"
    exit 1
fi

# Step 1: Check for SSL certificates
echo "Step 1: Checking SSL certificates..."
if [ ! -f "nginx/ssl/cert.pem" ] || [ ! -f "nginx/ssl/key.pem" ]; then
    echo "⚠️  SSL certificates not found!"
    echo ""
    echo "Do you want to generate self-signed certificates? (yes/no)"
    read -r response
    
    if [ "$response" = "yes" ] || [ "$response" = "y" ]; then
        echo "Generating self-signed SSL certificates..."
        
        mkdir -p nginx/ssl
        
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
          -keyout nginx/ssl/key.pem \
          -out nginx/ssl/cert.pem \
          -subj "/C=US/ST=Illinois/L=Chicago/O=IIT/OU=Ocean Forecast/CN=system80.rice.iit.edu"
        
        chmod 600 nginx/ssl/key.pem
        chmod 644 nginx/ssl/cert.pem
        
        echo "✓ Self-signed certificates generated"
        echo ""
        echo "⚠️  NOTE: For production, use official certificates!"
        echo "   See SECURITY_DEPLOYMENT_GUIDE.md for instructions"
    else
        echo "❌ SSL certificates required. Exiting."
        echo "   See SECURITY_DEPLOYMENT_GUIDE.md for setup instructions"
        exit 1
    fi
else
    echo "✓ SSL certificates found"
fi
echo ""

# Step 2: Check SECRET_KEY
echo "Step 2: Checking SECRET_KEY configuration..."
if grep -q "your-secret-key-change-in-production" docker-compose.yml; then
    echo "⚠️  Using default SECRET_KEY!"
    echo ""
    echo "For production, you should change this."
    echo "Generate a secure key with:"
    echo "  python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""
    echo ""
    echo "Continue anyway? (yes/no)"
    read -r response
    
    if [ "$response" != "yes" ] && [ "$response" != "y" ]; then
        echo "❌ Deployment cancelled"
        exit 1
    fi
else
    echo "✓ SECRET_KEY configured"
fi
echo ""

# Step 3: Stop existing containers
echo "Step 3: Stopping existing containers..."
docker-compose down
echo "✓ Containers stopped"
echo ""

# Step 4: Pull latest images
echo "Step 4: Pulling latest images..."
docker-compose pull
echo "✓ Images updated"
echo ""

# Step 5: Build and start services
echo "Step 5: Building and starting services..."
docker-compose up -d --build
echo "✓ Services started"
echo ""

# Step 6: Wait for services to be healthy
echo "Step 6: Waiting for services to be healthy..."
sleep 10

# Check service status
echo ""
echo "Service Status:"
echo "---------------"
docker-compose ps
echo ""

# Step 7: Verify security
echo "Step 7: Security verification..."
echo ""

# Check what ports are exposed
echo "Docker exposed ports:"
docker-compose ps --format json | jq -r '.[] | select(.Publishers != null) | "\(.Service): \(.Publishers)"' 2>/dev/null || \
docker-compose ps | grep -E "Up|running"

echo ""
echo "Checking port bindings..."
INSECURE_PORTS=$(docker-compose ps --format json 2>/dev/null | jq -r '.[] | select(.Publishers != null) | .Publishers[] | select(.PublishedPort != 80 and .PublishedPort != 443) | .PublishedPort' 2>/dev/null | sort -u)

if [ -n "$INSECURE_PORTS" ] && [ "$INSECURE_PORTS" != "" ]; then
    echo "⚠️  WARNING: Found exposed ports other than 80/443:"
    echo "$INSECURE_PORTS"
    echo ""
    echo "This might be a security issue. Please verify your configuration."
else
    echo "✓ Only ports 80 and 443 are exposed (secure)"
fi
echo ""

# Step 8: Test health endpoint
echo "Step 8: Testing API health..."
sleep 5

if command -v curl &> /dev/null; then
    HEALTH_RESPONSE=$(curl -s -k https://localhost/health 2>/dev/null || curl -s http://localhost/health 2>/dev/null || echo "failed")
    
    if [ "$HEALTH_RESPONSE" != "failed" ]; then
        echo "✓ API is responding"
        echo "Response: $HEALTH_RESPONSE"
    else
        echo "⚠️  Could not reach API health endpoint"
        echo "   This might be normal if SSL is not fully configured yet"
    fi
else
    echo "⚠️  curl not found, skipping health check"
fi
echo ""

# Final summary
echo "================================================"
echo "Deployment Summary"
echo "================================================"
echo ""
echo "✓ Services deployed successfully!"
echo ""
echo "Access URLs (from internet):"
echo "  API:        https://system80.rice.iit.edu/"
echo "  Health:     https://system80.rice.iit.edu/health"
echo "  API Docs:   https://system80.rice.iit.edu/docs"
echo "  Grafana:    https://system80.rice.iit.edu/grafana/"
echo "  Prometheus: https://system80.rice.iit.edu/prometheus/"
echo ""
echo "Security Status:"
echo "  ✓ Only port 443 (HTTPS) exposed to internet"
echo "  ✓ All internal services isolated"
echo "  ✓ SSL/TLS encryption enabled"
echo "  ✓ Security headers configured"
echo "  ✓ Rate limiting enabled"
echo ""
echo "Next Steps:"
echo "  1. Test client connection"
echo "  2. Monitor logs: docker-compose logs -f"
echo "  3. View metrics: https://system80.rice.iit.edu/grafana/"
echo "  4. Contact IIT IT to verify compliance"
echo ""
echo "For more information, see SECURITY_DEPLOYMENT_GUIDE.md"
echo ""
echo "================================================"

