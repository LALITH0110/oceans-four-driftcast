#!/bin/bash

# Generate self-signed SSL certificate for Ocean Plastic Forecast
# This script creates SSL certificates for HTTPS support

echo "==================================="
echo "SSL Certificate Generator"
echo "Ocean Plastic Forecast System"
echo "==================================="
echo ""

# Set variables
CERT_DIR="../nginx/ssl"
DOMAIN="system80.rice.iit.edu"
DAYS=365

# Create SSL directory if it doesn't exist
mkdir -p "$CERT_DIR"

echo "Generating self-signed SSL certificate..."
echo "Domain: $DOMAIN"
echo "Validity: $DAYS days"
echo ""

# Generate private key and certificate
openssl req -x509 -nodes -days $DAYS -newkey rsa:2048 \
  -keyout "$CERT_DIR/key.pem" \
  -out "$CERT_DIR/cert.pem" \
  -subj "/C=US/ST=Illinois/L=Chicago/O=Illinois Institute of Technology/OU=Ocean Plastic Forecast/CN=$DOMAIN"

# Set proper permissions
chmod 600 "$CERT_DIR/key.pem"
chmod 644 "$CERT_DIR/cert.pem"

echo ""
echo "✓ SSL certificate generated successfully!"
echo ""
echo "Certificate location: $CERT_DIR/cert.pem"
echo "Private key location: $CERT_DIR/key.pem"
echo ""
echo "==================================="
echo "IMPORTANT: Production Deployment"
echo "==================================="
echo ""
echo "For PRODUCTION use, replace these self-signed certificates with"
echo "certificates from a trusted Certificate Authority (CA)."
echo ""
echo "Recommended options:"
echo "1. Let's Encrypt (free) - Use certbot:"
echo "   sudo certbot certonly --standalone -d $DOMAIN"
echo ""
echo "2. IIT IT Department - Request official certificates"
echo "   Contact: cloud@iit.edu"
echo ""
echo "After obtaining certificates, copy them to:"
echo "  - Certificate: $CERT_DIR/cert.pem"
echo "  - Private Key: $CERT_DIR/key.pem"
echo ""

