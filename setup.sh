#!/bin/bash
# =============================================================================
# COMPLIANCE RAG SYSTEM - FULL SETUP SCRIPT
# =============================================================================
#
# This script sets up the complete production stack:
#   1. PostgreSQL + pgvector (vector store)
#   2. Redis (caching)
#   3. Airflow (orchestration)
#   4. Ollama (local LLM)
#
# Prerequisites:
#   - Docker & Docker Compose
#   - 16GB+ RAM recommended
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "COMPLIANCE RAG SYSTEM - SETUP"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Check Prerequisites
# -----------------------------------------------------------------------------

echo "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop."
    echo "   https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo "✓ Docker found"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found."
    exit 1
fi
echo "✓ Docker Compose found"

# Check available memory
if [[ "$OSTYPE" == "darwin"* ]]; then
    TOTAL_MEM=$(sysctl -n hw.memsize)
    TOTAL_MEM_GB=$((TOTAL_MEM / 1024 / 1024 / 1024))
else
    TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
fi

echo "✓ System memory: ${TOTAL_MEM_GB}GB"

if [ "$TOTAL_MEM_GB" -lt 8 ]; then
    echo "⚠️  Warning: Less than 8GB RAM. Some features may not work optimally."
fi

# -----------------------------------------------------------------------------
# Create directories
# -----------------------------------------------------------------------------

echo ""
echo "Creating directories..."

mkdir -p dags
mkdir -p logs
mkdir -p plugins
mkdir -p output
mkdir -p init_scripts/postgres
mkdir -p data

echo "✓ Directories created"

# -----------------------------------------------------------------------------
# Create .env file if not exists
# -----------------------------------------------------------------------------

if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    
    cat > .env << 'EOF'
# =============================================================================
# COMPLIANCE RAG SYSTEM - ENVIRONMENT CONFIGURATION
# =============================================================================

# PostgreSQL
POSTGRES_PASSWORD=compliance_dev_password_123

# Airflow
AIRFLOW_USER=admin
AIRFLOW_PASSWORD=admin
AIRFLOW_SECRET_KEY=your-secret-key-change-in-production
AIRFLOW_FERNET_KEY=

# Grafana
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin

# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
LLM_API_BASE=http://localhost:11434

# Snowflake (optional - for production data source)
# SNOWFLAKE_ACCOUNT=
# SNOWFLAKE_USER=
# SNOWFLAKE_PASSWORD=
# SNOWFLAKE_WAREHOUSE=COMPUTE_WH
# SNOWFLAKE_DATABASE=COMPLIANCE
# SNOWFLAKE_SCHEMA=PUBLIC
EOF
    
    echo "✓ .env file created"
    echo "  ⚠️  Please update .env with your credentials"
else
    echo "✓ .env file exists"
fi

# -----------------------------------------------------------------------------
# Start Docker services
# -----------------------------------------------------------------------------

echo ""
echo "Starting Docker services..."

# Use docker compose (v2) or docker-compose (v1)
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

$COMPOSE_CMD -f docker-compose.full.yml up -d postgres redis

echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Check if PostgreSQL is ready
for i in {1..30}; do
    if docker exec compliance_postgres pg_isready -U compliance_user -d compliance &> /dev/null; then
        echo "✓ PostgreSQL is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ PostgreSQL failed to start"
        exit 1
    fi
    sleep 2
done

echo "✓ Redis is ready"

# -----------------------------------------------------------------------------
# Start Airflow
# -----------------------------------------------------------------------------

echo ""
echo "Starting Airflow..."

$COMPOSE_CMD -f docker-compose.full.yml up -d airflow-init
echo "Waiting for Airflow initialization..."
sleep 30

$COMPOSE_CMD -f docker-compose.full.yml up -d airflow-webserver airflow-scheduler

echo "Waiting for Airflow to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8080/health | grep -q "healthy" &> /dev/null; then
        echo "✓ Airflow is ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "⚠️  Airflow may still be starting. Check http://localhost:8080"
    fi
    sleep 5
done

# -----------------------------------------------------------------------------
# Start Ollama
# -----------------------------------------------------------------------------

echo ""
echo "Starting Ollama..."

$COMPOSE_CMD -f docker-compose.full.yml up -d ollama

echo "Waiting for Ollama to be ready..."
sleep 10

for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "✓ Ollama is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Ollama may still be starting"
    fi
    sleep 2
done

# -----------------------------------------------------------------------------
# Pull LLM model
# -----------------------------------------------------------------------------

echo ""
echo "Pulling LLM model (this may take a few minutes)..."

# Get model from .env or default
LLM_MODEL=${LLM_MODEL:-llama3.1:8b}

docker exec compliance_ollama ollama pull $LLM_MODEL || {
    echo "⚠️  Failed to pull model automatically."
    echo "   Run manually: docker exec compliance_ollama ollama pull $LLM_MODEL"
}

# -----------------------------------------------------------------------------
# Start Grafana (optional)
# -----------------------------------------------------------------------------

echo ""
echo "Starting Grafana..."

$COMPOSE_CMD -f docker-compose.full.yml up -d grafana

echo "✓ Grafana started"

# -----------------------------------------------------------------------------
# Install Python dependencies
# -----------------------------------------------------------------------------

echo ""
echo "Installing Python dependencies..."

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    python3 -m venv .venv
    source .venv/bin/activate
fi

pip install -q --upgrade pip
pip install -q -r requirements.txt 2>/dev/null || {
    pip install -q psycopg2-binary redis requests pytest
}

echo "✓ Python dependencies installed"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "Services running:"
echo "  PostgreSQL:  localhost:5432  (user: compliance_user)"
echo "  Redis:       localhost:6379"
echo "  Airflow:     http://localhost:8080  (admin/admin)"
echo "  Ollama:      http://localhost:11434"
echo "  Grafana:     http://localhost:3000  (admin/admin)"
echo ""
echo "LLM Model: $LLM_MODEL"
echo ""
echo "Next steps:"
echo "  1. Update .env with your Snowflake credentials (if using)"
echo "  2. Open Airflow: http://localhost:8080"
echo "  3. Enable the 'daily_compliance_run' DAG"
echo "  4. Run: python run_demo.py"
echo ""
echo "To stop all services:"
echo "  docker compose -f docker-compose.full.yml down"
echo ""
