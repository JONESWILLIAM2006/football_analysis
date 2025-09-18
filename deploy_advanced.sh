#!/bin/bash

# Advanced Football Analysis System Deployment Script
# Supports GPU acceleration, monitoring, and production deployment

set -e

echo "🚀 Advanced Football Analysis System Deployment"
echo "=============================================="

# Configuration
COMPOSE_FILE="docker-compose-advanced.yml"
ENV_FILE=".env.production"
MODELS_DIR="./models"
DATA_DIR="./data"
MONITORING_DIR="./monitoring"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker (for GPU support)
    if command -v nvidia-docker &> /dev/null; then
        log_success "NVIDIA Docker detected - GPU acceleration available"
        export GPU_SUPPORT=true
    else
        log_warning "NVIDIA Docker not found - running in CPU mode"
        export GPU_SUPPORT=false
    fi
    
    # Check available memory
    AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$AVAILABLE_MEM" -lt 4 ]; then
        log_warning "Less than 4GB RAM available. System may be slow."
    fi
    
    log_success "Prerequisites check completed"
}

# Create directories
create_directories() {
    log_info "Creating required directories..."
    
    mkdir -p $MODELS_DIR
    mkdir -p $DATA_DIR
    mkdir -p $MONITORING_DIR/prometheus
    mkdir -p $MONITORING_DIR/grafana/dashboards
    mkdir -p $MONITORING_DIR/grafana/datasources
    mkdir -p ./outputs
    mkdir -p ./highlights
    mkdir -p ./logs
    
    log_success "Directories created"
}

# Download models
download_models() {
    log_info "Downloading pre-trained models..."
    
    cd $MODELS_DIR
    
    # YOLOv8 models
    if [ ! -f "yolov8n.pt" ]; then
        log_info "Downloading YOLOv8n..."
        wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    fi
    
    if [ ! -f "yolov8s.pt" ]; then
        log_info "Downloading YOLOv8s..."
        wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
    fi
    
    if [ ! -f "yolov8m.pt" ]; then
        log_info "Downloading YOLOv8m..."
        wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
    fi
    
    # RT-DETR model
    if [ ! -f "rtdetr-l.pt" ]; then
        log_info "Downloading RT-DETR..."
        wget -q https://github.com/ultralytics/assets/releases/download/v8.0.0/rtdetr-l.pt
    fi
    
    cd ..
    log_success "Models downloaded"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring configuration..."
    
    # Prometheus config
    cat > $MONITORING_DIR/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'webapp'
    static_configs:
      - targets: ['webapp:8501']
  
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
  
  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongo:27017']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF

    # Grafana datasource
    cat > $MONITORING_DIR/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Grafana dashboard
    cat > $MONITORING_DIR/grafana/dashboards/football-analysis.json << EOF
{
  "dashboard": {
    "id": null,
    "title": "Football Analysis System",
    "tags": ["football", "analysis"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "System Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "up",
            "legendFormat": "{{job}}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF

    log_success "Monitoring configuration created"
}

# Create environment file
create_env_file() {
    log_info "Creating environment configuration..."
    
    cat > $ENV_FILE << EOF
# Production Environment Configuration

# Application
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Database
MONGODB_URL=mongodb://admin:password@mongo:27017/football_analysis?authSource=admin
REDIS_URL=redis://redis:6379/0

# Object Storage
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false

# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# ML Models
MODEL_PATH=/app/models
DETECTION_CONFIDENCE=0.5
TRACKING_CONFIDENCE=0.7

# GPU Support
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)
EOF

    log_success "Environment file created"
}

# Build and start services
deploy_services() {
    log_info "Building and starting services..."
    
    # Build images
    log_info "Building Docker images..."
    docker-compose -f $COMPOSE_FILE build --parallel
    
    # Start services
    log_info "Starting services..."
    docker-compose -f $COMPOSE_FILE up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_service_health
    
    log_success "Services deployed successfully"
}

# Check service health
check_service_health() {
    log_info "Checking service health..."
    
    services=("webapp" "api" "redis" "mongo" "minio")
    
    for service in "${services[@]}"; do
        if docker-compose -f $COMPOSE_FILE ps $service | grep -q "Up"; then
            log_success "$service is running"
        else
            log_error "$service is not running"
        fi
    done
}

# Setup MinIO buckets
setup_minio() {
    log_info "Setting up MinIO buckets..."
    
    # Wait for MinIO to be ready
    sleep 10
    
    # Create buckets using MinIO client
    docker-compose -f $COMPOSE_FILE exec -T minio mc alias set local http://localhost:9000 minioadmin minioadmin
    docker-compose -f $COMPOSE_FILE exec -T minio mc mb local/videos --ignore-existing
    docker-compose -f $COMPOSE_FILE exec -T minio mc mb local/models --ignore-existing
    docker-compose -f $COMPOSE_FILE exec -T minio mc mb local/results --ignore-existing
    
    log_success "MinIO buckets created"
}

# Initialize database
init_database() {
    log_info "Initializing database..."
    
    # Wait for MongoDB to be ready
    sleep 15
    
    # Create indexes and initial data
    docker-compose -f $COMPOSE_FILE exec -T mongo mongosh football_analysis --eval "
        db.videos.createIndex({file_id: 1});
        db.analysis_jobs.createIndex({job_id: 1});
        db.analysis_results.createIndex({job_id: 1});
        db.events.createIndex({match_id: 1, frame: 1});
        db.rtsp_streams.createIndex({stream_id: 1});
    "
    
    log_success "Database initialized"
}

# Show deployment info
show_deployment_info() {
    echo ""
    echo "🎉 Deployment Complete!"
    echo "======================"
    echo ""
    echo "📊 Access Points:"
    echo "  • Web Interface:    http://localhost:8501"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • MinIO Console:    http://localhost:9001 (minioadmin/minioadmin)"
    echo "  • Grafana:          http://localhost:3000 (admin/admin)"
    echo "  • Prometheus:       http://localhost:9090"
    echo "  • MLflow:           http://localhost:5000"
    echo ""
    echo "🔧 Management Commands:"
    echo "  • View logs:        docker-compose -f $COMPOSE_FILE logs -f"
    echo "  • Stop services:    docker-compose -f $COMPOSE_FILE down"
    echo "  • Restart:          docker-compose -f $COMPOSE_FILE restart"
    echo "  • Scale workers:    docker-compose -f $COMPOSE_FILE up --scale worker=4 -d"
    echo ""
    echo "📈 System Status:"
    docker-compose -f $COMPOSE_FILE ps
    echo ""
    echo "🚀 System is ready for football analysis!"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    docker-compose -f $COMPOSE_FILE down
    docker system prune -f
    log_success "Cleanup completed"
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            create_directories
            download_models
            setup_monitoring
            create_env_file
            deploy_services
            setup_minio
            init_database
            show_deployment_info
            ;;
        "cleanup")
            cleanup
            ;;
        "restart")
            docker-compose -f $COMPOSE_FILE restart
            log_success "Services restarted"
            ;;
        "logs")
            docker-compose -f $COMPOSE_FILE logs -f
            ;;
        "status")
            docker-compose -f $COMPOSE_FILE ps
            ;;
        "scale")
            WORKERS=${2:-2}
            docker-compose -f $COMPOSE_FILE up --scale worker=$WORKERS -d
            log_success "Scaled to $WORKERS workers"
            ;;
        *)
            echo "Usage: $0 {deploy|cleanup|restart|logs|status|scale [num_workers]}"
            exit 1
            ;;
    esac
}

# Handle interrupts
trap cleanup INT TERM

# Run main function
main "$@"