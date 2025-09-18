# Football Analysis - Production Deployment

## Quick Start

1. **One-Command Deployment:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

2. **Manual Deployment:**
   ```bash
   docker-compose up --build -d
   ```

## Services

- **Web App**: http://localhost:8501
- **Task Monitor**: http://localhost:5555  
- **MongoDB**: localhost:27017
- **Redis**: localhost:6379

## Features Implemented

### 1. Enhanced User Dashboard 👤
- Multi-match comparison with side-by-side charts
- Player profiles aggregating stats across matches
- Performance trends over time
- Data persistence in MongoDB

### 2. AI Coach with LLM Function Calling 🧠
- Complex multi-step queries: "Compare player 7 and 10, show heatmap for better one"
- Conversation memory: "Show his heatmap" remembers context
- Function schemas for structured LLM interactions

### 3. Full Containerization 🚀
- Docker containers for all services
- One-command deployment with docker-compose
- Production-ready with persistent volumes
- Background processing with Celery workers

## Usage

```bash
# Start application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop application  
docker-compose down
```