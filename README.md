# ⚽ Advanced Football Analysis with AI

Professional-grade football analytics with AI-powered ball tracking, player analysis, and tactical insights.

## 🚀 Quick Start with Docker

### Prerequisites
- Docker Desktop installed
- Docker Compose installed
- At least 4GB RAM available

### 1. Clone and Setup
```bash
git clone <repository-url>
cd sri.html
```

### 2. Start the Application
```bash
chmod +x start.sh
./start.sh
```

### 3. Access the Application
- **Web Interface**: http://localhost:8501
- **Redis**: localhost:6379
- **MongoDB**: localhost:27017

## 🏗️ Architecture

### Services
- **webapp**: Streamlit frontend application
- **worker**: Celery background worker for video processing
- **redis**: Message broker for Celery tasks
- **mongo**: Database for storing analysis results

### Key Features Implemented

#### 1. ✅ Advanced Ball Detection Pipeline
- **YOLOv9c Fine-Tuned**: Custom model for football ball detection
- **SAHI Detection**: Slicing Aided Hyper Inference for tiny objects
- **Motion Enhancement**: Frame differencing + background subtraction
- **False Positive Filter**: Size, aspect ratio, and height validation
- **Ball Re-ID**: Trajectory-based re-identification

#### 2. ✅ ByteTracker Integration
- **Robust Tracking**: Reduced ID switches and improved occlusion handling
- **Multi-Object Tracking**: Consistent player IDs throughout the match
- **Team Classification**: Automatic team assignment by jersey color

#### 3. ✅ Docker Containerization
- **Multi-Service Stack**: Streamlit + Celery + Redis + MongoDB
- **One-Command Deployment**: `./start.sh` to launch everything
- **Production Ready**: Scalable architecture for cloud deployment

## 🛠️ Manual Setup (Alternative)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Start Services Manually
```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start MongoDB
mongod

# Terminal 3: Start Celery Worker
celery -A football_analysis worker --loglevel=info

# Terminal 4: Start Streamlit App
streamlit run football_analysis.py
```

## 📊 Usage

### 1. Ball Detection & Tracking
- Upload match video
- Configure detection settings
- View real-time ball tracking with trails
- Analyze possession statistics

### 2. Match Analysis
- Complete player tracking with constant IDs
- Team classification by jersey color
- Tactical pattern recognition
- Professional metrics (xG, PPDA)

### 3. AI Coach
- Conversational AI with memory
- Multi-step query processing
- Function calling capabilities
- Tactical insights generation

## 🐳 Docker Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up --build -d

# Scale workers
docker-compose up --scale worker=3 -d
```

## 🔧 Configuration

### Environment Variables
- `CELERY_BROKER_URL`: Redis connection URL
- `CELERY_RESULT_BACKEND`: Redis backend URL
- `MONGODB_URL`: MongoDB connection URL

### Volume Mounts
- `./outputs`: Processed videos and analysis results
- `./highlights`: Generated highlight clips
- `./animations`: Tactical scenario animations

## 📈 Performance

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible GPU for faster inference (optional)
- **Storage**: 10GB+ free space for video processing

### Optimization Tips
- Use GPU acceleration for YOLO inference
- Adjust detection confidence thresholds based on video quality
- Scale Celery workers based on CPU cores
- Use SSD storage for better I/O performance

## 🚨 Troubleshooting

### Common Issues

#### Docker Build Fails
```bash
# Clean Docker cache
docker system prune -a
docker-compose build --no-cache
```

#### Services Not Starting
```bash
# Check logs
docker-compose logs webapp
docker-compose logs worker
```

#### Port Already in Use
```bash
# Kill processes using ports
sudo lsof -ti:8501 | xargs kill -9
sudo lsof -ti:6379 | xargs kill -9
sudo lsof -ti:27017 | xargs kill -9
```

## 🔮 Advanced Features

### Ball Detection Pipeline
- High-resolution inference (1280px)
- Motion-based enhancement
- Kalman filter tracking
- Occlusion handling

### Player Tracking
- ByteTracker for robust ID consistency
- Team classification by jersey color
- Energy monitoring and fatigue detection
- Role identification (defender, midfielder, forward)

### Tactical Analysis
- Pattern recognition (pressing traps, build-up play)
- Real-time tactical alerts
- Trajectory prediction
- What-if scenario generation

## 📝 API Endpoints

### Background Processing
- `POST /api/process_video`: Submit video for background processing
- `GET /api/job_status/{job_id}`: Check processing status
- `GET /api/results/{job_id}`: Retrieve analysis results

### Real-time Analysis
- `WebSocket /ws/live_analysis`: Live video analysis stream
- `POST /api/semantic_search`: Search events using natural language

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8/YOLOv9 by Ultralytics
- ByteTracker for multi-object tracking
- Streamlit for the web interface
- OpenCV for computer vision
- Celery for distributed task processing