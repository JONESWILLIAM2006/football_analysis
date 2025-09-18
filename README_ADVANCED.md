# ⚽ Advanced Football Analysis System

Professional-grade football analytics with AI-powered detection, tracking, and tactical insights.

## 🚀 Quick Start

### Option 1: Simple Version (Minimal Dependencies)
```bash
# Install basic requirements
pip install streamlit opencv-python numpy pandas plotly

# Run simple version
streamlit run football_analysis_simple.py
```

### Option 2: Advanced Version (Full Features)
```bash
# Install advanced requirements
pip install -r requirements_advanced.txt

# Run advanced version
streamlit run football_analysis.py
```

### Option 3: Docker Deployment (Production)
```bash
# Deploy complete system
./deploy_advanced.sh deploy

# Access at http://localhost:8501
```

## 🏗️ System Architecture

### Core Components

#### 1. **Detection Pipeline**
- **RT-DETR-v3**: Primary detector for players, ball, referees
- **YOLOv10/YOLOv8**: Alternative detectors with segmentation masks
- **SAHI**: Slicing Aided Hyper Inference for small objects
- **Motion Enhancement**: Frame differencing + background subtraction

#### 2. **Tracking System**
- **ByteTracker**: Robust multi-object tracking with ID consistency
- **StrongSORT+ReID**: Long-term tracking with re-identification
- **Team Classification**: Automatic team assignment by jersey color
- **Energy Monitoring**: Player fatigue and performance tracking

#### 3. **Geometry & Calibration**
- **Auto Pitch Detection**: Line and point detection with RANSAC
- **Homography Calculation**: Pixel ↔ meters conversion
- **Camera Calibration**: Per-scene calibration with line/ellipse cues

#### 4. **Event Detection**
- **Temporal Models**: 1D-CNN/LSTM/Transformer for event classification
- **Pass/Shot/Tackle Detection**: Multi-modal event recognition
- **Press/Turnover Analysis**: Tactical event identification

#### 5. **VAR Analysis**
- **Offside Detection**: Automated offside line calculation
- **Goal-Line Technology**: Ball crossing detection
- **Foul/Handball Detection**: Pose + trajectory analysis

#### 6. **Prediction Models**
- **xG Model**: Expected Goals using XGBoost → Transformer
- **Pass Success Prediction**: Sequence modeling
- **Next-Action Policy**: Imitation learning for tactical decisions
- **Fatigue/Injury Risk**: Load monitoring with pose asymmetry

#### 7. **Tactical Analysis**
- **Formation Detection**: DBSCAN/HDBSCAN clustering
- **Passing Networks**: NetworkX graph analysis
- **Space Control**: Voronoi diagrams for pitch control
- **Pattern Recognition**: Pressing traps, build-up play detection

## 📊 Features Matrix

| Feature | Simple Version | Advanced Version | Docker Version |
|---------|---------------|------------------|----------------|
| Player Detection | ✅ OpenCV | ✅ RT-DETR/YOLO | ✅ RT-DETR/YOLO |
| Ball Detection | ✅ Color/Shape | ✅ SAHI + Motion | ✅ SAHI + Motion |
| Player Tracking | ✅ Centroid | ✅ ByteTracker | ✅ ByteTracker |
| Team Classification | ✅ Basic | ✅ Advanced | ✅ Advanced |
| Event Detection | ✅ Basic | ✅ ML-based | ✅ ML-based |
| VAR Analysis | ❌ | ✅ Full | ✅ Full |
| xG Prediction | ❌ | ✅ XGBoost | ✅ XGBoost |
| Tactical Analysis | ❌ | ✅ Full | ✅ Full |
| Real-time Processing | ✅ | ✅ | ✅ |
| Multi-camera Sync | ❌ | ✅ | ✅ |
| RTSP Ingestion | ❌ | ❌ | ✅ |
| Background Processing | ❌ | ❌ | ✅ |
| API Access | ❌ | ❌ | ✅ |
| Monitoring | ❌ | ❌ | ✅ |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- CUDA 11.8+ (for GPU acceleration)
- Docker & Docker Compose (for production)
- 8GB+ RAM (16GB recommended)
- NVIDIA GPU (optional but recommended)

### Simple Installation
```bash
git clone <repository-url>
cd sri.html

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install basic requirements
pip install streamlit opencv-python numpy pandas plotly

# Run simple version
streamlit run football_analysis_simple.py
```

### Advanced Installation
```bash
# Install advanced requirements
pip install -r requirements_advanced.txt

# Download models (optional - will download automatically)
mkdir models
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/rtdetr-l.pt
cd ..

# Run advanced version
streamlit run football_analysis.py
```

### Docker Production Deployment
```bash
# Make deployment script executable
chmod +x deploy_advanced.sh

# Deploy complete system
./deploy_advanced.sh deploy

# Check status
./deploy_advanced.sh status

# View logs
./deploy_advanced.sh logs

# Scale workers
./deploy_advanced.sh scale 4
```

## 🎯 Usage Guide

### 1. **Video Upload & Analysis**
1. Access web interface at `http://localhost:8501`
2. Upload match video (MP4, AVI, MOV)
3. Configure detection settings
4. Start analysis and monitor progress
5. Review results and download reports

### 2. **Real-time RTSP Streams**
```bash
# Add RTSP stream via API
curl -X POST http://localhost:8000/api/rtsp_stream \
  -H "Content-Type: application/json" \
  -d '{"url": "rtsp://camera1:554/stream", "name": "Main Camera"}'

# View active streams
curl http://localhost:8000/api/streams
```

### 3. **API Usage**
```python
import requests

# Upload video
with open('match.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/upload_video',
        files={'file': f}
    )
file_id = response.json()['file_id']

# Start analysis
response = requests.post(f'http://localhost:8000/api/start_analysis/{file_id}')
job_id = response.json()['job_id']

# Check status
response = requests.get(f'http://localhost:8000/api/job_status/{job_id}')
print(response.json())
```

### 4. **WebSocket Live Analysis**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live_analysis/session123');

ws.onopen = function() {
    // Send frame data
    ws.send(JSON.stringify({
        type: 'frame',
        frame: base64_encoded_frame,
        frame_idx: 0
    }));
};

ws.onmessage = function(event) {
    const results = JSON.parse(event.data);
    console.log('Analysis results:', results);
};
```

## 📈 Performance Optimization

### Hardware Recommendations
- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7+)
- **RAM**: 16GB+ (32GB for multiple streams)
- **GPU**: NVIDIA RTX 3060+ (RTX 4080+ recommended)
- **Storage**: NVMe SSD for video processing
- **Network**: 10GbE for multi-camera ingestion

### Optimization Settings
```python
# Model optimization
model_config = {
    'detection_confidence': 0.5,
    'tracking_confidence': 0.7,
    'nms_threshold': 0.4,
    'max_detections': 100,
    'input_resolution': 1280,  # Higher = better accuracy, slower
    'batch_size': 1,  # Increase for batch processing
    'half_precision': True,  # FP16 for speed
    'tensorrt_optimization': True  # NVIDIA TensorRT
}

# Processing optimization
processing_config = {
    'frame_skip': 1,  # Process every N frames
    'roi_detection': True,  # Focus on pitch area
    'temporal_smoothing': True,  # Reduce jitter
    'parallel_workers': 4,  # Background processing
    'gpu_memory_fraction': 0.8  # GPU memory usage
}
```

## 🔧 Configuration

### Environment Variables
```bash
# Application
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Database
MONGODB_URL=mongodb://admin:password@mongo:27017/football_analysis
REDIS_URL=redis://redis:6379/0

# ML Models
MODEL_PATH=/app/models
DETECTION_CONFIDENCE=0.5
TRACKING_CONFIDENCE=0.7

# GPU
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

### Model Configuration
```yaml
# models/config.yaml
detection:
  primary_model: "rtdetr-l.pt"
  fallback_model: "yolov8m.pt"
  confidence_threshold: 0.5
  nms_threshold: 0.4
  
tracking:
  algorithm: "bytetrack"
  max_disappeared: 30
  distance_threshold: 50
  
analysis:
  enable_var: true
  enable_tactical: true
  enable_xg: true
  temporal_window: 60  # frames
```

## 📊 Monitoring & Observability

### Access Points
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090
- **MLflow Tracking**: http://localhost:5000
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)

### Key Metrics
- **Detection FPS**: Frames processed per second
- **Tracking Accuracy**: ID consistency and MOTA scores
- **Memory Usage**: GPU and system memory consumption
- **Queue Depth**: Background processing backlog
- **Error Rate**: Failed processing attempts

### Alerts
- High memory usage (>90%)
- Low detection FPS (<10 fps)
- Queue backlog (>100 jobs)
- Service downtime
- GPU temperature (>80°C)

## 🚨 Troubleshooting

### Common Issues

#### 1. **CUDA/GPU Issues**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install NVIDIA Docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
```

#### 2. **Memory Issues**
```bash
# Increase swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Optimize Docker memory
echo '{"default-ulimits":{"memlock":{"Hard":-1,"Name":"memlock","Soft":-1}}}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
```

#### 3. **Model Loading Issues**
```bash
# Download models manually
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/rtdetr-l.pt

# Check model permissions
chmod 644 models/*.pt
```

#### 4. **Port Conflicts**
```bash
# Find processes using ports
sudo lsof -ti:8501 | xargs kill -9  # Streamlit
sudo lsof -ti:8000 | xargs kill -9  # FastAPI
sudo lsof -ti:6379 | xargs kill -9  # Redis
sudo lsof -ti:27017 | xargs kill -9 # MongoDB
```

### Performance Tuning

#### 1. **Detection Optimization**
- Reduce input resolution for speed
- Increase confidence threshold to reduce false positives
- Use model quantization (INT8) for inference
- Enable TensorRT optimization

#### 2. **Tracking Optimization**
- Adjust distance thresholds
- Reduce maximum disappeared frames
- Use lighter tracking algorithms for speed

#### 3. **System Optimization**
- Use SSD storage for video processing
- Increase worker processes for parallel processing
- Optimize Docker resource limits
- Use GPU memory pooling

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd sri.html

# Install development dependencies
pip install -r requirements_advanced.txt
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black .
flake8 .
mypy .
```

### Adding New Features
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

### Model Integration
```python
# Add new detection model
class NewDetector(BaseDetector):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        # Implementation
        pass

# Register in factory
DETECTOR_REGISTRY['new_detector'] = NewDetector
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8/YOLOv9 models
- **ByteTracker**: Multi-object tracking
- **RT-DETR**: Real-time detection transformer
- **Streamlit**: Web interface framework
- **FastAPI**: Backend API framework
- **OpenCV**: Computer vision library

## 📞 Support

- **Documentation**: [Wiki](https://github.com/your-repo/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@footballanalysis.ai

---

**⚽ Ready to revolutionize football analysis? Get started today!**