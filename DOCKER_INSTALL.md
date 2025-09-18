# 🐳 Docker Installation Guide for macOS

## Quick Installation

### Option 1: Docker Desktop (Recommended)
1. **Download Docker Desktop**:
   - Go to https://www.docker.com/products/docker-desktop/
   - Click "Download for Mac"
   - Choose your chip type (Intel or Apple Silicon)

2. **Install Docker Desktop**:
   - Open the downloaded `.dmg` file
   - Drag Docker to Applications folder
   - Launch Docker from Applications

3. **Verify Installation**:
   ```bash
   docker --version
   docker-compose --version
   ```

### Option 2: Homebrew (Command Line)
```bash
# Install Docker
brew install --cask docker

# Start Docker Desktop
open /Applications/Docker.app

# Verify installation
docker --version
docker-compose --version
```

## After Installation

1. **Start Docker Desktop** (if not already running)
2. **Wait for Docker to start** (you'll see the whale icon in your menu bar)
3. **Run the application**:
   ```bash
   cd sri.html
   chmod +x start.sh
   ./start.sh
   ```

## System Requirements
- macOS 10.15 or later
- At least 4GB RAM available for Docker
- 2GB free disk space

## Troubleshooting

### If Docker Desktop won't start:
```bash
# Reset Docker Desktop
rm -rf ~/Library/Group\ Containers/group.com.docker
rm -rf ~/Library/Containers/com.docker.docker
```

### If ports are in use:
```bash
# Kill processes on required ports
sudo lsof -ti:8501 | xargs kill -9
sudo lsof -ti:6379 | xargs kill -9  
sudo lsof -ti:27017 | xargs kill -9
```

## Quick Start After Installation
```bash
# Navigate to project
cd sri.html

# Make startup script executable
chmod +x start.sh

# Start the entire application stack
./start.sh
```

The application will be available at: http://localhost:8501