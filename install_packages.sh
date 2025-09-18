#!/bin/bash
# Install essential packages for football analysis

source .venv/bin/activate

# Install core packages that work without compilation
pip install ultralytics
pip install onnxruntime
pip install mediapipe

echo "Packages installed successfully!"