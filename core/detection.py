import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import onnxruntime as ort

class RTDETRDetector:
    """RT-DETR-v3 detector for players, ball, refs"""
    def __init__(self, model_path: str = "rtdetr-l.pt", device: str = "cuda"):
        self.model = YOLO(model_path)
        self.device = device
        
    def detect(self, frame: np.ndarray, conf: float = 0.5) -> List[Dict]:
        results = self.model(frame, conf=conf, device=self.device)
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': cls,
                        'center': [(x1+x2)/2, (y1+y2)/2]
                    })
        return detections

class SmallBallDetector:
    """Specialized small ball detector with motion enhancement"""
    def __init__(self, model_path: str = "ball_yolov10.onnx"):
        self.session = ort.InferenceSession(model_path) if model_path.endswith('.onnx') else None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        
    def detect_with_motion(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> List[Dict]:
        # Motion enhancement
        if prev_frame is not None:
            diff = cv2.absdiff(frame, prev_frame)
            motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            enhanced_frame = cv2.bitwise_and(frame, frame, mask=motion_mask)
        else:
            enhanced_frame = frame
            
        # High-res crops for small objects
        crops = self._generate_crops(enhanced_frame, overlap=0.2)
        
        all_detections = []
        for crop, (x_offset, y_offset) in crops:
            detections = self._detect_crop(crop)
            # Adjust coordinates back to full frame
            for det in detections:
                det['bbox'][0] += x_offset
                det['bbox'][1] += y_offset
                det['bbox'][2] += x_offset
                det['bbox'][3] += y_offset
            all_detections.extend(detections)
            
        return self._nms_filter(all_detections)
    
    def _generate_crops(self, frame: np.ndarray, crop_size: int = 640, overlap: float = 0.2) -> List[Tuple]:
        h, w = frame.shape[:2]
        step = int(crop_size * (1 - overlap))
        crops = []
        
        for y in range(0, h - crop_size + 1, step):
            for x in range(0, w - crop_size + 1, step):
                crop = frame[y:y+crop_size, x:x+crop_size]
                crops.append((crop, (x, y)))
        return crops
    
    def _detect_crop(self, crop: np.ndarray) -> List[Dict]:
        if self.session is None:
            return []
            
        # ONNX inference
        input_tensor = cv2.resize(crop, (640, 640)).astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, ...]
        
        outputs = self.session.run(None, {'images': input_tensor})
        return self._parse_outputs(outputs[0])
    
    def _parse_outputs(self, output: np.ndarray) -> List[Dict]:
        detections = []
        for detection in output[0]:
            if detection[4] > 0.3:  # Ball confidence threshold
                x1, y1, x2, y2, conf = detection[:5]
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': 32,  # Ball class
                    'center': [(x1+x2)/2, (y1+y2)/2]
                })
        return detections
    
    def _nms_filter(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        if not detections:
            return []
            
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.3, iou_threshold)
        
        if len(indices) > 0:
            return [detections[i] for i in indices.flatten()]
        return []

class SegmentationMasks:
    """YOLOv8-seg for precise player silhouettes"""
    def __init__(self, model_path: str = "yolov8n-seg.pt"):
        self.model = YOLO(model_path)
        
    def get_masks(self, frame: np.ndarray, detections: List[Dict]) -> Dict[int, np.ndarray]:
        results = self.model(frame)
        masks = {}
        
        for i, r in enumerate(results):
            if r.masks is not None:
                for j, mask in enumerate(r.masks.data):
                    mask_np = mask.cpu().numpy()
                    masks[j] = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
        
        return masks