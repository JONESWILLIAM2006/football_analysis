# Model Optimization & Training System
import torch
import torch.nn as nn
from ultralytics import YOLO
import onnx
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

class YOLOTrainer:
    def __init__(self, model_size='x'):
        self.model_size = model_size
        self.model = None
        
    def prepare_football_dataset(self, dataset_path, annotations_path):
        """Prepare dataset in YOLO format for football-specific training"""
        import yaml
        
        # Create dataset configuration
        dataset_config = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {
                0: 'player',
                1: 'referee', 
                2: 'ball',
                3: 'goal',
                4: 'penalty_area',
                5: 'center_circle'
            }
        }
        
        # Save dataset config
        config_path = Path(dataset_path) / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f)
        
        return str(config_path)
    
    def train_custom_model(self, dataset_config, epochs=100, imgsz=640):
        """Train custom YOLO model on football dataset"""
        # Load base model
        self.model = YOLO(f'yolov8{self.model_size}.pt')
        
        # Train model
        results = self.model.train(
            data=dataset_config,
            epochs=epochs,
            imgsz=imgsz,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch=16,
            workers=4,
            patience=50,
            save=True,
            cache=True,
            augment=True,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.0001,
            flipud=0.0,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4
        )
        
        return results
    
    def export_to_onnx(self, model_path, output_path):
        """Export trained model to ONNX format"""
        model = YOLO(model_path)
        model.export(format='onnx', dynamic=True, simplify=True)
        return output_path

class TensorRTOptimizer:
    def __init__(self):
        self.engine = None
        
    def convert_to_tensorrt(self, onnx_path, engine_path, precision='fp16'):
        """Convert ONNX model to TensorRT engine"""
        try:
            import tensorrt as trt
            
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            
            # Build engine
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            if precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
            
            engine = builder.build_engine(network, config)
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            return engine_path
            
        except ImportError:
            print("TensorRT not available, using ONNX Runtime instead")
            return onnx_path
    
    def load_tensorrt_engine(self, engine_path):
        """Load TensorRT engine for inference"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            
            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            self.engine = engine
            return True
            
        except ImportError:
            return False

class ONNXInference:
    def __init__(self, model_path, providers=None):
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
    
    def preprocess(self, image, input_size=(640, 640)):
        """Preprocess image for ONNX inference"""
        # Resize image
        resized = cv2.resize(image, input_size)
        
        # Normalize and transpose
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def inference(self, image):
        """Run inference on preprocessed image"""
        input_tensor = self.preprocess(image)
        
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        
        return outputs
    
    def postprocess(self, outputs, original_shape, conf_threshold=0.5):
        """Postprocess ONNX outputs to get detections"""
        predictions = outputs[0][0]  # Shape: (num_detections, 85)
        
        # Filter by confidence
        scores = predictions[:, 4]
        valid_indices = scores > conf_threshold
        valid_predictions = predictions[valid_indices]
        
        if len(valid_predictions) == 0:
            return []
        
        # Extract boxes and scores
        boxes = valid_predictions[:, :4]
        scores = valid_predictions[:, 4]
        class_scores = valid_predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        
        # Scale boxes to original image size
        h_orig, w_orig = original_shape[:2]
        boxes[:, [0, 2]] *= w_orig / 640
        boxes[:, [1, 3]] *= h_orig / 640
        
        detections = []
        for i in range(len(boxes)):
            detections.append({
                'bbox': boxes[i],
                'score': scores[i],
                'class_id': class_ids[i]
            })
        
        return detections

class ModelEnsemble:
    def __init__(self, model_paths):
        self.models = []
        for path in model_paths:
            if path.endswith('.onnx'):
                self.models.append(ONNXInference(path))
            else:
                self.models.append(YOLO(path))
    
    def predict_ensemble(self, image, weights=None):
        """Run ensemble prediction"""
        if weights is None:
            weights = [1.0] * len(self.models)
        
        all_detections = []
        
        for model, weight in zip(self.models, weights):
            if isinstance(model, ONNXInference):
                outputs = model.inference(image)
                detections = model.postprocess(outputs, image.shape)
            else:
                results = model(image, verbose=False)
                detections = self._yolo_to_detections(results[0])
            
            # Weight the confidence scores
            for det in detections:
                det['score'] *= weight
            
            all_detections.extend(detections)
        
        # Apply Non-Maximum Suppression
        final_detections = self._apply_nms(all_detections)
        return final_detections
    
    def _yolo_to_detections(self, result):
        """Convert YOLO results to detection format"""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                detections.append({
                    'bbox': box,
                    'score': score,
                    'class_id': int(cls)
                })
        
        return detections
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Group by class
        class_detections = {}
        for det in detections:
            class_id = det['class_id']
            if class_id not in class_detections:
                class_detections[class_id] = []
            class_detections[class_id].append(det)
        
        final_detections = []
        
        for class_id, dets in class_detections.items():
            # Sort by confidence
            dets.sort(key=lambda x: x['score'], reverse=True)
            
            keep = []
            while dets:
                current = dets.pop(0)
                keep.append(current)
                
                # Remove overlapping detections
                dets = [det for det in dets if self._calculate_iou(current['bbox'], det['bbox']) < iou_threshold]
            
            final_detections.extend(keep)
        
        return final_detections
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        # Calculate intersection
        inter_x1, inter_y1 = max(x1, x3), max(y1, y3)
        inter_x2, inter_y2 = min(x2, x4), min(y2, y4)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        
        return inter_area / (box1_area + box2_area - inter_area)

class GPUAccelerator:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    
    def optimize_inference(self, model):
        """Optimize model for GPU inference"""
        if torch.cuda.is_available():
            model = model.cuda()
            model = torch.jit.script(model)  # TorchScript optimization
        
        return model
    
    def batch_inference(self, model, images, batch_size=8):
        """Run batched inference for better GPU utilization"""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            if torch.cuda.is_available() and self.stream:
                with torch.cuda.stream(self.stream):
                    batch_results = model(batch)
            else:
                batch_results = model(batch)
            
            results.extend(batch_results)
        
        return results

class AutoTrainingPipeline:
    def __init__(self, base_model_path):
        self.base_model = YOLO(base_model_path)
        self.training_data = []
        self.performance_threshold = 0.8
    
    def add_training_sample(self, image, annotations):
        """Add new training sample to the pipeline"""
        self.training_data.append({
            'image': image,
            'annotations': annotations,
            'timestamp': time.time()
        })
    
    def should_retrain(self):
        """Determine if model should be retrained"""
        # Simple heuristic: retrain every 1000 samples
        return len(self.training_data) >= 1000
    
    def incremental_training(self, new_data_path, epochs=10):
        """Perform incremental training with new data"""
        if not self.should_retrain():
            return False
        
        # Prepare incremental dataset
        dataset_config = self._prepare_incremental_dataset(new_data_path)
        
        # Fine-tune model
        results = self.base_model.train(
            data=dataset_config,
            epochs=epochs,
            resume=True,  # Resume from existing weights
            patience=10,
            lr0=0.001,  # Lower learning rate for fine-tuning
            warmup_epochs=1
        )
        
        # Clear training data after successful training
        self.training_data = []
        
        return True
    
    def _prepare_incremental_dataset(self, data_path):
        """Prepare dataset for incremental training"""
        # Implementation would depend on your data format
        # This is a simplified version
        return {
            'path': data_path,
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'player', 1: 'referee', 2: 'ball'}
        }