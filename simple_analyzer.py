import cv2
import numpy as np
from ultralytics import YOLO

class SimpleFootballAnalyzer:
    def __init__(self, model_path="yolov8x.pt"):
        self.model = YOLO(model_path)
        self.ball_class_id = 32  # Sports ball class in COCO
        self.person_class_id = 0  # Person class in COCO
    
    def process_video(self, video_path, output_path=None):
        """Process video file and track players and ball
        
        Args:
            video_path (str): Path to input video
            output_path (str, optional): Path to save output video. Defaults to None.
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        ball_detections = []
        player_detections = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.model(frame)
                
                # Draw detections
                annotated_frame = frame.copy()
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get box coordinates and class
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            if class_id == self.ball_class_id:
                                # Ball detection (yellow)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                ball_detections.append({
                                    'frame': frame_count,
                                    'confidence': conf,
                                    'bbox': [x1, y1, x2, y2]
                                })
                            
                            elif class_id == self.person_class_id:
                                # Player detection (green)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                player_detections.append({
                                    'frame': frame_count,
                                    'confidence': conf,
                                    'bbox': [x1, y1, x2, y2]
                                })
                
                # Add frame counter
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame if output path is provided
                if output_path:
                    out.write(annotated_frame)
                
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"\rProcessing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end="")
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        finally:
            print("\nCleaning up...")
            cap.release()
            if output_path:
                out.release()
        
        return {
            'total_frames': frame_count,
            'ball_detections': ball_detections,
            'player_detections': player_detections
        }
