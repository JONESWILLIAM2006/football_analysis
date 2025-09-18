"""
Auto Broadcast System for Football Analysis
This module automatically connects to a video stream and runs the analysis
"""

import os
from rtsp_ingester import RTSPIngester
import cv2
import time
import asyncio
from football_analysis import DETECTOR_MODEL
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoBroadcast:
    def __init__(self, stream_url=None):
        self.stream_url = stream_url or "rtsp://0.0.0.0:8554/stream"  # Default RTSP URL
        self.ingester = RTSPIngester()
        self.detector = DETECTOR_MODEL
        self.running = False
        
    def start(self):
        """Start the auto broadcast system"""
        try:
            # Add the stream
            stream_id = "main_camera"
            self.ingester.add_stream(stream_id, self.stream_url, "Main Camera")
            
            # Start ingestion
            self.ingester.start_ingestion()
            self.running = True
            
            logger.info("Started auto broadcast system")
            
            # Main processing loop
            while self.running:
                # Get synced frames
                frames = self.ingester.get_synced_frames()
                
                if not frames:
                    logger.warning("No frames received")
                    time.sleep(1)
                    continue
                
                # Process main camera frame
                if stream_id in frames:
                    frame, timestamp = frames[stream_id]
                    
                    # Run detection
                    results = self.detector(frame)
                    
                    # Process results (you can add more analysis here)
                    for result in results:
                        boxes = result.boxes.cpu().numpy()
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Display the frame
                    cv2.imshow('Auto Broadcast', frame)
                    
                    # Break if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop()
                        break
                
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"Error in auto broadcast: {e}")
            self.stop()
            raise
            
    def stop(self):
        """Stop the auto broadcast system"""
        self.running = False
        self.ingester.stop_ingestion()
        cv2.destroyAllWindows()
        logger.info("Stopped auto broadcast system")

def main():
    # You can specify your RTSP URL here or use the default
    auto_broadcast = AutoBroadcast(stream_url="rtsp://your-camera-ip:port/stream")
    auto_broadcast.start()

if __name__ == "__main__":
    main()
