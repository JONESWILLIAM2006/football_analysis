#!/usr/bin/env python3
"""
Demo script showing enhanced ball detection and tracking
Run this to see the ball tracking system in action
"""

import cv2
import numpy as np
from ball_tracking_enhanced import EnhancedFootballTracker, process_video_with_ball_tracking

def demo_webcam_tracking():
    """Demo ball tracking using webcam"""
    print("🚀 Starting webcam ball tracking demo...")
    print("Press 'q' to quit")
    
    tracker = EnhancedFootballTracker()
    cap = cv2.VideoCapture(0)  # Use webcam
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        players, ball = tracker.process_frame(frame)
        
        # Draw annotations
        annotated_frame = tracker.draw_annotations(frame, players, ball)
        
        # Add demo info
        cv2.putText(annotated_frame, "Enhanced Ball Tracking Demo", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if ball:
            cv2.putText(annotated_frame, f"Ball detected at: {ball['center']}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Enhanced Ball Tracking Demo', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Demo completed!")

def demo_video_processing():
    """Demo processing a video file"""
    print("📹 Video processing demo")
    
    # You can replace this with your video file path
    video_path = input("Enter video file path (or press Enter for webcam demo): ").strip()
    
    if not video_path:
        demo_webcam_tracking()
        return
    
    output_path = "demo_output_with_ball_tracking.mp4"
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    
    results = process_video_with_ball_tracking(video_path, output_path)
    
    if results:
        print("\n📊 Processing Results:")
        print(f"✅ Total frames processed: {results['total_frames']}")
        print(f"⚽ Ball detections: {len(results['ball_detections'])}")
        print(f"📈 Detection rate: {results['detection_rate']:.1f}%")
        print(f"💾 Output saved to: {output_path}")
    else:
        print("❌ Error processing video")

def create_sample_data():
    """Create sample data for testing"""
    print("📝 Creating sample tracking data...")
    
    # Simulate ball positions for demonstration
    ball_positions = []
    for i in range(100):
        # Simulate ball moving in a curved path
        x = 400 + 200 * np.sin(i * 0.1)
        y = 300 + 100 * np.cos(i * 0.1)
        ball_positions.append((int(x), int(y)))
    
    # Create a simple visualization
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)  # Green background (like a pitch)
    
    # Draw trajectory
    for i in range(1, len(ball_positions)):
        cv2.line(frame, ball_positions[i-1], ball_positions[i], (255, 255, 255), 2)
    
    # Draw ball positions with fading effect
    for i, pos in enumerate(ball_positions):
        alpha = i / len(ball_positions)
        color_intensity = int(255 * alpha)
        cv2.circle(frame, pos, 5, (color_intensity, color_intensity, 255), -1)
    
    # Add title
    cv2.putText(frame, "Sample Ball Trajectory", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    # Save sample image
    cv2.imwrite("sample_ball_trajectory.png", frame)
    print("✅ Sample trajectory saved as 'sample_ball_trajectory.png'")
    
    # Show the image
    cv2.imshow("Sample Ball Trajectory", frame)
    cv2.waitKey(3000)  # Show for 3 seconds
    cv2.destroyAllWindows()

def main():
    """Main demo function"""
    print("⚽ Enhanced Football Ball Tracking Demo")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. 📹 Webcam tracking demo")
        print("2. 🎥 Process video file")
        print("3. 📊 Create sample trajectory")
        print("4. ❌ Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            demo_webcam_tracking()
        elif choice == "2":
            demo_video_processing()
        elif choice == "3":
            create_sample_data()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()