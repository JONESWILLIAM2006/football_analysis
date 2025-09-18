#!/usr/bin/env python3

import os
import sys
import time
import cv2
import argparse
from football_analysis import EnhancedMatchAnalysis

def parse_arguments():
    parser = argparse.ArgumentParser(description='Football Analysis System')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    return parser.parse_args()

def validate_video_file(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    return True

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate video file
    if not validate_video_file(args.video):
        sys.exit(1)
        
    video_path = args.video
    print(f"Starting analysis of video: {video_path}")
    
    # Initialize the analysis system
    analyzer = EnhancedMatchAnalysis()
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Generate output filename
    output_filename = f"output_video_{int(time.time())}.mp4"
    output_path = os.path.join("outputs", output_filename)
    
    try:
        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Process the video with progress reporting
        print(f"\nProcessing video: {video_path}")
        print(f"Total frames to process: {total_frames}")
        print("\nProgress:")
        
        # Process the video in chunks
        chunk_size = 100  # Process 100 frames at a time
        for start_frame in range(0, total_frames, chunk_size):
            end_frame = min(start_frame + chunk_size, total_frames)
            progress = (end_frame / total_frames) * 100
            print(f"\rProgress: {progress:.1f}% ({end_frame}/{total_frames} frames)", end="")
            
            # Process this chunk
            results = analyzer.analyze_video_with_ball_tracking(
                video_path, 
                output_path,
                start_frame=start_frame,
                end_frame=end_frame
            )
            
            # Save intermediate results if needed
            if results and results.get('ball_events'):
                print(f"\nDetected {len(results['ball_events'])} ball events in frames {start_frame}-{end_frame}")
        
        print("\n\nAnalysis completed successfully!")
        print(f"Output video saved to: {output_path}")
        if results:
            print("\nAnalysis Results:")
            print(f"Total Frames Processed: {results['total_frames']}")
            print(f"Ball Events Detected: {len(results['ball_events'])}")
            print("Possession Data:", results['possession_data'])
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        # Cleanup any temporary files if needed
        if os.path.exists(output_path):
            os.remove(output_path)
        sys.exit(0)
    except Exception as e:
        print(f"\nError during video analysis: {str(e)}")
        # Cleanup any temporary files if needed
        if os.path.exists(output_path):
            os.remove(output_path)
        sys.exit(1)

if __name__ == "__main__":
    main()
