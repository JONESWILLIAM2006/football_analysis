# Advanced pitch detection and homography mapping
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy import ndimage
import matplotlib.pyplot as plt

class AdvancedPitchDetector:
    """Advanced pitch detection using semantic segmentation and line detection"""
    
    def __init__(self, frame_width=1280, frame_height=720):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.pitch_mask = None
        self.homography_matrix = None
        self.pitch_corners_pixel = None
        
        # Standard FIFA pitch dimensions (in meters)
        self.pitch_length = 105  # meters
        self.pitch_width = 68   # meters
        
        # Real-world pitch corners (top-left, top-right, bottom-right, bottom-left)
        self.pitch_corners_world = np.array([
            [0, 0],
            [self.pitch_length, 0],
            [self.pitch_length, self.pitch_width],
            [0, self.pitch_width]
        ], dtype=np.float32)
        
        # Line detection parameters
        self.line_detection_params = {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 100,
            'min_line_length': 100,
            'max_line_gap': 10
        }
        
        # Calibration status
        self.is_calibrated = False
        self.calibration_confidence = 0.0
        
    def preprocess_frame(self, frame):
        """Preprocess frame for better line detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_field_mask(self, frame):
        """Detect field area using color segmentation"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for green field color
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green areas
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find largest contour (should be the field)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            field_mask = np.zeros_like(green_mask)
            cv2.fillPoly(field_mask, [largest_contour], 255)
            return field_mask
        
        return green_mask
    
    def detect_lines(self, frame, field_mask=None):
        """Detect pitch lines using advanced edge detection"""
        preprocessed = self.preprocess_frame(frame)
        
        # Apply field mask if available
        if field_mask is not None:
            preprocessed = cv2.bitwise_and(preprocessed, field_mask)
        
        # Edge detection
        edges = cv2.Canny(preprocessed, 
                         self.line_detection_params['canny_low'],
                         self.line_detection_params['canny_high'])
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges,
                               rho=1,
                               theta=np.pi/180,
                               threshold=self.line_detection_params['hough_threshold'],
                               minLineLength=self.line_detection_params['min_line_length'],
                               maxLineGap=self.line_detection_params['max_line_gap'])
        
        if lines is None:
            return [], []
        
        # Classify lines as horizontal or vertical
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle = abs(angle)
            
            # Classify based on angle
            if angle < 15 or angle > 165:  # Horizontal lines
                horizontal_lines.append(line[0])
            elif 75 < angle < 105:  # Vertical lines
                vertical_lines.append(line[0])
        
        return horizontal_lines, vertical_lines
    
    def cluster_lines(self, lines, axis=0):
        """Cluster parallel lines to find main field lines"""
        if not lines:
            return []
        
        # Extract line positions (y for horizontal, x for vertical)
        if axis == 0:  # Horizontal lines - cluster by y position
            positions = [(line[1] + line[3]) / 2 for line in lines]
        else:  # Vertical lines - cluster by x position
            positions = [(line[0] + line[2]) / 2 for line in lines]
        
        positions = np.array(positions).reshape(-1, 1)
        
        # Use DBSCAN to cluster lines
        clustering = DBSCAN(eps=20, min_samples=1).fit(positions)
        labels = clustering.labels_
        
        # Find representative line for each cluster
        clustered_lines = []
        for label in set(labels):
            if label == -1:  # Noise
                continue
            
            cluster_indices = np.where(labels == label)[0]
            cluster_lines = [lines[i] for i in cluster_indices]
            
            # Take the longest line in the cluster
            longest_line = max(cluster_lines, key=lambda l: np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2))
            clustered_lines.append(longest_line)
        
        return clustered_lines
    
    def find_line_intersections(self, h_lines, v_lines):
        """Find intersections between horizontal and vertical lines"""
        intersections = []
        
        for h_line in h_lines:
            for v_line in v_lines:
                intersection = self._line_intersection(h_line, v_line)
                if intersection is not None:
                    x, y = intersection
                    # Check if intersection is within frame bounds
                    if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                        intersections.append([x, y])
        
        return np.array(intersections)
    
    def _line_intersection(self, line1, line2):
        """Calculate intersection point of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
        
        return [int(px), int(py)]
    
    def find_pitch_corners(self, intersections):
        """Find the four corners of the pitch from line intersections"""
        if len(intersections) < 4:
            return None
        
        # Find corners by geometric properties
        # Top-left: minimum x + y
        # Top-right: maximum x, minimum y
        # Bottom-right: maximum x + y
        # Bottom-left: minimum x, maximum y
        
        corners = []
        
        # Top-left corner
        tl_scores = intersections[:, 0] + intersections[:, 1]
        tl_idx = np.argmin(tl_scores)
        corners.append(intersections[tl_idx])
        
        # Bottom-right corner
        br_scores = intersections[:, 0] + intersections[:, 1]
        br_idx = np.argmax(br_scores)
        corners.append(intersections[br_idx])
        
        # Top-right corner
        tr_scores = intersections[:, 0] - intersections[:, 1]
        tr_idx = np.argmax(tr_scores)
        corners.append(intersections[tr_idx])
        
        # Bottom-left corner
        bl_scores = intersections[:, 0] - intersections[:, 1]
        bl_idx = np.argmin(bl_scores)
        corners.append(intersections[bl_idx])
        
        # Remove duplicates and sort
        corners = np.array(corners)
        unique_corners = []
        for corner in corners:
            is_duplicate = False
            for existing in unique_corners:
                if np.linalg.norm(corner - existing) < 50:  # 50 pixel threshold
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_corners.append(corner)
        
        if len(unique_corners) < 4:
            return None
        
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        unique_corners = np.array(unique_corners)
        
        # Sort by y-coordinate first
        sorted_by_y = unique_corners[np.argsort(unique_corners[:, 1])]
        
        # Top two points
        top_points = sorted_by_y[:2]
        top_points = top_points[np.argsort(top_points[:, 0])]  # Sort by x
        
        # Bottom two points
        bottom_points = sorted_by_y[2:4]
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]  # Sort by x
        
        # Arrange as [top-left, top-right, bottom-right, bottom-left]
        ordered_corners = np.array([
            top_points[0],      # top-left
            top_points[1],      # top-right
            bottom_points[1],   # bottom-right
            bottom_points[0]    # bottom-left
        ])
        
        return ordered_corners.astype(np.float32)
    
    def calculate_homography(self, pixel_corners):
        """Calculate homography matrix from pixel corners to world coordinates"""
        if pixel_corners is None or len(pixel_corners) != 4:
            return None
        
        try:
            # Calculate homography matrix
            homography_matrix = cv2.getPerspectiveTransform(pixel_corners, self.pitch_corners_world)
            
            # Validate homography by checking if it produces reasonable results
            test_points = np.array([[self.frame_width//2, self.frame_height//2]], dtype=np.float32)
            test_points = test_points.reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(test_points, homography_matrix)
            
            # Check if center point maps to reasonable field coordinates
            center_x, center_y = transformed[0][0]
            if 0 <= center_x <= self.pitch_length and 0 <= center_y <= self.pitch_width:
                return homography_matrix
            
        except Exception as e:
            print(f"Homography calculation failed: {e}")
        
        return None
    
    def auto_calibrate(self, frame):
        """Automatically calibrate pitch detection from a single frame"""
        # Step 1: Detect field area
        field_mask = self.detect_field_mask(frame)
        
        # Step 2: Detect lines
        h_lines, v_lines = self.detect_lines(frame, field_mask)
        
        # Step 3: Cluster lines to find main field boundaries
        main_h_lines = self.cluster_lines(h_lines, axis=0)
        main_v_lines = self.cluster_lines(v_lines, axis=1)
        
        if len(main_h_lines) < 2 or len(main_v_lines) < 2:
            return False
        
        # Step 4: Find intersections
        intersections = self.find_line_intersections(main_h_lines, main_v_lines)
        
        if len(intersections) < 4:
            return False
        
        # Step 5: Find pitch corners
        corners = self.find_pitch_corners(intersections)
        
        if corners is None:
            return False
        
        # Step 6: Calculate homography
        homography = self.calculate_homography(corners)
        
        if homography is None:
            return False
        
        # Store results
        self.pitch_corners_pixel = corners
        self.homography_matrix = homography
        self.pitch_mask = field_mask
        self.is_calibrated = True
        
        # Calculate calibration confidence based on line quality
        self.calibration_confidence = min(1.0, (len(main_h_lines) + len(main_v_lines)) / 8.0)
        
        return True
    
    def pixel_to_world(self, pixel_points):
        """Convert pixel coordinates to world coordinates"""
        if self.homography_matrix is None:
            return None
        
        if isinstance(pixel_points, (list, tuple)):
            pixel_points = np.array([pixel_points], dtype=np.float32)
        elif len(pixel_points.shape) == 1:
            pixel_points = pixel_points.reshape(1, -1)
        
        pixel_points = pixel_points.astype(np.float32).reshape(-1, 1, 2)
        
        try:
            world_points = cv2.perspectiveTransform(pixel_points, self.homography_matrix)
            return world_points.reshape(-1, 2)
        except Exception as e:
            print(f"Coordinate transformation failed: {e}")
            return None
    
    def world_to_pixel(self, world_points):
        """Convert world coordinates to pixel coordinates"""
        if self.homography_matrix is None:
            return None
        
        if isinstance(world_points, (list, tuple)):
            world_points = np.array([world_points], dtype=np.float32)
        elif len(world_points.shape) == 1:
            world_points = world_points.reshape(1, -1)
        
        world_points = world_points.astype(np.float32).reshape(-1, 1, 2)
        
        try:
            inv_homography = np.linalg.inv(self.homography_matrix)
            pixel_points = cv2.perspectiveTransform(world_points, inv_homography)
            return pixel_points.reshape(-1, 2)
        except Exception as e:
            print(f"Inverse coordinate transformation failed: {e}")
            return None
    
    def is_inside_pitch(self, world_points):
        """Check if world coordinates are inside the pitch"""
        if isinstance(world_points, (list, tuple)):
            world_points = np.array([world_points])
        
        inside = []
        for point in world_points:
            x, y = point
            inside.append(0 <= x <= self.pitch_length and 0 <= y <= self.pitch_width)
        
        return inside if len(inside) > 1 else inside[0]
    
    def filter_detections(self, detections):
        """Filter detections to only include those inside the pitch"""
        if not self.is_calibrated:
            return detections
        
        filtered = []
        for detection in detections:
            # Assume detection has 'center' or 'bbox' attribute
            if 'center' in detection:
                pixel_point = detection['center']
            elif 'bbox' in detection:
                x1, y1, x2, y2 = detection['bbox']
                pixel_point = [(x1 + x2) / 2, (y1 + y2) / 2]
            else:
                continue
            
            world_point = self.pixel_to_world([pixel_point])
            if world_point is not None and self.is_inside_pitch(world_point[0]):
                filtered.append(detection)
        
        return filtered
    
    def visualize_calibration(self, frame):
        """Visualize the calibration results on the frame"""
        vis_frame = frame.copy()
        
        if self.pitch_corners_pixel is not None:
            # Draw pitch corners
            for i, corner in enumerate(self.pitch_corners_pixel):
                cv2.circle(vis_frame, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
                cv2.putText(vis_frame, f'C{i+1}', 
                           (int(corner[0]) + 15, int(corner[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw pitch boundary
            cv2.polylines(vis_frame, [self.pitch_corners_pixel.astype(int)], 
                         True, (0, 255, 0), 3)
        
        # Draw field mask
        if self.pitch_mask is not None:
            mask_overlay = cv2.applyColorMap(self.pitch_mask, cv2.COLORMAP_GREEN)
            vis_frame = cv2.addWeighted(vis_frame, 0.8, mask_overlay, 0.2, 0)
        
        # Add calibration status
        status_text = f"Calibrated: {self.is_calibrated}"
        confidence_text = f"Confidence: {self.calibration_confidence:.2f}"
        
        cv2.putText(vis_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if self.is_calibrated else (0, 0, 255), 2)
        cv2.putText(vis_frame, confidence_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return vis_frame
    
    def get_pitch_zones(self):
        """Get predefined pitch zones in world coordinates"""
        zones = {
            'defensive_third': [(0, 0), (35, 68)],
            'middle_third': [(35, 0), (70, 68)],
            'attacking_third': [(70, 0), (105, 68)],
            'penalty_area_left': [(0, 13.84), (16.5, 54.16)],
            'penalty_area_right': [(88.5, 13.84), (105, 54.16)],
            'goal_area_left': [(0, 24.84), (5.5, 43.16)],
            'goal_area_right': [(99.5, 24.84), (105, 43.16)],
            'center_circle': [(52.5, 34), (52.5, 34)]  # Center point and radius
        }
        return zones
    
    def get_zone_for_position(self, world_position):
        """Determine which zone a world position belongs to"""
        x, y = world_position
        
        # Defensive/Middle/Attacking thirds
        if x < 35:
            third = "defensive_third"
        elif x < 70:
            third = "middle_third"
        else:
            third = "attacking_third"
        
        # Penalty areas
        if 0 <= x <= 16.5 and 13.84 <= y <= 54.16:
            return "penalty_area_left"
        elif 88.5 <= x <= 105 and 13.84 <= y <= 54.16:
            return "penalty_area_right"
        
        # Goal areas
        if 0 <= x <= 5.5 and 24.84 <= y <= 43.16:
            return "goal_area_left"
        elif 99.5 <= x <= 105 and 24.84 <= y <= 43.16:
            return "goal_area_right"
        
        # Center circle (approximate)
        center_dist = np.sqrt((x - 52.5)**2 + (y - 34)**2)
        if center_dist <= 9.15:  # Center circle radius
            return "center_circle"
        
        return third