# Advanced Pitch Detection & Homography
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class PitchHomographyDetector:
    def __init__(self, frame_width=1280, frame_height=720):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.homography_matrix = None
        self.pitch_corners = None
        
        # FIFA standard pitch dimensions (meters)
        self.pitch_world = np.array([
            [0, 0], [105, 0], [105, 68], [0, 68]
        ], dtype=np.float32)
    
    def detect_field_lines(self, frame):
        # Convert to grayscale and enhance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        
        # Detect edges
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return [], []
        
        # Classify lines
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            
            if abs(angle) < 15 or abs(angle) > 165:
                h_lines.append(line[0])
            elif 75 < abs(angle) < 105:
                v_lines.append(line[0])
        
        return h_lines, v_lines
    
    def find_pitch_corners(self, h_lines, v_lines):
        if len(h_lines) < 2 or len(v_lines) < 2:
            return None
        
        # Find intersections
        intersections = []
        for h_line in h_lines[:2]:  # Top 2 horizontal lines
            for v_line in v_lines[:2]:  # Top 2 vertical lines
                intersection = self._line_intersection(h_line, v_line)
                if intersection and self._is_valid_point(intersection):
                    intersections.append(intersection)
        
        if len(intersections) < 4:
            return None
        
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        intersections = np.array(intersections)
        
        # Find corners by position
        center = np.mean(intersections, axis=0)
        
        corners = []
        # Top-left: min x+y
        tl_idx = np.argmin(intersections[:, 0] + intersections[:, 1])
        corners.append(intersections[tl_idx])
        
        # Top-right: max x, min y
        tr_idx = np.argmax(intersections[:, 0] - intersections[:, 1])
        corners.append(intersections[tr_idx])
        
        # Bottom-right: max x+y
        br_idx = np.argmax(intersections[:, 0] + intersections[:, 1])
        corners.append(intersections[br_idx])
        
        # Bottom-left: min x, max y
        bl_idx = np.argmin(intersections[:, 0] - intersections[:, 1])
        corners.append(intersections[bl_idx])
        
        return np.array(corners, dtype=np.float32)
    
    def _line_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
        
        px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
        py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
        
        return [int(px), int(py)]
    
    def _is_valid_point(self, point):
        x, y = point
        return 0 <= x < self.frame_width and 0 <= y < self.frame_height
    
    def calibrate_homography(self, frame):
        h_lines, v_lines = self.detect_field_lines(frame)
        corners = self.find_pitch_corners(h_lines, v_lines)
        
        if corners is None:
            return False
        
        try:
            self.homography_matrix = cv2.getPerspectiveTransform(corners, self.pitch_world)
            self.pitch_corners = corners
            return True
        except:
            return False
    
    def pixel_to_world(self, pixel_points):
        if self.homography_matrix is None:
            return None
        
        if isinstance(pixel_points, (list, tuple)):
            pixel_points = np.array([pixel_points], dtype=np.float32)
        
        pixel_points = pixel_points.reshape(-1, 1, 2)
        world_points = cv2.perspectiveTransform(pixel_points, self.homography_matrix)
        return world_points.reshape(-1, 2)
    
    def world_to_pixel(self, world_points):
        if self.homography_matrix is None:
            return None
        
        if isinstance(world_points, (list, tuple)):
            world_points = np.array([world_points], dtype=np.float32)
        
        world_points = world_points.reshape(-1, 1, 2)
        inv_h = np.linalg.inv(self.homography_matrix)
        pixel_points = cv2.perspectiveTransform(world_points, inv_h)
        return pixel_points.reshape(-1, 2)
    
    def get_pitch_zones(self):
        return {
            'defensive_third': [(0, 0), (35, 68)],
            'middle_third': [(35, 0), (70, 68)],
            'attacking_third': [(70, 0), (105, 68)],
            'penalty_left': [(0, 13.84), (16.5, 54.16)],
            'penalty_right': [(88.5, 13.84), (105, 54.16)]
        }
    
    def get_zone_for_position(self, world_pos):
        x, y = world_pos
        
        if x < 35:
            return "defensive_third"
        elif x < 70:
            return "middle_third"
        else:
            return "attacking_third"