# Enhanced tracking with ByteTrack and improved ball detection
import numpy as np
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import torch

class ByteTracker:
    """Simplified ByteTrack implementation for better ID consistency"""
    def __init__(self, frame_rate=30, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        
    def update(self, output_results):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        if output_results is not None:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            
            # Separate high and low confidence detections
            remain_inds = scores > self.track_thresh
            inds_low = scores > 0.1
            inds_high = scores < self.track_thresh
            
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            scores_second = scores[inds_second]
        else:
            dets = []
            scores_keep = []
            
        # Create STrack objects
        if len(dets) > 0:
            detections = [STrack(bbox, score) for bbox, score in zip(dets, scores_keep)]
        else:
            detections = []
            
        # Update tracked objects
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # Associate with high score detection boxes
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # Handle unmatched detections
        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        # Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        
        # Initialize new tracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.frame_id)
            activated_starcks.append(track)
        
        # Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        
        return [track for track in self.tracked_stracks if track.is_activated]

class STrack:
    """Single object track"""
    shared_kalman = None
    track_id_count = 1
    
    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = score
        self.tracklet_len = 0
        self.state = TrackState.New
        
        self.history = deque(maxlen=30)
        
    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
    
    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.kalman_filter = KalmanFilter()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        
        self.score = new_track.score
        self.history.append(new_tlwh)
    
    @staticmethod
    def next_id():
        STrack.track_id_count += 1
        return STrack.track_id_count
    
    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
                st.mean, st.covariance = st.kalman_filter.predict(multi_mean[i], multi_covariance[i])

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class EnhancedBallDetector:
    """Specialized ball detector with motion blur handling"""
    def __init__(self, model_path="yolov8x.pt"):
        self.model = YOLO(model_path)
        self.ball_class_id = self._get_ball_class_id()
        
        # Ball-specific parameters
        self.min_ball_size = 5
        self.max_ball_size = 50
        self.confidence_threshold = 0.1  # Lower for small balls
        
        # Motion blur detection
        self.prev_frame = None
        self.motion_threshold = 30
        
        # Temporal consistency
        self.ball_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        
    def _get_ball_class_id(self):
        class_names = self.model.names
        for class_id, name in class_names.items():
            if 'ball' in name.lower() or 'sports ball' in name.lower():
                return class_id
        return 37
    
    def detect_motion_blur(self, frame):
        """Detect motion blur using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def enhance_ball_detection(self, frame):
        """Enhanced ball detection with preprocessing"""
        # Motion blur detection
        blur_score = self.detect_motion_blur(frame)
        
        # Preprocessing for better ball detection
        enhanced_frame = frame.copy()
        
        if blur_score < self.motion_threshold:
            # Apply sharpening for blurry frames
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced_frame = cv2.filter2D(enhanced_frame, -1, kernel)
        
        # Run detection
        results = self.model(enhanced_frame, verbose=False)
        
        ball_detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            for box, score, cls_id in zip(boxes, scores, class_ids):
                if int(cls_id) == self.ball_class_id and score > self.confidence_threshold:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    
                    # Size filtering
                    if self.min_ball_size <= max(w, h) <= self.max_ball_size:
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        ball_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'center': (cx, cy),
                            'confidence': score,
                            'size': max(w, h)
                        })
        
        # Temporal filtering
        return self._temporal_filter(ball_detections)
    
    def _temporal_filter(self, detections):
        """Filter detections using temporal consistency"""
        if not detections:
            return None
        
        # If we have history, prefer detections close to predicted position
        if self.ball_history:
            last_pos = self.ball_history[-1]['center']
            
            # Predict next position using velocity
            if len(self.velocity_history) > 0:
                avg_velocity = np.mean(self.velocity_history, axis=0)
                predicted_pos = (last_pos[0] + avg_velocity[0], last_pos[1] + avg_velocity[1])
            else:
                predicted_pos = last_pos
            
            # Find closest detection to prediction
            best_detection = min(detections, 
                               key=lambda d: np.linalg.norm(np.array(d['center']) - np.array(predicted_pos)))
            
            # Update velocity history
            velocity = np.array(best_detection['center']) - np.array(last_pos)
            self.velocity_history.append(velocity)
        else:
            # No history, take highest confidence
            best_detection = max(detections, key=lambda d: d['confidence'])
        
        # Update history
        self.ball_history.append(best_detection)
        return best_detection

class MultiObjectTracker:
    """Enhanced multi-object tracker with separate trackers for different object types"""
    def __init__(self):
        self.player_tracker = ByteTracker(track_thresh=0.6, match_thresh=0.8)
        self.referee_tracker = ByteTracker(track_thresh=0.5, match_thresh=0.7)
        self.ball_detector = EnhancedBallDetector()
        
        # Appearance embeddings (simplified)
        self.player_embeddings = {}
        self.referee_embeddings = {}
        
    def extract_appearance_features(self, frame, bbox):
        """Extract simple appearance features"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return np.zeros(64)
        
        # Simple color histogram as appearance feature
        crop_resized = cv2.resize(crop, (32, 32))
        hist_b = cv2.calcHist([crop_resized], [0], None, [8], [0, 256])
        hist_g = cv2.calcHist([crop_resized], [1], None, [8], [0, 256])
        hist_r = cv2.calcHist([crop_resized], [2], None, [8], [0, 256])
        
        features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        return features / (np.linalg.norm(features) + 1e-6)
    
    def update(self, frame, detections):
        """Update all trackers"""
        player_dets = []
        referee_dets = []
        
        # Separate detections by type
        for det in detections:
            if det['class'] == 'person':
                if det.get('is_referee', False):
                    referee_dets.append(det)
                else:
                    player_dets.append(det)
        
        # Update trackers
        player_tracks = self.player_tracker.update(
            np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['confidence']] 
                     for d in player_dets]) if player_dets else None
        )
        
        referee_tracks = self.referee_tracker.update(
            np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['confidence']] 
                     for d in referee_dets]) if referee_dets else None
        )
        
        # Ball detection
        ball_detection = self.ball_detector.enhance_ball_detection(frame)
        
        return {
            'players': player_tracks,
            'referees': referee_tracks,
            'ball': ball_detection
        }

# Utility functions for ByteTracker
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

# Simplified matching and Kalman filter classes
class matching:
    @staticmethod
    def iou_distance(atracks, btracks):
        """Compute IoU distance matrix"""
        if len(atracks) == 0 or len(btracks) == 0:
            return np.zeros((len(atracks), len(btracks)))
        
        cost_matrix = np.zeros((len(atracks), len(btracks)))
        for i, atrack in enumerate(atracks):
            for j, btrack in enumerate(btracks):
                cost_matrix[i, j] = 1 - bbox_iou(atrack.tlwh, btrack.tlwh)
        return cost_matrix
    
    @staticmethod
    def linear_assignment(cost_matrix, thresh):
        """Simple linear assignment"""
        matches, unmatched_a, unmatched_b = [], [], []
        
        if cost_matrix.size == 0:
            return matches, list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        # Simple greedy assignment
        used_rows, used_cols = set(), set()
        
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] < thresh and i not in used_rows and j not in used_cols:
                    matches.append([i, j])
                    used_rows.add(i)
                    used_cols.add(j)
        
        unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in used_rows]
        unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in used_cols]
        
        return matches, unmatched_a, unmatched_b

def bbox_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to x1, y1, x2, y2 format
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2
    
    # Calculate intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

class KalmanFilter:
    """Simplified Kalman filter for tracking"""
    def __init__(self):
        ndim, dt = 4, 1.
        
        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, measurement):
        """Create track from unassociated measurement"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step"""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step"""
        projected_mean, projected_cov = self.project(mean, covariance)
        
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
    
    def project(self, mean, covariance):
        """Project state distribution to measurement space"""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

# Add scipy import for Kalman filter
try:
    import scipy.linalg
except ImportError:
    print("Warning: scipy not available, using simplified Kalman filter")
    
    class KalmanFilter:
        def __init__(self):
            pass
        def initiate(self, measurement):
            return measurement, np.eye(len(measurement))
        def predict(self, mean, covariance):
            return mean, covariance
        def update(self, mean, covariance, measurement):
            return measurement, covariance
        def project(self, mean, covariance):
            return mean, covariance