import cv2
import math
import numpy as np
import facial_tracking.conf as conf
from facial_tracking.iris import Iris

class Eye:
    """
    Eye object computing 6-point pixel-space 3D Euclidean EAR and iris gaze with adaptive calibration.
    """
    # 6 landmark indices for standard Euclidean EAR calculation
    # Right Eye: [P1:33 (outer), P2:160 (top-outer), P3:158 (top-inner), P4:133 (inner), P5:153 (bot-inner), P6:144 (bot-outer)]
    # Left Eye:  [P1:263 (outer), P2:387 (top-outer), P3:385 (top-inner), P4:362 (inner), P5:380 (bot-inner), P6:373 (bot-outer)]
    EAR_INDICES_RIGHT = [33, 160, 158, 133, 153, 144]
    EAR_INDICES_LEFT = [263, 387, 385, 362, 380, 373]

    def __init__(self, frame, face_landmarks, eye_id, is_left=True):
        self.frame = frame
        self.face_landmarks = face_landmarks
        self.eye_id = eye_id
        self.is_left = is_left

        self.iris = Iris(frame, face_landmarks, eye_id)
        self.pos = self._get_eye_pos()
        self.ear_indices = self.EAR_INDICES_LEFT if is_left else self.EAR_INDICES_RIGHT

        self.ear = self._compute_pixel_ear()
        # Backwards-compatible name used by the original test and UI code.
        self.eye_veti_to_hori = self.ear
        self.iris_relative_to_eye = self._get_gaze_ratio()

    def _get_eye_pos(self):
        """Get the outer, inner, top, and bottom positions of eye."""
        h, w = self.frame.shape[:2]
        eye_pos = []
        for id in self.eye_id[:4]:
            pos = self.face_landmarks.landmark[id]
            cx = int(pos.x * w)
            cy = int(pos.y * h)
            eye_pos.append([cx, cy])
        return eye_pos

    def _compute_pixel_ear(self):
        """Compute standard 6-point Euclidean EAR in real pixel coordinates."""
        h, w = self.frame.shape[:2]
        pts = []
        for idx in self.ear_indices:
            lm = self.face_landmarks.landmark[idx]
            # Convert normalized coordinates into true pixel dimensions to maintain accurate aspect ratio
            pts.append(np.array([lm.x * w, lm.y * h]))

        # Vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        h_dist = np.linalg.norm(pts[0] - pts[3])

        if h_dist < 1e-4:
            return 0.30
        return (v1 + v2) / (2.0 * h_dist)

    def _get_gaze_ratio(self):
        """Get the ratio of iris relative to eye corners."""
        try:
            denom = (self.pos[0][0] - self.pos[1][0])
            if abs(denom) < 1e-4:
                return [0.5, 0.5, 0.5]
            ratiol = (self.pos[0][0] - self.iris.pos[1][0]) / denom
            ratioc = (self.pos[0][0] - self.iris.pos[0][0]) / denom
            ratior = (self.pos[0][0] - self.iris.pos[3][0]) / denom
            return [ratiol, ratioc, ratior]
        except Exception:
            return [0.5, 0.5, 0.5]

    def gaze_left(self, threshold=conf.GAZE_LEFT):
        return self.iris_relative_to_eye[0] < threshold

    def gaze_right(self, threshold=conf.GAZE_RIGHT):
        return self.iris_relative_to_eye[2] > threshold

    def gaze_center(self):
        return not self.gaze_left() and not self.gaze_right()

    def eye_closed(self, threshold=0.18):
        """True eye closure check (open eyes are ~0.28-0.35, closed eyes are <0.18)."""
        return self.ear < threshold

    def draw_eye(self):
        """Draw the target landmarks of eye."""
        for pos in self.pos:
            cv2.circle(self.frame, tuple(pos), 2, conf.LM_COLOR, -1, lineType=cv2.LINE_AA)
