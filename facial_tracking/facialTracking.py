import cv2
import time
import numpy as np
import facial_tracking.conf as conf

from facial_tracking.faceMesh import FaceMesh
from facial_tracking.eye import Eye
from facial_tracking.lips import Lips

class FacialTracker:
    """
    Facial tracker with robust pixel-space EAR, adaptive eyeglasses calibration,
    scale-invariant 3D head pose estimation (Pitch, Yaw, Roll), and gaze tracking.
    """
    def __init__(self):
        self.fm = FaceMesh()
        self.left_eye = None
        self.right_eye = None
        self.lips = None
        self.left_eye_closed_frames = 0
        self.right_eye_closed_frames = 0
        self.detected = False
        self.eyes_status = 'Normal'
        self.yawn_status = 'Normal'

        # 3D Head Pose
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.head_direction = "FORWARD"

        # Baseline calibration
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0
        self.pose_calib_frames = 0

        # Adaptive EAR baseline learning
        self.ear_baseline = 0.30
        self.baseline_frames = 0
        self.current_ear = 0.30

    def process_frame(self, frame):
        """Process frame and analyze facial status and head pose."""
        self.detected = False
        self.fm.process_frame(frame)
        self.fm.draw_mesh_lips()

        if self.fm.mesh_result.multi_face_landmarks:
            self.detected = True
            for face_landmarks in self.fm.mesh_result.multi_face_landmarks:
                self.left_eye = Eye(frame, face_landmarks, conf.LEFT_EYE, is_left=True)
                self.right_eye = Eye(frame, face_landmarks, conf.RIGHT_EYE, is_left=False)
                self.lips = Lips(frame, face_landmarks, conf.LIPS)

                # Compute combined true pixel EAR
                avg_ear = (self.left_eye.ear + self.right_eye.ear) / 2.0
                self.current_ear = avg_ear

                # Adaptive baseline learning during open eye states (>0.24)
                if avg_ear > 0.24 and self.baseline_frames < 60:
                    self.ear_baseline = 0.95 * self.ear_baseline + 0.05 * avg_ear
                    self.baseline_frames += 1

                # Estimate Geometric 3D Head Pose
                self._estimate_head_pose(frame, face_landmarks)

                self._check_eyes_status()
                self._check_yawn_status()
        else:
            self.eyes_status = 'Face Not Detected'
            self.yawn_status = 'Normal'
            self.head_direction = "UNKNOWN"

    def _estimate_head_pose(self, frame, face_landmarks):
        """Estimate robust, scale-invariant 3D head pose angles (Pitch, Yaw, Roll)."""
        h, w = frame.shape[:2]
        lm = face_landmarks.landmark

        # Key landmark points:
        # 1: Nose tip, 152: Chin, 33: Right eye outer corner, 263: Left eye outer corner, 168: Midpoint between eyes
        try:
            nose = np.array([lm[1].x * w, lm[1].y * h])
            chin = np.array([lm[152].x * w, lm[152].y * h])
            r_eye = np.array([lm[33].x * w, lm[33].y * h])
            l_eye = np.array([lm[263].x * w, lm[263].y * h])
            mid_eyes = np.array([lm[168].x * w, lm[168].y * h])

            # 1. Yaw (Horizontal asymmetry: nose distance to right vs left eye)
            d_right = np.linalg.norm(nose - r_eye)
            d_left = np.linalg.norm(nose - l_eye)
            if d_right + d_left > 1e-4:
                raw_yaw = float(((d_right - d_left) / (d_right + d_left)) * 90.0)
            else:
                raw_yaw = 0.0

            # 2. Pitch (Vertical proportion: nose relative to eyes and chin)
            face_height = chin[1] - mid_eyes[1]
            if face_height > 10.0:
                rel_nose_y = (nose[1] - mid_eyes[1]) / face_height
                raw_pitch = float((0.45 - rel_nose_y) * 110.0)
            else:
                raw_pitch = 0.0

            # 3. Roll (Angle between eye corners)
            dx = l_eye[0] - r_eye[0]
            dy = l_eye[1] - r_eye[1]
            raw_roll = float(np.degrees(np.arctan2(dy, dx)))

            # Dynamic neutral baseline calibration
            if self.pose_calib_frames < 30:
                self.baseline_yaw = 0.9 * self.baseline_yaw + 0.1 * raw_yaw
                self.baseline_pitch = 0.9 * self.baseline_pitch + 0.1 * raw_pitch
                self.pose_calib_frames += 1

            self.yaw = raw_yaw - self.baseline_yaw
            self.pitch = raw_pitch - self.baseline_pitch
            self.roll = raw_roll

            # Direction classification
            if self.pitch < -16.0:
                self.head_direction = "LOOKING DOWN (LAP/PHONE)"
            elif self.pitch > 18.0:
                self.head_direction = "LOOKING UP"
            elif self.yaw < -20.0:
                self.head_direction = "LOOKING LEFT"
            elif self.yaw > 20.0:
                self.head_direction = "LOOKING RIGHT"
            else:
                self.head_direction = "FORWARD"
        except Exception:
            self.head_direction = "FORWARD"

    def _check_eyes_status(self):
        self.eyes_status = 'Normal'
        left_closed = self._left_eye_closed()
        right_closed = self._right_eye_closed()

        if left_closed:
            self.left_eye_closed_frames += 1
        else:
            self.left_eye_closed_frames = max(0, self.left_eye_closed_frames - 2)
            self.left_eye.iris.draw_iris(True)

        if right_closed:
            self.right_eye_closed_frames += 1
        else:
            self.right_eye_closed_frames = max(0, self.right_eye_closed_frames - 2)
            self.right_eye.iris.draw_iris(True)

        if self.left_eye_closed_frames >= 6 and self.right_eye_closed_frames >= 6:
            self.eyes_status = 'eye closed'
            return

        if not left_closed and not right_closed:
            if self.left_eye.gaze_right() and self.right_eye.gaze_right():
                self.eyes_status = 'gazing right'
            elif self.left_eye.gaze_left() and self.right_eye.gaze_left():
                self.eyes_status = 'gazing left'
            elif self.left_eye.gaze_center() and self.right_eye.gaze_center():
                self.eyes_status = 'gazing center'

    def _check_yawn_status(self):
        self.yawn_status = 'Normal'
        if self.lips and self.lips.mouth_open():
            self.yawn_status = 'yawning'

    def _left_eye_closed(self, threshold=6):
        """Return the instantaneous left-eye closure or its temporal status."""
        if self.left_eye is None:
            return self.left_eye_closed_frames >= threshold
        closure_thresh = max(0.16, min(0.22, self.ear_baseline * 0.65))
        return self.left_eye.eye_closed(threshold=closure_thresh)

    def _right_eye_closed(self, threshold=6):
        """Return the instantaneous right-eye closure or its temporal status."""
        if self.right_eye is None:
            return self.right_eye_closed_frames >= threshold
        closure_thresh = max(0.16, min(0.22, self.ear_baseline * 0.65))
        return self.right_eye.eye_closed(threshold=closure_thresh)
