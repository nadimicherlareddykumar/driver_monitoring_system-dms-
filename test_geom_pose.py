import numpy as np

def compute_geometric_head_pose(landmarks, w, h):
    # Key landmark points:
    # 1: Nose tip
    # 152: Chin
    # 33: Right eye outer corner (subject's right)
    # 263: Left eye outer corner (subject's left)
    # 168: Midpoint between eyes (glabella)
    
    nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    r_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    l_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    mid_eyes = np.array([landmarks[168].x * w, landmarks[168].y * h])
    
    # 1. Yaw (Horizontal ratio: nose to right eye vs nose to left eye)
    d_right = np.linalg.norm(nose - r_eye)
    d_left = np.linalg.norm(nose - l_eye)
    
    if d_right + d_left > 1e-4:
        # Symmetrical when looking straight: (d_right - d_left) / (d_right + d_left) = 0.0
        yaw_norm = (d_right - d_left) / (d_right + d_left)
        yaw_deg = float(yaw_norm * 90.0) # Maps roughly to -45 to +45 degrees
    else:
        yaw_deg = 0.0

    # 2. Pitch (Vertical position of nose relative to eyes and chin)
    face_height = chin[1] - mid_eyes[1]
    if face_height > 10.0:
        rel_nose_y = (nose[1] - mid_eyes[1]) / face_height
        # At neutral forward gaze, nose is around 0.45 of face height from eyes to chin
        pitch_norm = 0.45 - rel_nose_y
        pitch_deg = float(pitch_norm * 110.0) # Looking down produces negative pitch, looking up produces positive pitch
    else:
        pitch_deg = 0.0
        
    # 3. Roll (Angle between eye corners)
    dx = l_eye[0] - r_eye[0]
    dy = l_eye[1] - r_eye[1]
    roll_deg = float(np.degrees(np.arctan2(dy, dx)))
    
    return pitch_deg, yaw_deg, roll_deg

# Test neutral face
class DummyPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

dummy_landmarks = {
    1: DummyPoint(0.50, 0.50),   # Nose
    152: DummyPoint(0.50, 0.70), # Chin
    33: DummyPoint(0.40, 0.40),  # Right eye
    263: DummyPoint(0.60, 0.40), # Left eye
    168: DummyPoint(0.50, 0.38)  # Mid eyes
}

pitch, yaw, roll = compute_geometric_head_pose(dummy_landmarks, 640, 480)
print(f"Neutral face test -> Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°, Roll: {roll:.1f}°")

# Test looking left (nose closer to left eye)
dummy_landmarks_left = {
    1: DummyPoint(0.44, 0.50),
    152: DummyPoint(0.46, 0.70),
    33: DummyPoint(0.40, 0.40),
    263: DummyPoint(0.58, 0.40),
    168: DummyPoint(0.48, 0.38)
}
pitch_l, yaw_l, roll_l = compute_geometric_head_pose(dummy_landmarks_left, 640, 480)
print(f"Looking left test -> Pitch: {pitch_l:.1f}°, Yaw: {yaw_l:.1f}°, Roll: {roll_l:.1f}°")
