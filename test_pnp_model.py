import cv2
import numpy as np

def estimate_head_pose_test():
    # Standard 3D generic facial model points (in mm/normalized unit space)
    # Nose tip, Chin, Left Eye Corner, Right Eye Corner, Left Mouth Corner, Right Mouth Corner
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (index 1)
        (0.0, -330.0, -65.0),        # Chin (index 152)
        (-225.0, 170.0, -135.0),     # Left eye left corner (index 33)
        (225.0, 170.0, -135.0),      # Right eye right corner (index 263)
        (-150.0, -150.0, -125.0),    # Left Mouth corner (index 61)
        (150.0, -150.0, -125.0)      # Right mouth corner (index 291)
    ], dtype=np.float64)

    w, h = 640, 480
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Simulated frontal face 2D image points
    image_points = np.array([
        (320, 240), # Nose tip
        (320, 350), # Chin
        (260, 190), # Left eye
        (380, 190), # Right eye
        (280, 300), # Left mouth
        (360, 300)  # Right mouth
    ], dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    print(f"Frontal face -> Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°, Roll: {roll:.1f}°")

estimate_head_pose_test()
