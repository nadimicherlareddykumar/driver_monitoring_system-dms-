import cv2
import numpy as np

def estimate_head_pose(landmarks, frame_shape):
    h, w = frame_shape[:2]
    # 2D image points from MediaPipe
    # 1: Nose tip, 152: Chin, 33: Left eye outer, 263: Right eye outer, 61: Left mouth, 291: Right mouth
    face_2d = []
    face_3d = []
    
    key_indices = [1, 152, 33, 263, 61, 291]
    for idx in key_indices:
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z * w])
        
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    # Camera matrix
    focal_length = 1.0 * w
    cam_matrix = np.array([[focal_length, 0, w / 2],
                           [0, focal_length, h / 2],
                           [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360
    
    return pitch, yaw, roll

print("Head pose estimator function ready!")
