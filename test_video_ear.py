import cv2
import numpy as np
import mediapipe as mp
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Find video
vid_path = "scratch/sample_driver_test.mp4"
if not os.path.exists(vid_path):
    uploads = [os.path.join("uploads", f) for f in os.listdir("uploads") if f.endswith(".mp4")]
    if uploads:
        vid_path = uploads[0]

cap = cv2.VideoCapture(vid_path)
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if results.multi_face_landmarks:
        for lm in results.multi_face_landmarks:
            r_indices = [33, 160, 158, 133, 153, 144]
            pts_norm = [np.array([lm.landmark[idx].x, lm.landmark[idx].y]) for idx in r_indices]
            pts_pixel = [np.array([lm.landmark[idx].x * w, lm.landmark[idx].y * h]) for idx in r_indices]
            
            # Normalized EAR
            v1_norm = np.linalg.norm(pts_norm[1] - pts_norm[5])
            v2_norm = np.linalg.norm(pts_norm[2] - pts_norm[4])
            h_norm = np.linalg.norm(pts_norm[0] - pts_norm[3])
            ear_norm = (v1_norm + v2_norm) / (2.0 * h_norm)
            
            # Pixel EAR
            v1_px = np.linalg.norm(pts_pixel[1] - pts_pixel[5])
            v2_px = np.linalg.norm(pts_pixel[2] - pts_pixel[4])
            h_px = np.linalg.norm(pts_pixel[0] - pts_pixel[3])
            ear_pixel = (v1_px + v2_px) / (2.0 * h_px)
            
            print(f"Video Frame {i}: Normalized EAR={ear_norm:.3f} | Pixel EAR={ear_pixel:.3f}")
cap.release()
