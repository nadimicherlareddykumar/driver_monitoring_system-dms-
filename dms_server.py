import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from flask import Flask, render_template_string, Response, jsonify, request, send_from_directory, make_response
from werkzeug.utils import secure_filename
import numpy as np
import torch
import time
import threading
import glob
import io
import csv
import uuid
from collections import deque
from datetime import datetime

from dms_utils.dms_utils import ACTIONS
from net import MobileNet
from facial_tracking.facialTracking import FacialTracker
import facial_tracking.conf as conf
from dms_server.auth import require_api_token
from dms_server.api import api_bp

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('DMS_MAX_UPLOAD_BYTES', 1024 * 1024 * 1024))
app.config['DMS_STATE'] = globals()
app.register_blueprint(api_bp)

base_dir = os.path.dirname(os.path.abspath(__file__))
uploads_dir = os.path.join(base_dir, 'uploads')
incidents_dir = os.path.join(base_dir, 'incidents')
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(incidents_dir, exist_ok=True)

video_source_mode = 'webcam'  # 'webcam' or 'video'
ai_engine_mode = 'standard'
night_vision_mode = 'auto'    # 'auto', 'on', 'off'
selected_camera_index = 0     # 0 = Laptop Camera, 1 = External USB Webcam
uploaded_video_path = None
uploaded_video_name = None
video_analysis_complete = False
safety_scores_history = []

# Speed & Dynamic Sensitivity (km/h)
current_vehicle_speed = 60  # Default 60 km/h

# Global camera capture handle & lock
global_cam_cap = None
active_cap_index = None
cam_lock = threading.Lock()
last_cam_check_time = 0.0
cam_consecutive_failures = 0

# 60-second sliding window buffer for Euro NCAP PERCLOS
perclos_window = deque()
perclos_lock = threading.Lock()

# Blackbox Recorder state
last_blackbox_capture_time = 0.0
blackbox_incidents_list = []

# Telematics Session Log for CSV Export
telematics_log_buffer = []
telematics_lock = threading.Lock()
last_log_record_time = 0.0

# Medical Incapacitation / Slump State Tracking
driver_slump_duration = 0.0
prev_medical_emergency = False

# State tracking for counts & temporal filters
prev_yawn = False
prev_drowsy = False
prev_distraction = False
prev_occlusion = False
prev_perclos_hazard = False
std_no_face_duration = 0.0
std_head_diverted_duration = 0.0
std_phone_duration = 0.0
std_eyes_closed_duration = 0.0

# Pre-created CLAHE object for low-light night-vision enhancement
clahe_enhancer = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

telemetry = {
    'engine': 'standard',
    'eyes_status': 'Normal',
    'yawn_status': 'Normal',
    'action': 'Normal',
    'safety_score': 100,
    'risk_level': 'SAFE',
    'fps': 30,
    'speed_kmh': 60,
    'speed_profile': 'URBAN (NORMAL)',
    'grace_period': 1.5,
    'blind_distance_m': 0.0,
    'camera_index': 0,
    'camera_label': 'LAPTOP CAM (0)',
    'night_vision_active': False,
    'night_vision_mode': 'auto',
    'ambient_lum': 120.0,
    'face_detected': True,
    'pitch': 0.0,
    'yaw': 0.0,
    'roll': 0.0,
    'perclos': 0.0,
    'perclos_level': 'OPTIMAL (NORMAL)',
    'sleep_prob': 0.0,
    'phone_prob': 0.0,
    'sunglasses_prob': 0.0,
    'left_blink': 0.0,
    'right_blink': 0.0,
    'total_yawns': 0,
    'total_drowsy': 0,
    'total_distractions': 0,
    'total_blackbox_records': 0,
    'medical_emergency': False,
    'take_a_break': False,
    'session_start': time.time(),
    'video_mode': 'webcam',
    'video_name': 'Live Webcam',
    'video_complete': False,
    'event_log': []
}

def apply_night_vision_clahe(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe_enhancer.apply(l)
        merged = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return image

def get_speed_thresholds(speed_kmh):
    if speed_kmh <= 30:
        return 2.5, 'PARKING / LOW SPEED'
    elif speed_kmh <= 80:
        return 1.5, 'URBAN (STANDARD)'
    else:
        return 0.8, 'HIGHWAY (STRICT)'

def capture_blackbox_incident(frame, hazard_type, score, speed):
    global last_blackbox_capture_time, blackbox_incidents_list
    now = time.time()
    if now - last_blackbox_capture_time < 5.0:
        return
    last_blackbox_capture_time = now

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_display = datetime.now().strftime("%H:%M:%S")
    filename = f"incident_{timestamp_str}_{int(now)}.jpg"
    filepath = os.path.join(incidents_dir, filename)

    try:
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, h - 40), (w, h), (0, 0, 180), -1)
        cv2.putText(annotated, f"BLACKBOX HAZARD: {hazard_type.upper()} | SCORE: {score}% | SPEED: {speed} KM/H | {time_display}",
                    (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.imwrite(filepath, annotated)

        blackbox_incidents_list.append({
            'filename': filename,
            'time': time_display,
            'hazard': hazard_type,
            'score': score,
            'speed': speed,
            'url': f'/incidents/{filename}'
        })
        telemetry['total_blackbox_records'] = len(blackbox_incidents_list)
    except Exception as e:
        print(f"[Blackbox Error] {e}")

def reset_telemetry():
    global telemetry, prev_yawn, prev_drowsy, prev_distraction, prev_occlusion, prev_perclos_hazard, safety_scores_history, perclos_window
    global std_no_face_duration, std_head_diverted_duration, std_phone_duration, std_eyes_closed_duration, driver_slump_duration, prev_medical_emergency
    global telematics_log_buffer
    with perclos_lock:
        perclos_window.clear()
    with telematics_lock:
        telematics_log_buffer.clear()

    grace, profile = get_speed_thresholds(current_vehicle_speed)
    cam_label = "LAPTOP CAM (0)" if selected_camera_index == 0 else f"USB WEBCAM ({selected_camera_index})"
    telemetry['engine'] = ai_engine_mode
    telemetry['eyes_status'] = 'Normal'
    telemetry['yawn_status'] = 'Normal'
    telemetry['action'] = 'Normal'
    telemetry['safety_score'] = 100
    telemetry['risk_level'] = 'SAFE'
    telemetry['fps'] = 30
    telemetry['speed_kmh'] = current_vehicle_speed
    telemetry['speed_profile'] = profile
    telemetry['grace_period'] = grace
    telemetry['blind_distance_m'] = 0.0
    telemetry['camera_index'] = selected_camera_index
    telemetry['camera_label'] = cam_label
    telemetry['face_detected'] = True
    telemetry['pitch'] = 0.0
    telemetry['yaw'] = 0.0
    telemetry['roll'] = 0.0
    telemetry['perclos'] = 0.0
    telemetry['perclos_level'] = 'OPTIMAL (NORMAL)'
    telemetry['sleep_prob'] = 0.0
    telemetry['phone_prob'] = 0.0
    telemetry['sunglasses_prob'] = 0.0
    telemetry['left_blink'] = 0.0
    telemetry['right_blink'] = 0.0
    telemetry['total_yawns'] = 0
    telemetry['total_drowsy'] = 0
    telemetry['total_distractions'] = 0
    telemetry['total_blackbox_records'] = len(blackbox_incidents_list)
    telemetry['medical_emergency'] = False
    telemetry['take_a_break'] = False
    telemetry['session_start'] = time.time()
    telemetry['video_mode'] = video_source_mode
    telemetry['video_name'] = uploaded_video_name if video_source_mode == 'video' else 'Live Webcam'
    telemetry['video_complete'] = False
    telemetry['event_log'] = []
    prev_yawn = False
    prev_drowsy = False
    prev_distraction = False
    prev_occlusion = False
    prev_perclos_hazard = False
    prev_medical_emergency = False
    driver_slump_duration = 0.0
    std_no_face_duration = 0.0
    std_head_diverted_duration = 0.0
    std_phone_duration = 0.0
    std_eyes_closed_duration = 0.0
    safety_scores_history = []

def update_perclos(is_closed):
    now = time.time()
    with perclos_lock:
        perclos_window.append((now, 1 if is_closed else 0))
        while perclos_window and perclos_window[0][0] < (now - 60.0):
            perclos_window.popleft()

        if len(perclos_window) < 5:
            return 0.0, 'CALIBRATING...'

        total_frames = len(perclos_window)
        closed_frames = sum(val for _, val in perclos_window)
        perclos_pct = round((closed_frames / total_frames) * 100.0, 1)

        if perclos_pct < 8.0:
            level = 'OPTIMAL ALERTNESS'
        elif perclos_pct < 15.0:
            level = 'EARLY FATIGUE'
        else:
            level = 'CRITICAL DROWSINESS'

        return perclos_pct, level

current_raw_frame = None

# Model loading is deliberately lazy.  Importing the Flask app must not access a
# camera, download a repository, or fail merely because an optional model is absent.
model = None
yolo_model = None
facial_tracker = None
model_load_lock = threading.Lock()
models_initialized = False
model_load_errors = []
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def initialize_models():
    """Load available inference components once and retain useful fallbacks."""
    global model, yolo_model, facial_tracker, models_initialized
    with model_load_lock:
        if models_initialized:
            return

        def load_component(name, loader):
            try:
                return loader()
            except Exception as exc:
                model_load_errors.append(f"{name}: {exc}")
                print(f"[Model Loader] {name} unavailable: {exc}")
                return None

        def load_action_model():
            action_model = MobileNet()
            action_model.load_weights(os.path.join(base_dir, "models", "model_split.h5"))
            return action_model

        model = load_component("MobileNet action classifier", load_action_model)

        def load_yolo():
            detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
            detector.classes = [67]
            detector.conf = 0.55
            detector.iou = 0.45
            return detector

        yolo_model = load_component("YOLO phone detector", load_yolo)
        facial_tracker = load_component("MediaPipe facial tracker", FacialTracker)
        models_initialized = True


def is_allowed_video(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Driver Monitoring System - Cockpit Command Center</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #080C14;
            --bg-card: rgba(15, 23, 42, 0.80);
            --border-card: rgba(51, 65, 85, 0.6);
            --neon-blue: #00F0FF;
            --neon-green: #00FF88;
            --neon-yellow: #FFD600;
            --neon-red: #FF0055;
            --neon-purple: #B026FF;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 16px 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            background-image:
                radial-gradient(circle at 10% 20%, rgba(0, 240, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(176, 38, 255, 0.05) 0%, transparent 40%),
                linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 30px 30px, 30px 30px;
        }

        /* Glassmorphic Container */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-card);
            border-radius: 14px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
            padding: 16px 20px;
            position: relative;
            overflow: hidden;
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
        }

        /* Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .logo-icon {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.5);
        }

        .logo-icon svg {
            width: 22px;
            height: 22px;
            fill: #000;
        }

        .brand-text h1 {
            font-size: 20px;
            font-weight: 800;
            letter-spacing: 0.5px;
            background: linear-gradient(135deg, #FFFFFF, var(--neon-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-text p {
            font-size: 11px;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 1px;
        }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .source-switcher {
            display: flex;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-card);
            border-radius: 8px;
            padding: 3px;
            gap: 3px;
        }

        .source-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 700;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .source-btn.active {
            background: rgba(0, 240, 255, 0.15);
            color: var(--neon-blue);
            border: 1px solid var(--neon-blue);
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
        }

        .source-btn.active-cam {
            background: rgba(0, 255, 136, 0.2);
            color: var(--neon-green);
            border: 1px solid var(--neon-green);
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.4);
        }

        .btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-card);
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            font-size: 12px;
            font-weight: 600;
            padding: 7px 14px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .btn:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: var(--neon-blue);
            color: var(--neon-blue);
        }

        .btn-upload {
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.2), rgba(176, 38, 255, 0.2));
            border-color: var(--neon-blue);
            color: #FFFFFF;
        }
        .btn-upload:hover {
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.4), rgba(176, 38, 255, 0.4));
            box-shadow: 0 0 12px rgba(0, 240, 255, 0.4);
        }

        .btn-blackbox {
            background: linear-gradient(135deg, rgba(255, 0, 85, 0.2), rgba(255, 214, 0, 0.2));
            border-color: var(--neon-red);
            color: #FFFFFF;
        }
        .btn-blackbox:hover {
            background: linear-gradient(135deg, rgba(255, 0, 85, 0.4), rgba(255, 214, 0, 0.4));
            box-shadow: 0 0 12px rgba(255, 0, 85, 0.5);
        }

        .btn-export {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 240, 255, 0.2));
            border-color: var(--neon-green);
            color: #FFFFFF;
        }
        .btn-export:hover {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.4), rgba(0, 240, 255, 0.4));
            box-shadow: 0 0 12px rgba(0, 255, 136, 0.4);
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 20px;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid var(--neon-green);
            color: var(--neon-green);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 700;
        }

        .pulse-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: var(--neon-green);
            box-shadow: 0 0 8px var(--neon-green);
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.7; }
            50% { transform: scale(1.3); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.7; }
        }

        /* Speed Control Widget in Header */
        .speed-widget {
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-card);
            border-radius: 8px;
            padding: 4px 12px;
        }

        .speed-val {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 800;
            font-size: 13px;
            color: var(--neon-blue);
            min-width: 65px;
        }

        .speed-slider {
            -webkit-appearance: none;
            width: 80px;
            height: 4px;
            background: rgba(255, 255, 255, 0.15);
            border-radius: 2px;
            outline: none;
        }
        .speed-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--neon-blue);
            cursor: pointer;
            box-shadow: 0 0 8px var(--neon-blue);
        }

        /* Layout Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 16px;
            flex: 1;
        }

        /* Video Feed HUD */
        .video-container {
            position: relative;
            width: 100%;
            height: 480px;
            border-radius: 12px;
            overflow: hidden;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .video-feed {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .hud-overlay {
            position: absolute;
            inset: 0;
            pointer-events: none;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 16px;
        }

        .reticle-corner {
            position: absolute;
            width: 20px;
            height: 20px;
            border: 2px solid var(--neon-blue);
            opacity: 0.6;
        }
        .top-left { top: 12px; left: 12px; border-right: none; border-bottom: none; }
        .top-right { top: 12px; right: 12px; border-left: none; border-bottom: none; }
        .bottom-left { bottom: 12px; left: 12px; border-right: none; border-top: none; }
        .bottom-right { bottom: 12px; right: 12px; border-left: none; border-top: none; }

        .hud-badge {
            background: rgba(0, 0, 0, 0.65);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 4px 10px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--neon-blue);
        }

        .hazard-banner {
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(255, 0, 85, 0.90);
            color: #fff;
            padding: 10px 28px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 15px;
            letter-spacing: 1px;
            border: 2px solid #fff;
            box-shadow: 0 0 30px var(--neon-red);
            display: none;
            animation: blink 0.6s infinite;
            z-index: 10;
            text-align: center;
        }

        /* Medical Emergency SOS Banner */
        .medical-banner {
            position: absolute;
            top: 65px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #FF0055, #990000);
            color: #fff;
            padding: 14px 32px;
            border-radius: 12px;
            font-weight: 900;
            font-size: 16px;
            letter-spacing: 1.5px;
            border: 3px solid #FFFFFF;
            box-shadow: 0 0 50px rgba(255, 0, 85, 0.9);
            display: none;
            animation: pulse-sos 0.8s infinite;
            z-index: 25;
            text-align: center;
        }

        @keyframes pulse-sos {
            0%, 100% { transform: translateX(-50%) scale(1); box-shadow: 0 0 50px rgba(255, 0, 85, 0.9); }
            50% { transform: translateX(-50%) scale(1.06); box-shadow: 0 0 70px rgba(255, 255, 255, 1); }
        }

        .take-a-break-banner {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, rgba(255, 0, 85, 0.95), rgba(176, 38, 255, 0.95));
            color: #fff;
            padding: 12px 30px;
            border-radius: 12px;
            font-weight: 800;
            font-size: 16px;
            letter-spacing: 1px;
            border: 2px solid var(--neon-yellow);
            box-shadow: 0 0 40px rgba(255, 214, 0, 0.6);
            display: none;
            animation: pulse-slow 1.2s infinite;
            z-index: 15;
            text-align: center;
        }

        @keyframes pulse-slow {
            0%, 100% { transform: translateX(-50%) scale(1); }
            50% { transform: translateX(-50%) scale(1.04); }
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.35; }
        }

        .video-box.hazard {
            border-color: var(--neon-red) !important;
            box-shadow: 0 0 30px rgba(255, 0, 85, 0.6) !important;
        }

        .video-toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
            flex-wrap: wrap;
            gap: 8px;
        }

        .toolbar-left {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .active-source-indicator {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--neon-blue);
            background: rgba(0, 240, 255, 0.08);
            border: 1px solid rgba(0, 240, 255, 0.3);
            padding: 4px 10px;
            border-radius: 6px;
        }

        /* Telemetry Panel */
        .telemetry-panel {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        /* Circular Safety Gauge */
        .gauge-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 16px;
            position: relative;
        }

        .gauge-svg {
            width: 140px;
            height: 140px;
            transform: rotate(-90deg);
        }

        .gauge-bg {
            fill: none;
            stroke: rgba(255, 255, 255, 0.06);
            stroke-width: 12;
        }

        .gauge-progress {
            fill: none;
            stroke: var(--neon-green);
            stroke-width: 12;
            stroke-linecap: round;
            stroke-dasharray: 439.8;
            stroke-dashoffset: 0;
            transition: stroke-dashoffset 0.4s ease, stroke 0.4s ease;
        }

        .gauge-text {
            position: absolute;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .gauge-score {
            font-size: 32px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: #fff;
        }

        .gauge-label {
            font-size: 10px;
            color: var(--text-secondary);
            letter-spacing: 1px;
        }

        /* PERCLOS Gauge Widget */
        .perclos-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 8px;
            padding: 10px 14px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .perclos-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .perclos-title {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-secondary);
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .perclos-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            font-weight: 800;
            color: var(--neon-green);
        }

        .perclos-bar-bg {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }

        .perclos-bar-fill {
            height: 100%;
            width: 0%;
            background: var(--neon-green);
            border-radius: 3px;
            transition: width 0.3s ease, background-color 0.3s ease;
        }

        .perclos-sub {
            display: flex;
            justify-content: space-between;
            font-size: 9px;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Status List */
        .status-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .item-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 8px;
            padding: 10px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .item-info {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .item-icon {
            font-size: 16px;
        }

        .item-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .item-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 700;
        }

        /* Counters Grid */
        .counters-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }

        .counter-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 8px;
            padding: 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .counter-num {
            font-size: 20px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: var(--neon-blue);
        }

        .counter-lbl {
            font-size: 9px;
            color: var(--text-secondary);
            letter-spacing: 0.5px;
            margin-top: 2px;
        }

        /* Bottom Row */
        .bottom-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .chart-container {
            width: 100%;
            height: 120px;
            position: relative;
            margin-top: 8px;
        }

        canvas {
            width: 100%;
            height: 100%;
        }

        .log-table-container {
            max-height: 120px;
            overflow-y: auto;
            margin-top: 8px;
        }

        .log-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
        }

        .log-table th {
            text-align: left;
            color: var(--text-secondary);
            padding: 4px 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .log-table td {
            padding: 4px 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        .log-badge-hazard {
            color: var(--neon-red);
            font-weight: 700;
        }

        .log-badge-warning {
            color: var(--neon-yellow);
            font-weight: 700;
        }

        .log-badge-info {
            color: var(--neon-blue);
            font-weight: 700;
        }

        /* Modal Styles */
        .modal-backdrop {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(12px);
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }

        .modal-content {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 24px 32px;
            max-width: 560px;
            width: 90%;
            max-height: 85vh;
            overflow-y: auto;
            text-align: center;
            display: flex;
            flex-direction: column;
            gap: 16px;
            box-shadow: 0 0 40px rgba(0, 240, 255, 0.2);
        }

        .grade-badge {
            font-size: 48px;
            font-weight: 900;
            font-family: 'JetBrains Mono', monospace;
            background: linear-gradient(135deg, #FFFFFF, var(--neon-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .summary-stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 12px 0;
        }

        /* Blackbox Incident Gallery */
        .incident-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 12px;
            margin-top: 12px;
            max-height: 380px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .incident-card {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 0, 85, 0.4);
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .incident-card:hover {
            border-color: var(--neon-red);
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(255, 0, 85, 0.4);
        }

        .incident-img {
            width: 100%;
            height: 95px;
            object-fit: cover;
            background: #000;
        }

        .incident-info {
            padding: 6px 8px;
            font-size: 10px;
            text-align: left;
            font-family: 'JetBrains Mono', monospace;
        }
    </style>
</head>
<body>

    <!-- Hidden File Input for Video Upload -->
    <input type="file" id="video-file-input" accept="video/mp4,video/avi,video/mov,video/mkv,.mp4,.avi,.mov" style="display:none" onchange="uploadSelectedVideo(this)">

    <!-- Header -->
    <header class="glass-card">
        <div class="brand">
            <div class="logo-icon">
                <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
            </div>
            <div class="brand-text">
                <h1>DMS COCKPIT COMMAND CENTER</h1>
                <p>AI DRIVER MONITORING & ACCIDENT PREVENTION SYSTEM</p>
            </div>
        </div>

        <div class="header-controls">
            <!-- Source Switcher -->
            <div class="source-switcher">
                <button id="btn-src-cam" class="source-btn active" onclick="setVideoSource('webcam')">
                    <span>📹</span> LIVE WEBCAM
                </button>
                <button id="btn-src-vid" class="source-btn" onclick="setVideoSource('video')">
                    <span>🎬</span> VIDEO FILE
                </button>
            </div>

            <!-- Switchable Camera Selector (Laptop vs External USB) -->
            <div class="source-switcher" title="Switch Video Hardware Input">
                <button id="btn-cam-0" class="source-btn active-cam" onclick="setCameraDevice(0)">
                    <span>💻</span> LAPTOP (0)
                </button>
                <button id="btn-cam-1" class="source-btn" onclick="setCameraDevice(1)">
                    <span>📹</span> USB WEBCAM (1)
                </button>
            </div>

            <!-- Speed-Adaptive Sensitivity Widget -->
            <div class="speed-widget" title="Euro NCAP Speed-Adaptive Distraction Threshold">
                <span style="font-size:12px;">🚗</span>
                <span id="speed-display" class="speed-val">60 KM/H</span>
                <input type="range" min="0" max="140" step="10" value="60" id="speed-slider" class="speed-slider" oninput="changeSpeed(this.value)">
            </div>

            <!-- Blackbox Incident Gallery Button -->
            <button class="btn btn-blackbox" onclick="openBlackboxModal()">
                <span>📼</span> BLACKBOX (<span id="btn-blackbox-cnt">0</span>)
            </button>

            <!-- Export Telematics CSV -->
            <button class="btn btn-export" onclick="exportTelematicsData()">
                <span>📥</span> EXPORT TRIP
            </button>

            <!-- Upload Button -->
            <button class="btn btn-upload" onclick="document.getElementById('video-file-input').click()">
                <span>📁</span> UPLOAD VIDEO
            </button>

            <div id="status-pill" class="status-pill">
                <div class="pulse-dot"></div>
                <span id="status-text">SYSTEM ACTIVE</span>
            </div>

            <button class="btn" id="audio-toggle" onclick="toggleAudio()">
                <span id="audio-icon">🔊</span>
                <span id="audio-label">ALARM ON</span>
            </button>

            <button class="btn" onclick="toggleFullscreen()">
                <span>🖥️</span> FULLSCREEN
            </button>
        </div>
    </header>

    <!-- Main Grid -->
    <div class="dashboard-grid">

        <!-- Left: Video Feed HUD -->
        <div class="glass-card">
            <div class="video-container" id="video-box">
                <img src="/video_feed" class="video-feed" alt="Live Stream">

                <div id="hazard-banner" class="hazard-banner">🚨 HAZARD DETECTED</div>

                <!-- Medical Emergency Incapacitation Banner -->
                <div id="medical-banner" class="medical-banner">
                    <div>🚨 MEDICAL EMERGENCY: DRIVER INCAPACITATED</div>
                    <div style="font-size: 13px; font-weight: 500; margin-top: 4px;">UNNATURAL HEAD SLUMP / UNRESPONSIVE DRIVER DETECTED</div>
                    <button class="btn" style="margin: 8px auto 0; background: rgba(0,0,0,0.6); font-size: 11px; border-color:#fff;" onclick="resetMedicalEmergency()">🛡️ OVERRIDE & RESET</button>
                </div>

                <!-- Take a Break Emergency Banner -->
                <div id="break-banner" class="take-a-break-banner">
                    <div>🛑 CRITICAL FATIGUE DETECTED!</div>
                    <div style="font-size: 13px; font-weight: 500; margin-top: 4px;">PLEASE PULL OVER AND TAKE A REST</div>
                    <button class="btn" style="margin: 8px auto 0; background: rgba(0,0,0,0.5); font-size: 11px;" onclick="acknowledgeBreak()">☕ ACKNOWLEDGE & RESET</button>
                </div>

                <div class="hud-overlay">
                    <div class="reticle-corner top-left"></div>
                    <div class="reticle-corner top-right"></div>
                    <div class="reticle-corner bottom-left"></div>
                    <div class="reticle-corner bottom-right"></div>

                    <div style="display: flex; justify-content: space-between;">
                        <div class="hud-badge" id="hud-fps">FPS: --</div>
                        <div class="hud-badge" id="hud-cam-badge">CAM: LAPTOP (0)</div>
                        <div class="hud-badge" id="hud-timer">SESSION: 00:00:00</div>
                    </div>
                </div>
            </div>

            <div class="video-toolbar">
                <div class="toolbar-left">
                    <button class="btn" onclick="takeSnapshot()">📷 SNAPSHOT FRAME</button>
                    <button class="btn" onclick="toggleCameraCycle()">🔄 SWITCH CAMERA</button>
                    <button class="btn" id="nv-toggle-btn" onclick="cycleNightVision()">🌙 NIGHT VISION: AUTO</button>
                    <button class="btn" onclick="reloadStream()">🔄 REFRESH FEED</button>
                </div>
                <div class="active-source-indicator" id="active-source-lbl">SOURCE: LIVE WEBCAM | CAM 0 | ENGINE: MEDIAPIPE+YOLO</div>
            </div>
        </div>

        <!-- Right: Telemetry Panel -->
        <div class="telemetry-panel">

            <!-- Circular Safety Gauge -->
            <div class="glass-card gauge-card">
                <svg class="gauge-svg" viewBox="0 0 160 160">
                    <circle class="gauge-bg" cx="80" cy="80" r="70"></circle>
                    <circle id="gauge-bar" class="gauge-progress" cx="80" cy="80" r="70"></circle>
                </svg>
                <div class="gauge-text">
                    <div id="safety-score" class="gauge-score">100%</div>
                    <div class="gauge-label">ATTENTION SCORE</div>
                </div>
            </div>

            <!-- Euro NCAP PERCLOS Gauge Widget -->
            <div class="perclos-card">
                <div class="perclos-header">
                    <div class="perclos-title">
                        <span>⏱️</span> EURO NCAP PERCLOS (60S)
                    </div>
                    <div id="perclos-val" class="perclos-val">0.0%</div>
                </div>
                <div class="perclos-bar-bg">
                    <div id="perclos-bar-fill" class="perclos-bar-fill"></div>
                </div>
                <div class="perclos-sub">
                    <span id="perclos-level" style="color:var(--neon-green)">OPTIMAL ALERTNESS</span>
                    <span>LIMIT: 15.0%</span>
                </div>
            </div>

            <!-- Status List -->
            <div class="status-list">
                <div class="item-card">
                    <div class="item-info">
                        <span class="item-icon">👁️</span>
                        <span class="item-title">EYES STATUS</span>
                    </div>
                    <span id="eyes-val" class="item-val" style="color: var(--neon-green)">NORMAL</span>
                </div>

                <div class="item-card">
                    <div class="item-info">
                        <span class="item-icon">🧭</span>
                        <span class="item-title">HEAD POSE (YAW/PITCH)</span>
                    </div>
                    <span id="pose-val" class="item-val" style="color: var(--neon-blue)">0.0° / 0.0°</span>
                </div>

                <div class="item-card">
                    <div class="item-info">
                        <span class="item-icon">⚡</span>
                        <span class="item-title">DRIVER ACTION</span>
                    </div>
                    <span id="action-val" class="item-val" style="color: var(--neon-green)">NORMAL DRIVING</span>
                </div>

                <div class="item-card">
                    <div class="item-info">
                        <span class="item-icon">🏎️</span>
                        <span class="item-title">SPEED SENSITIVITY</span>
                    </div>
                    <span id="speed-profile-val" class="item-val" style="color: var(--neon-yellow)">URBAN (1.5S GRACE)</span>
                </div>

                <div class="item-card">
                    <div class="item-info">
                        <span class="item-icon">🤖</span>
                        <span class="item-title">ACTIVE AI ENGINE</span>
                    </div>
                    <span id="engine-val" class="item-val" style="color: var(--neon-green)">MEDIAPIPE+YOLO</span>
                </div>
            </div>

            <!-- Counters Grid -->
            <div class="counters-grid">
                <div class="counter-card">
                    <div id="cnt-yawns" class="counter-num">0</div>
                    <div class="counter-lbl">YAWNS</div>
                </div>
                <div class="counter-card">
                    <div id="cnt-drowsy" class="counter-num">0</div>
                    <div class="counter-lbl">DROWSY</div>
                </div>
                <div class="counter-card">
                    <div id="cnt-distractions" class="counter-num">0</div>
                    <div class="counter-lbl">HAZARDS</div>
                </div>
            </div>

        </div>

    </div>

    <!-- Bottom Analytics & Incident Log -->
    <div class="bottom-grid">
        <div class="glass-card">
            <h3 style="font-size: 14px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px;">LIVE ATTENTION TELEMETRY (30S)</h3>
            <div class="chart-container">
                <canvas id="telemetryChart"></canvas>
            </div>
        </div>

        <div class="glass-card">
            <h3 style="font-size: 14px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px;">INCIDENT LOG HISTORY</h3>
            <div class="log-table-container">
                <table class="log-table">
                    <thead>
                        <tr>
                            <th>TIME</th>
                            <th>SEVERITY</th>
                            <th>EVENT DESCRIPTION</th>
                        </tr>
                    </thead>
                    <tbody id="log-body">
                        <tr>
                            <td colspan="3" style="text-align: center; color: var(--text-secondary)">No incident logs yet. Monitoring active.</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Blackbox Incidents Modal -->
    <div id="blackbox-modal" class="modal-backdrop">
        <div class="modal-content">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h2 style="font-size: 18px; color:var(--neon-red); letter-spacing: 0.5px;">📼 BLACKBOX INCIDENT RECORDER</h2>
                <button class="btn" onclick="closeBlackboxModal()">✕</button>
            </div>
            <p style="font-size: 12px; color: var(--text-secondary);">
                Automatic forensic snapshots captured during critical driver hazard events.
            </p>

            <div id="incident-grid" class="incident-grid">
                <!-- Dynamically loaded incident cards -->
            </div>

            <div style="display: flex; gap: 10px; justify-content: center; margin-top:10px;">
                <button class="btn btn-blackbox" onclick="clearIncidents()">🗑️ CLEAR ARCHIVE</button>
                <button class="btn" onclick="closeBlackboxModal()">CLOSE</button>
            </div>
        </div>
    </div>

    <!-- Video Analysis Scorecard Modal -->
    <div id="scorecard-modal" class="modal-backdrop">
        <div class="modal-content">
            <h2 style="font-size: 20px; letter-spacing: 0.5px;">DRIVER ANALYSIS SCORECARD</h2>
            <p id="modal-video-name" style="font-size: 12px; color: var(--text-secondary);"></p>

            <div>
                <div id="modal-grade" class="grade-badge">A+</div>
                <div id="modal-rating" style="font-weight: 700; color: var(--neon-green); font-size: 14px;">EXCELLENT ATTENTION</div>
            </div>

            <div class="summary-stats-grid">
                <div class="counter-card">
                    <div id="modal-yawns" class="counter-num">0</div>
                    <div class="counter-lbl">YAWNS</div>
                </div>
                <div class="counter-card">
                    <div id="modal-drowsy" class="counter-num">0</div>
                    <div class="counter-lbl">DROWSY</div>
                </div>
                <div class="counter-card">
                    <div id="modal-hazards" class="counter-num">0</div>
                    <div class="counter-lbl">HAZARDS</div>
                </div>
            </div>

            <div style="display: flex; gap: 10px; justify-content: center;">
                <button class="btn btn-upload" onclick="replayVideo()">🔁 REPLAY VIDEO</button>
                <button class="btn" onclick="closeModalAndSwitchCam()">📹 SWITCH TO WEBCAM</button>
            </div>
        </div>
    </div>

    <!-- Audio & Interactive Telemetry Script -->
    <script>
        let audioCtx = null;
        let audioEnabled = true;
        let isAlarmPlaying = false;
        let isChimePlaying = false;
        let isSirenPlaying = false;
        let modalShown = false;
        let lastVoiceTime = 0;
        let breakAcknowledged = false;
        let medicalOverridden = false;
        let nightVisionState = 'auto';
        let currentCameraIndex = 0;

        function initAudio() {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx && audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
        }

        document.addEventListener('click', () => {
            initAudio();
        }, { once: false });

        // Medical Emergency Siren (Alternating European Two-Tone)
        function playMedicalEmergencySiren() {
            if (!audioEnabled) return;
            initAudio();
            if (isSirenPlaying) return;
            isSirenPlaying = true;

            try {
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();

                osc.type = 'triangle';
                osc.frequency.setValueAtTime(660, audioCtx.currentTime);
                osc.frequency.setValueAtTime(880, audioCtx.currentTime + 0.25);
                osc.frequency.setValueAtTime(660, audioCtx.currentTime + 0.50);
                osc.frequency.setValueAtTime(880, audioCtx.currentTime + 0.75);

                gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
                gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 1.0);

                osc.connect(gain);
                gain.connect(audioCtx.destination);

                osc.start();
                osc.stop(audioCtx.currentTime + 1.0);
            } catch (e) {}

            setTimeout(() => { isSirenPlaying = false; }, 1050);
        }

        // Urgent Hazard Alarm (Pulsating buzzer)
        function playHazardAlarm() {
            if (!audioEnabled || isSirenPlaying) return;
            initAudio();
            if (isAlarmPlaying) return;
            isAlarmPlaying = true;

            try {
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();

                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(950, audioCtx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(420, audioCtx.currentTime + 0.22);

                gain.gain.setValueAtTime(0.28, audioCtx.currentTime);
                gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.22);

                osc.connect(gain);
                gain.connect(audioCtx.destination);

                osc.start();
                osc.stop(audioCtx.currentTime + 0.22);
            } catch (e) {}

            setTimeout(() => { isAlarmPlaying = false; }, 260);
        }

        // Warning Chime (Cautionary two-tone ping)
        function playWarningChime() {
            if (!audioEnabled || isSirenPlaying) return;
            initAudio();
            if (isChimePlaying || isAlarmPlaying) return;
            isChimePlaying = true;

            try {
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();

                osc.type = 'sine';
                osc.frequency.setValueAtTime(650, audioCtx.currentTime);
                osc.frequency.setValueAtTime(880, audioCtx.currentTime + 0.12);

                gain.gain.setValueAtTime(0.18, audioCtx.currentTime);
                gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.25);

                osc.connect(gain);
                gain.connect(audioCtx.destination);

                osc.start();
                osc.stop(audioCtx.currentTime + 0.25);
            } catch (e) {}

            setTimeout(() => { isChimePlaying = false; }, 500);
        }

        function speakVoiceAlert(text) {
            if (!audioEnabled) return;
            const now = Date.now();
            if (now - lastVoiceTime < 8000) return;
            lastVoiceTime = now;

            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const msg = new SpeechSynthesisUtterance(text);
                msg.rate = 1.05;
                msg.pitch = 1.05;
                window.speechSynthesis.speak(msg);
            }
        }

        function acknowledgeBreak() {
            breakAcknowledged = true;
            document.getElementById('break-banner').style.display = 'none';
            fetch('/reset_break', { method: 'POST' });
        }

        function resetMedicalEmergency() {
            medicalOverridden = true;
            document.getElementById('medical-banner').style.display = 'none';
            fetch('/reset_medical', { method: 'POST' });
        }

        function toggleAudio() {
            audioEnabled = !audioEnabled;
            document.getElementById('audio-icon').textContent = audioEnabled ? '🔊' : '🔇';
            document.getElementById('audio-label').textContent = audioEnabled ? 'ALARM ON' : 'ALARM MUTED';
            if (audioEnabled) initAudio();
        }

        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else {
                if (document.exitFullscreen) document.exitFullscreen();
            }
        }

        function reloadStream() {
            const feed = document.querySelector('.video-feed');
            if (feed) {
                feed.src = '/video_feed?' + new Date().getTime();
            }
        }

        window.addEventListener('DOMContentLoaded', () => {
            const feed = document.querySelector('.video-feed');
            if (feed) {
                feed.onerror = function() {
                    setTimeout(reloadStream, 1500);
                };
            }
        });

        function takeSnapshot() {
            window.open('/snapshot', '_blank');
        }

        function exportTelematicsData() {
            window.location.href = '/api/export_telematics';
        }

        // Camera Switching Functions
        async function setCameraDevice(idx) {
            document.getElementById('status-text').textContent = `CONNECTING TO CAM ${idx}...`;
            try {
                const res = await fetch('/set_camera', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ camera_index: idx })
                });
                const data = await res.json();
                if (data.success) {
                    currentCameraIndex = idx;
                    const btn0 = document.getElementById('btn-cam-0');
                    const btn1 = document.getElementById('btn-cam-1');
                    if (idx === 0) {
                        btn0.classList.add('active-cam');
                        btn1.classList.remove('active-cam');
                    } else {
                        btn1.classList.add('active-cam');
                        btn0.classList.remove('active-cam');
                    }
                    document.getElementById('hud-cam-badge').textContent = `CAM: ${data.label}`;
                    document.getElementById('status-text').textContent = `CAM ${idx} ACTIVE`;
                    reloadStream();
                } else {
                    document.getElementById('status-text').textContent = `CAM ${idx} NOT DETECTED`;
                }
            } catch (e) {
                document.getElementById('status-text').textContent = 'CAMERA ERROR';
            }
            setTimeout(() => {
                document.getElementById('status-text').textContent = 'SYSTEM ACTIVE';
            }, 1500);
        }

        function toggleCameraCycle() {
            const nextIdx = currentCameraIndex === 0 ? 1 : 0;
            setCameraDevice(nextIdx);
        }

        async function cycleNightVision() {
            if (nightVisionState === 'auto') nightVisionState = 'on';
            else if (nightVisionState === 'on') nightVisionState = 'off';
            else nightVisionState = 'auto';

            const btn = document.getElementById('nv-toggle-btn');
            btn.textContent = `🌙 NIGHT VISION: ${nightVisionState.toUpperCase()}`;
            try {
                await fetch('/set_night_vision', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ mode: nightVisionState })
                });
            } catch (e) {}
        }

        async function changeSpeed(val) {
            document.getElementById('speed-display').textContent = `${val} KM/H`;
            try {
                await fetch('/set_speed', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ speed: parseInt(val) })
                });
            } catch (e) {}
        }

        // Blackbox Modal functions
        async function openBlackboxModal() {
            try {
                const res = await fetch('/api/incidents');
                const data = await res.json();
                const grid = document.getElementById('incident-grid');

                if (data.incidents.length === 0) {
                    grid.innerHTML = '<div style="grid-column: 1/-1; padding: 20px; color: var(--text-secondary);">No critical hazard incidents recorded yet. Safe driving!</div>';
                } else {
                    grid.innerHTML = data.incidents.map(inc => `
                        <div class="incident-card" onclick="window.open('${inc.url}', '_blank')">
                            <img src="${inc.url}" class="incident-img" alt="Incident Frame">
                            <div class="incident-info">
                                <div style="color:var(--neon-red); font-weight:700;">${inc.hazard}</div>
                                <div style="color:var(--text-secondary);">${inc.time} | ${inc.speed} km/h</div>
                            </div>
                        </div>
                    `).join('');
                }
            } catch (e) {}
            document.getElementById('blackbox-modal').style.display = 'flex';
        }

        function closeBlackboxModal() {
            document.getElementById('blackbox-modal').style.display = 'none';
        }

        async function clearIncidents() {
            try {
                await fetch('/api/clear_incidents', { method: 'POST' });
                openBlackboxModal();
            } catch (e) {}
        }

        async function setAIEngine(engine) {}

        async function uploadSelectedVideo(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            const formData = new FormData();
            formData.append('video_file', file);

            document.getElementById('status-text').textContent = 'UPLOADING VIDEO...';

            try {
                const res = await fetch('/upload_video', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (data.success) {
                    modalShown = false;
                    breakAcknowledged = false;
                    medicalOverridden = false;
                    document.getElementById('scorecard-modal').style.display = 'none';
                    updateSourceUI('video', data.filename);
                    reloadStream();
                } else {
                    alert('Upload failed: ' + (data.error || 'Unknown error'));
                }
            } catch (err) {
                alert('Upload error: ' + err);
            } finally {
                document.getElementById('status-text').textContent = 'SYSTEM ACTIVE';
            }
        }

        async function setVideoSource(source) {
            try {
                const res = await fetch('/set_source', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ source })
                });
                const data = await res.json();
                if (data.success) {
                    modalShown = false;
                    breakAcknowledged = false;
                    medicalOverridden = false;
                    document.getElementById('scorecard-modal').style.display = 'none';
                    updateSourceUI(source);
                    reloadStream();
                }
            } catch (e) {}
        }

        function updateSourceUI(source, filename) {
            const btnCam = document.getElementById('btn-src-cam');
            const btnVid = document.getElementById('btn-src-vid');
            const lbl = document.getElementById('active-source-lbl');
            const engText = 'MEDIAPIPE+YOLO';

            if (source === 'video') {
                btnCam.classList.remove('active');
                btnVid.classList.add('active');
                lbl.textContent = `SOURCE: VIDEO (${filename || 'LOADED FILE'}) | ENGINE: ${engText}`;
            } else {
                btnVid.classList.remove('active');
                btnCam.classList.add('active');
                lbl.textContent = `SOURCE: LIVE WEBCAM | CAM ${currentCameraIndex} | ENGINE: ${engText}`;
            }
        }

        async function showScorecardModal() {
            try {
                const res = await fetch('/video_summary');
                const data = await res.json();

                document.getElementById('modal-video-name').textContent = data.filename ? `Analysis report for ${data.filename}` : 'Uploaded video report';
                document.getElementById('modal-grade').textContent = data.safety_grade.split(' ')[0];
                document.getElementById('modal-rating').textContent = data.rating;
                document.getElementById('modal-yawns').textContent = data.total_yawns;
                document.getElementById('modal-drowsy').textContent = data.total_drowsy;
                document.getElementById('modal-hazards').textContent = data.total_distractions;

                document.getElementById('scorecard-modal').style.display = 'flex';
            } catch (e) {}
        }

        function replayVideo() {
            document.getElementById('scorecard-modal').style.display = 'none';
            modalShown = false;
            breakAcknowledged = false;
            medicalOverridden = false;
            setVideoSource('video');
        }

        function closeModalAndSwitchCam() {
            document.getElementById('scorecard-modal').style.display = 'none';
            modalShown = false;
            breakAcknowledged = false;
            medicalOverridden = false;
            setVideoSource('webcam');
        }

        // Live Canvas Chart Logic
        const canvas = document.getElementById('telemetryChart');
        const ctx = canvas.getContext('2d');
        let chartData = Array(30).fill(100);

        function renderChart() {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;

            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.lineWidth = 1;
            for (let y = 0; y <= h; y += h / 4) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }

            const grad = ctx.createLinearGradient(0, 0, 0, h);
            grad.addColorStop(0, 'rgba(0, 240, 255, 0.4)');
            grad.addColorStop(1, 'rgba(0, 240, 255, 0.0)');

            ctx.beginPath();
            const step = w / (chartData.length - 1);
            for (let i = 0; i < chartData.length; i++) {
                const x = i * step;
                const y = h - (chartData[i] / 100) * (h - 20) - 10;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }

            ctx.strokeStyle = '#00F0FF';
            ctx.lineWidth = 3;
            ctx.stroke();

            ctx.lineTo(w, h);
            ctx.lineTo(0, h);
            ctx.fillStyle = grad;
            ctx.fill();
        }

        // Continuous Telemetry & Alarm Poller (Runs every 300ms)
        setInterval(async () => {
            try {
                const res = await fetch('/telemetry');
                const d = await res.json();

                const eyesVal = document.getElementById('eyes-val');
                const poseVal = document.getElementById('pose-val');
                const actionVal = document.getElementById('action-val');
                const engineVal = document.getElementById('engine-val');
                const speedProfileVal = document.getElementById('speed-profile-val');
                const perclosVal = document.getElementById('perclos-val');
                const perclosBar = document.getElementById('perclos-bar-fill');
                const perclosLevel = document.getElementById('perclos-level');
                const blackboxCnt = document.getElementById('btn-blackbox-cnt');

                eyesVal.textContent = d.eyes_status.toUpperCase();
                poseVal.textContent = `${d.yaw > 0 ? '+' : ''}${d.yaw}° / ${d.pitch > 0 ? '+' : ''}${d.pitch}°`;
                actionVal.textContent = d.action.toUpperCase();
                engineVal.textContent = 'MEDIAPIPE+YOLO';
                blackboxCnt.textContent = d.total_blackbox_records || 0;

                speedProfileVal.textContent = `${d.speed_profile.split(' ')[0]} (${d.grace_period}S GRACE)`;
                if (d.grace_period <= 0.8) speedProfileVal.style.color = 'var(--neon-red)';
                else if (d.grace_period <= 1.5) speedProfileVal.style.color = 'var(--neon-yellow)';
                else speedProfileVal.style.color = 'var(--neon-green)';

                eyesVal.style.color = (d.eyes_status.includes('closed') || d.eyes_status.includes('DROWSY') || d.eyes_status.includes('eye closed') || d.eyes_status.includes('OCCLUDED') || d.eyes_status.includes('UNOBSERVED')) ? 'var(--neon-red)' : 'var(--neon-green)';
                actionVal.style.color = (d.action !== 'Normal' && d.action !== 'NORMAL DRIVING') ? 'var(--neon-red)' : 'var(--neon-green)';

                // Update PERCLOS Widget
                const pVal = d.perclos || 0.0;
                perclosVal.textContent = `${pVal.toFixed(1)}%`;
                perclosBar.style.width = `${Math.min(100, (pVal / 25.0) * 100)}%`;
                perclosLevel.textContent = d.perclos_level;

                if (pVal < 8.0) {
                    perclosVal.style.color = 'var(--neon-green)';
                    perclosBar.style.backgroundColor = 'var(--neon-green)';
                    perclosLevel.style.color = 'var(--neon-green)';
                } else if (pVal < 15.0) {
                    perclosVal.style.color = 'var(--neon-yellow)';
                    perclosBar.style.backgroundColor = 'var(--neon-yellow)';
                    perclosLevel.style.color = 'var(--neon-yellow)';
                } else {
                    perclosVal.style.color = 'var(--neon-red)';
                    perclosBar.style.backgroundColor = 'var(--neon-red)';
                    perclosLevel.style.color = 'var(--neon-red)';
                }

                document.getElementById('hud-fps').textContent = `FPS: ${d.fps}`;
                document.getElementById('hud-timer').textContent = `SESSION: ${d.session_time}`;

                document.getElementById('cnt-yawns').textContent = d.total_yawns;
                document.getElementById('cnt-drowsy').textContent = d.total_drowsy;
                document.getElementById('cnt-distractions').textContent = d.total_distractions;

                const score = d.safety_score;
                document.getElementById('safety-score').textContent = `${score}%`;
                const bar = document.getElementById('gauge-bar');
                const circumference = 2 * Math.PI * 70;
                const offset = circumference - (score / 100) * circumference;
                bar.style.strokeDashoffset = offset;

                if (score >= 75) bar.style.stroke = 'var(--neon-green)';
                else if (score >= 40) bar.style.stroke = 'var(--neon-yellow)';
                else bar.style.stroke = 'var(--neon-red)';

                const vBox = document.getElementById('video-box');
                const banner = document.getElementById('hazard-banner');
                const breakBanner = document.getElementById('break-banner');
                const medicalBanner = document.getElementById('medical-banner');

                // Medical Incapacitation Check
                if (d.medical_emergency && !medicalOverridden) {
                    medicalBanner.style.display = 'block';
                    vBox.classList.add('hazard');
                    playMedicalEmergencySiren();
                    speakVoiceAlert("Warning! Driver incapacitation detected! Initiating emergency protocols.");
                } else {
                    medicalBanner.style.display = 'none';
                }

                if (d.take_a_break && !breakAcknowledged) {
                    breakBanner.style.display = 'block';
                    speakVoiceAlert("Warning! Extreme driver fatigue detected. Please pull over and take a break immediately.");
                }

                if (d.risk_level === 'CRITICAL HAZARD') {
                    vBox.classList.add('hazard');
                    banner.style.display = 'block';
                    banner.style.background = 'rgba(255, 0, 85, 0.90)';

                    let hazardDesc = '🚨 DROWSINESS HAZARD';
                    if (d.action.includes('OCCLUDED') || d.action.includes('BLOCKED') || d.action.includes('UNOBSERVED')) hazardDesc = '🚨 CAMERA BLOCKED / DRIVER MISSING';
                    else if (d.action.includes('PHONE') || d.action.includes('texting') || d.action.includes('phonecall')) hazardDesc = '🚨 PHONE DISTRACTION HAZARD';
                    else if (d.action.includes('PERCLOS')) hazardDesc = '🚨 CRITICAL 60S PERCLOS FATIGUE';
                    else if (d.action.includes('LOOKING')) hazardDesc = `🚨 ${d.action.toUpperCase()} (${d.speed_kmh} KM/H)`;
                    else if (d.action !== 'Normal' && d.action !== 'NORMAL DRIVING') hazardDesc = `🚨 ${d.action.toUpperCase()}`;

                    banner.textContent = hazardDesc;
                    if (!d.medical_emergency) playHazardAlarm();

                } else if (d.risk_level === 'WARNING') {
                    vBox.classList.remove('hazard');
                    banner.style.display = 'block';
                    banner.style.background = 'rgba(245, 158, 11, 0.9)';
                    banner.textContent = `⚠️ DRIVER DISTRACTED: ${d.action.toUpperCase()}`;
                    if (!d.medical_emergency) playWarningChime();

                } else {
                    if (!d.medical_emergency) vBox.classList.remove('hazard');
                    banner.style.display = 'none';
                }

                if (d.video_complete && !modalShown) {
                    modalShown = true;
                    showScorecardModal();
                }

                chartData.shift();
                chartData.push(score);
                renderChart();

                if (d.event_log && d.event_log.length > 0) {
                    const tbody = document.getElementById('log-body');
                    const rows = d.event_log.slice(-10).reverse().map(ev => {
                        let badgeClass = 'log-badge-info';
                        if (ev.severity === 'HAZARD') badgeClass = 'log-badge-hazard';
                        else if (ev.severity === 'WARNING') badgeClass = 'log-badge-warning';
                        return `<tr><td>${ev.time}</td><td class="${badgeClass}">${ev.severity}</td><td>${ev.desc}</td></tr>`;
                    }).join('');
                    tbody.innerHTML = rows;
                }
            } catch (e) {}
        }, 300);
    </script>
</body>
</html>
"""

def get_webcam_frame():
    global global_cam_cap, active_cap_index, last_cam_check_time, selected_camera_index, cam_consecutive_failures
    with cam_lock:
        current_time = time.time()
        # If camera index changed or cap is not opened
        if global_cam_cap is None or not global_cam_cap.isOpened() or active_cap_index != selected_camera_index:
            if current_time - last_cam_check_time > 1.0:
                last_cam_check_time = current_time
                if global_cam_cap is not None:
                    try:
                        global_cam_cap.release()
                    except Exception:
                        pass
                    global_cam_cap = None

                # Check chosen index first, then fallback to others
                candidate_indices = [selected_camera_index] + [i for i in [0, 1, 2] if i != selected_camera_index]
                for idx in candidate_indices:
                    for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                        try:
                            temp_cap = cv2.VideoCapture(idx, backend)
                            if temp_cap.isOpened():
                                ret, frame = temp_cap.read()
                                if ret and frame is not None:
                                    global_cam_cap = temp_cap
                                    active_cap_index = idx
                                    selected_camera_index = idx
                                    cam_consecutive_failures = 0
                                    telemetry['camera_index'] = idx
                                    telemetry['camera_label'] = "LAPTOP CAM (0)" if idx == 0 else f"USB WEBCAM ({idx})"
                                    return frame
                                temp_cap.release()
                        except Exception:
                            pass
            return None
        else:
            try:
                ret, frame = global_cam_cap.read()
                if ret and frame is not None:
                    cam_consecutive_failures = 0
                    return frame
                else:
                    cam_consecutive_failures += 1
                    if cam_consecutive_failures >= 15:
                        try:
                            global_cam_cap.release()
                        except Exception:
                            pass
                        global_cam_cap = None
                        active_cap_index = None
                        cam_consecutive_failures = 0
                    return None
            except Exception:
                cam_consecutive_failures += 1
                if cam_consecutive_failures >= 15:
                    global_cam_cap = None
                    active_cap_index = None
                    cam_consecutive_failures = 0
                return None

def generate_frames():
    global current_raw_frame, telemetry, prev_yawn, prev_drowsy, prev_distraction, prev_occlusion, prev_perclos_hazard, safety_scores_history, video_analysis_complete
    global std_no_face_duration, std_head_diverted_duration, std_phone_duration, std_eyes_closed_duration, current_vehicle_speed
    global driver_slump_duration, prev_medical_emergency, night_vision_mode, last_log_record_time, telematics_log_buffer

    initialize_models()
    video_cap = None
    frame_interval = 1.0 / 25.0
    ptime = time.time()
    last_loop_time = time.time()

    while True:
        loop_start = time.time()
        dt = min(max(loop_start - last_loop_time, 0.001), 0.3)
        last_loop_time = loop_start

        speed_grace_period, speed_profile = get_speed_thresholds(current_vehicle_speed)

        if video_source_mode == 'video':
            if uploaded_video_path and os.path.exists(uploaded_video_path):
                if video_cap is None or not video_cap.isOpened():
                    video_cap = cv2.VideoCapture(uploaded_video_path)
                    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
                    frame_interval = 1.0 / max(video_fps if video_fps and video_fps > 0 else 25.0, 1.0)

                success, image = video_cap.read()
                if not success or image is None:
                    video_analysis_complete = True
                    err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(err_frame, "VIDEO ANALYSIS COMPLETE", (130, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2)
                    cv2.putText(err_frame, f"Driver Safety Score: {telemetry['safety_score']}%", (160, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(err_frame, f"Yawns: {telemetry['total_yawns']}  Drowsy: {telemetry['total_drowsy']}  Hazards: {telemetry['total_distractions']}", (90, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
                    ret, buf = cv2.imencode('.jpg', err_frame)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                    time.sleep(1.0)
                    continue
            else:
                err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(err_frame, "NO VIDEO LOADED. PLEASE UPLOAD A VIDEO.", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                ret, buf = cv2.imencode('.jpg', err_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue
        else:
            image = get_webcam_frame()
            if image is None:
                cam_name = "LAPTOP CAM (0)" if selected_camera_index == 0 else f"USB WEBCAM ({selected_camera_index})"
                err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(err_frame, f"CONNECTING TO {cam_name}...", (90, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
                ret, buf = cv2.imencode('.jpg', err_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                time.sleep(0.4)
                continue

        current_raw_frame = image.copy()

        # Compute Ambient Luminance & Apply Night-Vision Optimizer (CLAHE)
        gray_temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ambient_lum = float(np.mean(gray_temp))
        telemetry['ambient_lum'] = round(ambient_lum, 1)

        apply_nv = (night_vision_mode == 'on') or (night_vision_mode == 'auto' and ambient_lum < 45.0)
        telemetry['night_vision_active'] = apply_nv
        telemetry['night_vision_mode'] = night_vision_mode

        if apply_nv:
            image = apply_night_vision_clahe(image)

        # Compute FPS
        ctime = time.time()
        fps = int(1.0 / max((ctime - ptime), 0.001))
        ptime = ctime

        timestamp_str = datetime.now().strftime("%H:%M:%S")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Global YOLO Phone Detection (Confidence threshold >= 0.55)
        phone_detected = False
        if yolo_model is not None:
            try:
                yolo_result = yolo_model(rgb_image)
                if yolo_result is not None and len(yolo_result.xyxy[0]) > 0:
                    conf_val = float(yolo_result.xyxy[0][0][4])
                    if conf_val >= 0.55:
                        phone_detected = True
            except Exception:
                pass

        # ----------------------------------------------------
        # Standard MediaPipe + YOLOv5 Pipeline
        # ----------------------------------------------------
        eyes_status = 'Normal'
        yawn_status = 'Normal'
        action = 'NORMAL DRIVING'
        face_detected = False
        perclos_val, perclos_level = 0.0, 'CALIBRATING...'

        try:
            if facial_tracker is None:
                raise RuntimeError("MediaPipe facial tracker is unavailable")
            facial_tracker.process_frame(image)
            if facial_tracker.detected:
                face_detected = True
                std_no_face_duration = max(0.0, std_no_face_duration - (2.5 * dt))
                eyes_status = facial_tracker.eyes_status if facial_tracker.eyes_status else 'Normal'
                yawn_status = facial_tracker.yawn_status if facial_tracker.yawn_status else 'Normal'
            else:
                std_no_face_duration += dt

            # Sudden Medical Incapacitation Check
            is_slumping = (facial_tracker.pitch < -40.0 or abs(facial_tracker.roll) > 35.0)
            if is_slumping:
                driver_slump_duration += dt
            else:
                driver_slump_duration = max(0.0, driver_slump_duration - (2.0 * dt))

            # Sustained Eye Closure Timer
            is_eye_closed_instant = (eyes_status == 'eye closed')
            if is_eye_closed_instant:
                std_eyes_closed_duration += dt
            else:
                std_eyes_closed_duration = max(0.0, std_eyes_closed_duration - (3.0 * dt))

            # Update Euro NCAP PERCLOS
            perclos_val, perclos_level = update_perclos(is_eye_closed_instant)

            # Sustained Head Pose Diversion Timer
            if facial_tracker.head_direction != 'FORWARD' and facial_tracker.head_direction != 'UNKNOWN':
                std_head_diverted_duration += dt
            else:
                std_head_diverted_duration = max(0.0, std_head_diverted_duration - (2.0 * dt))

            # Sustained Phone Usage Timer
            if phone_detected:
                std_phone_duration += dt
                resized_rgb = cv2.resize(rgb_image, (224, 224))
                if model is not None:
                    y = model.predict(resized_rgb)
                    result = np.argmax(y, axis=1)
                    action = 'phonecall' if result[0] == 0 else 'texting'
                else:
                    action = 'PHONE DETECTED'
            else:
                std_phone_duration = max(0.0, std_phone_duration - (2.0 * dt))
                if std_head_diverted_duration > (speed_grace_period * 0.8):
                    action = facial_tracker.head_direction
                else:
                    action = 'NORMAL DRIVING'
        except Exception as err:
            print(f"[Standard Engine Error] {err}")

        # Calculate Safety Score with Speed-Adaptive Grace Periods & Medical Alert
        score = 100
        medical_emergency = (driver_slump_duration > 3.0)
        is_occluded = (std_no_face_duration > 2.5)
        is_drowsy = (std_eyes_closed_duration > 1.0 or perclos_val >= 15.0)
        is_phone = (std_phone_duration > 0.8)
        is_diverted = (std_head_diverted_duration > speed_grace_period)

        speed_mps = (current_vehicle_speed * 1000.0) / 3600.0
        blind_distance = round(std_head_diverted_duration * speed_mps, 1)

        if medical_emergency:
            score = 0
            action = 'MEDICAL INCAPACITATION / SLUMP'
            risk_level = 'CRITICAL HAZARD'
            if not prev_medical_emergency:
                telemetry['event_log'].append({'time': timestamp_str, 'severity': 'HAZARD', 'desc': '🚨 EMERGENCY: Sudden driver slump / incapacitation detected!'})
                capture_blackbox_incident(image, "MEDICAL EMERGENCY / DRIVER SLUMP", 0, current_vehicle_speed)
            prev_medical_emergency = True
        elif is_occluded:
            prev_medical_emergency = False
            face_detected = False
            score = 0
            action = 'CAMERA BLOCKED / FACE OCCLUDED'
            eyes_status = 'FACE UNOBSERVED'
            risk_level = 'CRITICAL HAZARD'
        elif is_drowsy:
            prev_medical_emergency = False
            score = 15
            eyes_status = 'DROWSY / ASLEEP'
            risk_level = 'CRITICAL HAZARD'
        elif is_phone:
            prev_medical_emergency = False
            score = 30
            risk_level = 'CRITICAL HAZARD'
        elif is_diverted:
            prev_medical_emergency = False
            score = 35 if std_head_diverted_duration > (speed_grace_period * 1.5) else 55
            risk_level = 'CRITICAL HAZARD'
        else:
            prev_medical_emergency = False
            if yawn_status == 'yawning':
                score -= 30
                risk_level = 'WARNING'
            elif perclos_val >= 8.0:
                score -= 20
                risk_level = 'WARNING'
            elif 'gaze' in eyes_status and eyes_status != 'gazing center':
                score -= 15
                risk_level = 'WARNING'
            else:
                risk_level = 'SAFE'

        score = max(0, min(100, score))
        safety_scores_history.append(score)

        is_yawning = (yawn_status == 'yawning')
        if is_yawning and not prev_yawn:
            telemetry['total_yawns'] += 1
            telemetry['event_log'].append({'time': timestamp_str, 'severity': 'WARNING', 'desc': 'MediaPipe: Yawn / driver fatigue detected.'})
        prev_yawn = is_yawning

        if is_drowsy and not prev_drowsy:
            telemetry['total_drowsy'] += 1
            telemetry['event_log'].append({'time': timestamp_str, 'severity': 'HAZARD', 'desc': 'MediaPipe: Driver eyes closed (drowsiness).'})
        prev_drowsy = is_drowsy

        is_distraction = (is_phone or is_diverted or is_occluded or medical_emergency)
        if is_distraction and not prev_distraction:
            telemetry['total_distractions'] += 1
            telemetry['event_log'].append({'time': timestamp_str, 'severity': 'HAZARD', 'desc': f'MediaPipe: Hazard detected ({action}) at {current_vehicle_speed} km/h.'})
        prev_distraction = is_distraction

        # Automated Blackbox Incident Capture Trigger
        if risk_level == 'CRITICAL HAZARD':
            capture_blackbox_incident(image, action, score, current_vehicle_speed)

        telemetry['engine'] = 'standard'
        telemetry['eyes_status'] = eyes_status
        telemetry['yawn_status'] = yawn_status
        telemetry['action'] = action
        telemetry['safety_score'] = score
        telemetry['risk_level'] = risk_level
        telemetry['fps'] = fps
        telemetry['speed_kmh'] = current_vehicle_speed
        telemetry['speed_profile'] = speed_profile
        telemetry['grace_period'] = speed_grace_period
        telemetry['blind_distance_m'] = blind_distance
        telemetry['camera_index'] = selected_camera_index
        telemetry['camera_label'] = "LAPTOP CAM (0)" if selected_camera_index == 0 else f"USB WEBCAM ({selected_camera_index})"
        telemetry['face_detected'] = face_detected

        pitch_val = round(facial_tracker.pitch, 1) if facial_tracker is not None else 0.0
        yaw_val = round(facial_tracker.yaw, 1) if facial_tracker is not None else 0.0
        roll_val = round(facial_tracker.roll, 1) if facial_tracker is not None else 0.0
        head_dir = facial_tracker.head_direction if facial_tracker is not None else 'UNKNOWN'

        telemetry['pitch'] = pitch_val
        telemetry['yaw'] = yaw_val
        telemetry['roll'] = roll_val
        telemetry['perclos'] = perclos_val
        telemetry['perclos_level'] = perclos_level
        telemetry['medical_emergency'] = medical_emergency
        telemetry['take_a_break'] = (telemetry['total_drowsy'] >= 3)
        telemetry['video_mode'] = video_source_mode
        telemetry['video_complete'] = video_analysis_complete

        # Standard Engine HUD Overlay
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, 0), (w, 35), (20, 24, 30), -1)
        nv_tag = " [🌙 NV ON]" if apply_nv else ""
        cv2.putText(image, f"[🤖 MEDIAPIPE+YOLO]{nv_tag} {current_vehicle_speed} KM/H", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 136), 2)
        cv2.putText(image, f"SCORE: {score}%", (w - 170, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 136) if score >= 75 else (0, 0, 255), 2)

        # Head Pose 3D Reticle Widget (Top-Right)
        center_x, center_y = w - 75, 95
        cv2.circle(image, (center_x, center_y), 32, (50, 60, 70), 2)
        arrow_dx = int(np.clip(-yaw_val * 1.2, -28, 28))
        arrow_dy = int(np.clip(pitch_val * 1.2, -28, 28))
        reticle_color = (0, 240, 255) if not is_diverted else (0, 0, 255)
        cv2.line(image, (center_x, center_y), (center_x + arrow_dx, center_y + arrow_dy), reticle_color, 3)
        cv2.circle(image, (center_x + arrow_dx, center_y + arrow_dy), 4, reticle_color, -1)
        cv2.putText(image, f"YAW: {yaw_val:+.1f}°", (center_x - 55, center_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(image, f"PITCH: {pitch_val:+.1f}°", (center_x - 55, center_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Bottom Status HUD Box
        hud_w, hud_h = 340, 85
        color_stat = (0, 255, 100) if risk_level == 'SAFE' else ((0, 200, 255) if risk_level == 'WARNING' else (0, 0, 255))
        cv2.rectangle(image, (15, h - hud_h - 15), (15 + hud_w, h - 15), (15, 18, 22), -1)
        cv2.rectangle(image, (15, h - hud_h - 15), (15 + hud_w, h - 15), color_stat, 2)
        cv2.putText(image, f"EYES: {eyes_status.upper()} (P:{perclos_val}%)", (25, h - hud_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"POSE: {head_dir} ({speed_grace_period}s)", (25, h - hud_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"ACTION: {action.upper()}", (25, h - hud_h + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255) if action in ['Normal', 'NORMAL DRIVING'] else (0, 0, 255), 1)

        # Telematics Periodic Log Recorder (every 1.0s)
        now_time = time.time()
        if now_time - last_log_record_time >= 1.0:
            last_log_record_time = now_time
            with telematics_lock:
                telematics_log_buffer.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'engine': telemetry['engine'],
                    'camera_index': telemetry['camera_index'],
                    'speed_kmh': telemetry['speed_kmh'],
                    'safety_score': telemetry['safety_score'],
                    'risk_level': telemetry['risk_level'],
                    'perclos_pct': telemetry['perclos'],
                    'eyes_status': telemetry['eyes_status'],
                    'driver_action': telemetry['action'],
                    'yaw_deg': telemetry['yaw'],
                    'pitch_deg': telemetry['pitch'],
                    'ambient_lum': telemetry['ambient_lum'],
                    'night_vision': telemetry['night_vision_active'],
                    'total_yawns': telemetry['total_yawns'],
                    'total_drowsy': telemetry['total_drowsy'],
                    'total_distractions': telemetry['total_distractions']
                })
                if len(telematics_log_buffer) > 7200:
                    telematics_log_buffer.pop(0)

        ret, buffer = cv2.imencode('.jpg', image)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        if video_source_mode == 'video':
            elapsed_processing = time.time() - loop_start
            sleep_time = frame_interval - elapsed_processing
            if sleep_time > 0:
                time.sleep(sleep_time)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def get_telemetry():
    elapsed = int(time.time() - telemetry['session_start'])
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    session_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    resp = dict(telemetry)
    resp['session_time'] = session_time
    resp['total_blackbox_records'] = len(blackbox_incidents_list)
    return jsonify(resp)


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_initialized': models_initialized,
        'action_model_available': model is not None,
        'phone_detector_available': yolo_model is not None,
        'facial_tracker_available': facial_tracker is not None,
        'model_load_errors': list(model_load_errors),
    })


@app.errorhandler(413)
def upload_too_large(_error):
    max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return jsonify({'success': False, 'error': f'Video exceeds the {max_mb} MB upload limit'}), 413

@app.route('/set_camera', methods=['POST'])
@require_api_token
def set_camera():
    global selected_camera_index, global_cam_cap, active_cap_index, last_cam_check_time, cam_consecutive_failures
    data = request.get_json(silent=True) or {}
    try:
        idx = int(data.get('camera_index', 0))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'camera_index must be an integer'}), 400
    if not 0 <= idx <= 9:
        return jsonify({'success': False, 'error': 'camera_index must be between 0 and 9'}), 400
    with cam_lock:
        if idx == active_cap_index and global_cam_cap is not None and global_cam_cap.isOpened():
            selected_camera_index = idx
            cam_label = "LAPTOP CAM (0)" if idx == 0 else f"USB WEBCAM ({idx})"
            return jsonify({'success': True, 'camera_index': idx, 'label': cam_label})

        # Test if requested camera index is available
        test_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        opened = False
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None:
                opened = True
            test_cap.release()

        if not opened:
            curr_idx = active_cap_index if active_cap_index is not None else 0
            curr_label = "LAPTOP CAM (0)" if curr_idx == 0 else f"USB WEBCAM ({curr_idx})"
            return jsonify({
                'success': False,
                'error': f'Camera {idx} is not connected or accessible.',
                'active_index': curr_idx,
                'active_label': curr_label
            }), 404

        if global_cam_cap is not None:
            try:
                global_cam_cap.release()
            except Exception:
                pass
            global_cam_cap = None
        selected_camera_index = idx
        active_cap_index = None
        cam_consecutive_failures = 0
        last_cam_check_time = 0.0  # Force immediate reconnection

    cam_label = "LAPTOP CAM (0)" if idx == 0 else f"USB WEBCAM ({idx})"
    telemetry['camera_index'] = selected_camera_index
    telemetry['camera_label'] = cam_label
    telemetry['event_log'].append({
        'time': datetime.now().strftime("%H:%M:%S"),
        'severity': 'INFO',
        'desc': f'Switched video camera device to {cam_label}.'
    })
    return jsonify({'success': True, 'camera_index': selected_camera_index, 'label': cam_label})

@app.route('/set_speed', methods=['POST'])
@require_api_token
def set_speed():
    global current_vehicle_speed
    data = request.get_json(silent=True) or {}
    try:
        speed = int(data.get('speed', 60))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'speed must be an integer'}), 400
    current_vehicle_speed = max(0, min(200, speed))
    grace, profile = get_speed_thresholds(current_vehicle_speed)
    telemetry['speed_kmh'] = current_vehicle_speed
    telemetry['speed_profile'] = profile
    telemetry['grace_period'] = grace
    return jsonify({'success': True, 'speed': current_vehicle_speed, 'grace_period': grace, 'profile': profile})

@app.route('/set_night_vision', methods=['POST'])
@require_api_token
def set_night_vision():
    global night_vision_mode
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'auto')
    if mode in ['auto', 'on', 'off']:
        night_vision_mode = mode
        telemetry['night_vision_mode'] = mode
        return jsonify({'success': True, 'mode': night_vision_mode})
    return jsonify({'success': False, 'error': 'Invalid night vision mode'}), 400

@app.route('/api/export_telematics')
def export_telematics():
    with telematics_lock:
        data_copy = list(telematics_log_buffer)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Timestamp', 'AI Engine', 'Camera Index', 'Speed (km/h)', 'Attention Score (%)', 'Risk Level',
        'Euro NCAP PERCLOS (%)', 'Eyes Status', 'Driver Action', 'Yaw (deg)', 'Pitch (deg)',
        'Ambient Lum', 'Night Vision Active', 'Total Yawns', 'Total Drowsy', 'Total Hazards'
    ])

    for row in data_copy:
        writer.writerow([
            row['timestamp'], row['engine'], row.get('camera_index', 0), row['speed_kmh'], row['safety_score'], row['risk_level'],
            row['perclos_pct'], row['eyes_status'], row['driver_action'], row['yaw_deg'], row['pitch_deg'],
            row['ambient_lum'], row['night_vision'], row['total_yawns'], row['total_drowsy'], row['total_distractions']
        ])

    csv_data = output.getvalue()
    filename = f"telematics_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    response = make_response(csv_data)
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    response.headers['Content-type'] = 'text/csv'
    return response

@app.route('/api/incidents')
def get_incidents():
    return jsonify({'incidents': list(reversed(blackbox_incidents_list))})

@app.route('/incidents/<path:filename>')
def serve_incident_image(filename):
    return send_from_directory(incidents_dir, filename)

@app.route('/api/clear_incidents', methods=['POST'])
@require_api_token
def clear_incidents():
    global blackbox_incidents_list
    blackbox_incidents_list = []
    telemetry['total_blackbox_records'] = 0
    for f in glob.glob(os.path.join(incidents_dir, "*.jpg")):
        try:
            os.remove(f)
        except Exception:
            pass
    return jsonify({'success': True})

@app.route('/reset_break', methods=['POST'])
@require_api_token
def reset_break():
    telemetry['take_a_break'] = False
    telemetry['total_drowsy'] = 0
    return jsonify({'success': True})

@app.route('/reset_medical', methods=['POST'])
@require_api_token
def reset_medical():
    global driver_slump_duration, prev_medical_emergency
    driver_slump_duration = 0.0
    prev_medical_emergency = False
    telemetry['medical_emergency'] = False
    return jsonify({'success': True})

@app.route('/set_engine', methods=['POST'])
@require_api_token
def set_engine():
    global ai_engine_mode
    ai_engine_mode = 'standard'
    telemetry['engine'] = 'standard'
    return jsonify({'success': True, 'engine': 'standard'})

@app.route('/upload_video', methods=['POST'])
@require_api_token
def upload_video():
    global video_source_mode, uploaded_video_path, uploaded_video_name, video_analysis_complete
    if 'video_file' not in request.files:
        return jsonify({'success': False, 'error': 'No video file uploaded'}), 400

    file = request.files['video_file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    original_filename = secure_filename(file.filename)
    if not original_filename or not is_allowed_video(original_filename):
        return jsonify({'success': False, 'error': 'Unsupported video type'}), 400

    filename = f"{uuid.uuid4().hex}_{original_filename}"

    save_path = os.path.join(uploads_dir, filename)
    file.save(save_path)

    uploaded_video_path = save_path
    uploaded_video_name = filename
    video_source_mode = 'video'
    video_analysis_complete = False

    reset_telemetry()
    telemetry['event_log'].append({
        'time': datetime.now().strftime("%H:%M:%S"),
        'severity': 'INFO',
        'desc': f'Uploaded video "{filename}" loaded for analysis.'
    })

    return jsonify({
        'success': True,
        'filename': filename,
        'mode': 'video'
    })

@app.route('/set_source', methods=['POST'])
@require_api_token
def set_source():
    global video_source_mode, video_analysis_complete
    data = request.get_json(silent=True) or {}
    source = data.get('source', 'webcam')
    if source in ['webcam', 'video']:
        video_source_mode = source
        video_analysis_complete = False
        reset_telemetry()
        telemetry['event_log'].append({
            'time': datetime.now().strftime("%H:%M:%S"),
            'severity': 'INFO',
            'desc': f'Switched source to {source.upper()}.'
        })
        return jsonify({'success': True, 'mode': video_source_mode})
    return jsonify({'success': False, 'error': 'Invalid source'}), 400

@app.route('/video_summary')
def get_video_summary():
    global safety_scores_history, telemetry, uploaded_video_name
    avg_score = int(sum(safety_scores_history) / max(len(safety_scores_history), 1)) if safety_scores_history else telemetry['safety_score']

    if avg_score >= 85 and telemetry['total_distractions'] == 0:
        grade = 'A+ (EXCELLENT)'
        rating = 'SAFE & ATTENTIVE DRIVER'
    elif avg_score >= 70:
        grade = 'B (MODERATE RISK)'
        rating = 'OCCASIONAL FATIGUE/INATTENTION'
    elif avg_score >= 50:
        grade = 'C (HIGH RISK)'
        rating = 'FREQUENT DISTRACTIONS & YAWNING'
    else:
        grade = 'F (CRITICAL HAZARD)'
        rating = 'DANGEROUS DRIVING BEHAVIOR'

    return jsonify({
        'filename': uploaded_video_name,
        'average_score': avg_score,
        'safety_grade': grade,
        'rating': rating,
        'total_yawns': telemetry['total_yawns'],
        'total_drowsy': telemetry['total_drowsy'],
        'total_distractions': telemetry['total_distractions'],
        'total_events': len(telemetry['event_log']),
        'is_complete': video_analysis_complete
    })

@app.route('/snapshot')
def get_snapshot():
    global current_raw_frame
    if current_raw_frame is not None:
        ret, buf = cv2.imencode('.jpg', current_raw_frame)
        return Response(buf.tobytes(), mimetype='image/jpeg')
    return "No frame available", 404

if __name__ == '__main__':
    print("Pre-initializing AI models & camera...")
    initialize_models()
    get_webcam_frame()
    print("Starting Next-Gen DMS Web Server on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)
