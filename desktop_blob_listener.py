import serial
import struct
import cv2
import numpy as np
import pandas as pd
import joblib
import math
import pygame
import os
import time
from project_paths import RF_MODEL_PATH

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'   # <--- CHECK DEVICE MANAGER
BAUD_RATE = 115200     
MODEL_PATH = os.fspath(RF_MODEL_PATH)
AUDIO_PATH = r"C:\Users\pratt\Downloads\snoopspeechify_wattuptho-[AudioTrimmer.com].mp3"

CONFIDENCE_THRESHOLD = 0.6
AUDIO_COOLDOWN = 3.0

# MUST MATCH TRAINING EXACTLY
FEATURE_NAMES = [
    "AspectRatio", "Extent", "Solidity", "Circularity", 
    "Zone_TopMid_Density", "Zone_TopSides_Density", "X_Gap_Signal"
]

# --- 1. EXACT TRAINING LOGIC ---

def calculate_spatial_features(thresh_img, bbox):
    """Calculates spatial features using the Bounding Box indices."""
    x, y, w, h = bbox
    # Safety check
    if w <= 0 or h <= 0: return {}

    y_third = h // 3
    x_third = w // 3
    
    zone_top_mid = thresh_img[y : y + y_third, x + x_third : x + 2 * x_third]
    zone_top_left = thresh_img[y : y + y_third, x : x + x_third]
    zone_top_right = thresh_img[y : y + y_third, x + 2 * x_third : x + w]
    
    d_top_mid = cv2.countNonZero(zone_top_mid) / (zone_top_mid.size + 1e-5)
    d_top_sides = (cv2.countNonZero(zone_top_left) + cv2.countNonZero(zone_top_right)) / \
                  (zone_top_left.size + zone_top_right.size + 1e-5)

    # X-Projection
    hand_region = thresh_img[y : y + h, x : x + w]
    x_projection = np.sum(hand_region, axis=0, dtype=np.float32)
    max_val = np.max(x_projection)
    if max_val > 0: x_projection /= max_val

    mid_proj_val = np.mean(x_projection[x_third : 2 * x_third])
    side_proj_val = np.mean(np.concatenate([x_projection[:x_third], x_projection[2 * x_third:]]))
    
    gap_signal = side_proj_val / (mid_proj_val + 1e-5)

    return {
        "Zone_TopMid_Density": d_top_mid,
        "Zone_TopSides_Density": d_top_sides,
        "X_Gap_Signal": gap_signal
    }

def extract_features_exact(img):
    """
    Exact replica of your training function, but takes an image object
    instead of a filepath.
    """
    # 1. Threshold (Matches training: 30)
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
    
    # Get largest contour
    cnt = max(contours, key=cv2.contourArea)
    
    # Basic Geometry
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area == 0 or perimeter == 0: return None

    # Bounding Box
    x, y, w, h = cv2.boundingRect(cnt)
    
    rect_area = w * h
    aspect_ratio = float(h) / w  # H/W
    extent = float(area) / rect_area

    # Convex Hull
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = 0 if hull_area == 0 else float(area) / hull_area

    # Circularity
    circularity = (4 * math.pi * area) / (perimeter * perimeter)

    # Spatial Features
    spatial_feats = calculate_spatial_features(thresh, (x, y, w, h))

    return [
        aspect_ratio,
        extent,
        solidity,
        circularity,
        spatial_feats["Zone_TopMid_Density"],
        spatial_feats["Zone_TopSides_Density"],
        spatial_feats["X_Gap_Signal"]
    ]

# --- 2. SETUP ---

if not os.path.exists(MODEL_PATH):
    print("Model not found!")
    exit()
clf = joblib.load(MODEL_PATH)

audio_enabled = False
if os.path.exists(AUDIO_PATH):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(AUDIO_PATH)
    audio_enabled = True
    print("Audio Loaded.")

def recvall(ser, n):
    data = b''
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk: return None
        data += chunk
    return data

# --- 3. RUN LOOP ---

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected. Waiting for blobs...")
except Exception as e:
    print(f"Serial Error: {e}")
    exit()

last_played = 0

while True:
    try:
        # Sync Header
        while True:
            if ser.read(1) == b'I':
                if ser.read(1) == b'M':
                    if ser.read(1) == b'G':
                        break
        
        # Read Size
        size_data = recvall(ser, 4)
        if not size_data: continue
        img_size = struct.unpack("<I", size_data)[0]
        
        # Read Image
        img_data = recvall(ser, img_size)
        if not img_data: continue
        
        # Decode
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        
        if frame is not None:
            # EXTRACT
            feats = extract_features_exact(frame)
            
            if feats:
                # INFERENCE
                df = pd.DataFrame([feats], columns=FEATURE_NAMES)
                probs = clf.predict_proba(df)[0]
                peace_prob = probs[1]
                
                # OUTPUT
                if peace_prob > CONFIDENCE_THRESHOLD:
                    print(f"✌️ PEACE! ({peace_prob:.2f})")
                    
                    # Visual Feedback (Green Box)
                    display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(display, (0,0), (frame.shape[1], frame.shape[0]), (0,255,0), 2)
                    
                    if audio_enabled and (time.time() - last_played > AUDIO_COOLDOWN):
                        sound.play()
                        last_played = time.time()
                else:
                    print(f"Background ({peace_prob:.2f})")
                    display = frame
                
                # Show what the computer is analyzing
                cv2.imshow("Streamed Blob", cv2.resize(display, (200, 200)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"Error: {e}")
