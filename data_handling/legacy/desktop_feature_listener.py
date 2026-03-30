import serial
import json
import joblib
import numpy as np
import pandas as pd
import time
import sys
import os
import pygame
from collections import deque # <--- NEW IMPORT FOR THE X/10 RULE
from data_handling.project_paths import XGB_MODEL_PATH

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'   # <--- CHECK YOUR DEVICE MANAGER
BAUD_RATE = 115200     
MODEL_PATH = os.fspath(XGB_MODEL_PATH)
AUDIO_PATH = r"C:\Users\pratt\Downloads\snoopspeechify_wattuptho-[AudioTrimmer.com].mp3"

CONFIDENCE_THRESHOLD = 0.775 
AUDIO_COOLDOWN = 3.0        # seconds   

# --- NEW STABILITY SETTINGS (The X/10 Rule) ---
WINDOW_SIZE = 8   # The "10" in "X/10"
REQUIRED_HITS = 4  # The "X". We need 5 positive frames out of the last 8 to trigger.
# ---------------------

# MUST MATCH TRAINING SCRIPT EXACTLY (Order matters!)
FEATURE_NAMES = [
    "aspect_ratio_hw", 
    "extent", 
    "solidity", 
    "circularity", 
    "angle",
    "d_top_mid", 
    "d_top_left", 
    "d_top_right", 
    "v_contrast", 
    "palm_contrast"
]

def setup_audio():
    """Initializes audio mixer and loads the sound file."""
    if not os.path.exists(AUDIO_PATH):
        print(f"WARNING: Audio file not found at: {AUDIO_PATH}")
        return None
    try:
        pygame.mixer.init()
        sound = pygame.mixer.Sound(AUDIO_PATH)
        print("Audio loaded successfully!")
        return sound
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found.")
        sys.exit(1)
    
    print(f"Loading model: {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def run_inference():
    # 1. Setup Resources
    clf = load_model()
    sound_effect = setup_audio()
    test_sound = False
    
    # Audio Test
    if test_sound and sound_effect:
        print("\nðŸŽµ TESTING AUDIO...")
        sound_effect.play()
        time.sleep(1)
        print("Starting camera listener...\n")
    
    last_played_time = 0 

    # --- INITIALIZE HISTORY BUFFER ---
    # This list will automatically pop the oldest item when full
    history_buffer = deque(maxlen=WINDOW_SIZE) 

    # 2. Connect to Camera
    try:
        print(f"Connecting to {SERIAL_PORT}...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print("Connected! Waiting for data stream...")
    except serial.SerialException as e:
        print(f"\nCould not open serial port {SERIAL_PORT}.")
        return

    # 3. Main Loop
    while True:
        try:
            if ser.in_waiting > 0:
                line_bytes = ser.readline()
                try:
                    line = line_bytes.decode('utf-8').strip()
                except UnicodeDecodeError:
                    continue 

                if line.startswith('{') and '"d":' in line:
                    try:
                        data = json.loads(line)
                        features_list = data.get("d")

                        # UPDATED CHECK: Expect 10 features now
                        if features_list and len(features_list) == len(FEATURE_NAMES):
                            
                            # Convert list to DataFrame with names matching training
                            feat_df = pd.DataFrame([features_list], columns=FEATURE_NAMES)
                            
                            # Predict
                            probs = clf.predict_proba(feat_df)[0]
                            peace_prob = probs[1]
                            
                            # --- STABILITY LOGIC ---
                            # 1. Did this single frame pass the threshold?
                            frame_result = 1 if peace_prob > CONFIDENCE_THRESHOLD else 0
                            
                            # 2. Add result to history (removes oldest if len > 10)
                            history_buffer.append(frame_result)
                            
                            # 3. Calculate "Score" (How many 1s in the last 10 frames?)
                            current_score = sum(history_buffer)
                            
                            print(f"Score: {current_score}/{WINDOW_SIZE} | Conf: {peace_prob:.2f}")

                            # 4. Trigger Audio ONLY if we meet the "X/10" rule
                            if current_score >= REQUIRED_HITS:
                                
                                # Audio Logic
                                current_time = time.time()
                                if sound_effect and (current_time - last_played_time > AUDIO_COOLDOWN):
                                    print(">>> ðŸŽµ PLAYING AUDIO! ðŸŽµ <<<")
                                    sound_effect.play()
                                    last_played_time = current_time
                                    
                                    # Optional: Clear buffer to prevent double triggering immediately
                                    # history_buffer.clear() 

                        else:
                            print(f"Warning: Data mismatch. Got {len(features_list) if features_list else 0} features, expected {len(FEATURE_NAMES)}.")

                    except json.JSONDecodeError:
                        pass 
                
                elif line and not line.startswith('{'):
                    # Print debug messages from OpenMV
                    print(f"[CAM]: {line}")

        except KeyboardInterrupt:
            print("\nStopping...")
            if ser.is_open: ser.close()
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    run_inference()
