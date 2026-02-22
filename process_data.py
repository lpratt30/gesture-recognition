import cv2
import os
import json
import numpy as np
import math

# --- CONFIGURATION ---
TARGET_FOLDERS = ["gesture_data/dataset_v0/dataset/positive_peace_sign", "gesture_data/dataset_v0/dataset/negative_peace_sign"] 
# ---------------------

def calculate_spatial_features(gray_img, bbox):
    """
    Calculates spatial features using Grayscale Mean Brightness.
    Matches OpenMV's get_statistics().mean() logic.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0: return {}

    y_third = h // 3
    x_third = w // 3
    
    # Define Slices
    # Note: We use the raw grayscale image (gray_img), NOT a binary thresh
    zone_top_mid = gray_img[y : y + y_third, x + x_third : x + 2 * x_third]
    zone_top_left = gray_img[y : y + y_third, x : x + x_third]
    zone_top_right = gray_img[y : y + y_third, x + 2 * x_third : x + w]
    
    # --- 1. Zonal Density (Mean Brightness) ---
    # On OpenMV: get_statistics().mean() / 255.0 includes background (0).
    # This acts as a density proxy.
    d_top_mid = np.mean(zone_top_mid) / 255.0 if zone_top_mid.size > 0 else 0
    
    val_left = np.mean(zone_top_left) / 255.0 if zone_top_left.size > 0 else 0
    val_right = np.mean(zone_top_right) / 255.0 if zone_top_right.size > 0 else 0
    d_top_sides = (val_left + val_right) / 2.0

    # --- 2. Gap Signal (Vertical Mean Brightness) ---
    # We compare the average brightness of the middle column vs side columns
    
    vert_mid = gray_img[y : y + h, x + x_third : x + 2 * x_third]
    vert_left = gray_img[y : y + h, x : x + x_third]
    vert_right = gray_img[y : y + h, x + 2 * x_third : x + w]

    den_vert_mid = np.mean(vert_mid) / 255.0 if vert_mid.size > 0 else 0
    den_vert_left = np.mean(vert_left) / 255.0 if vert_left.size > 0 else 0
    den_vert_right = np.mean(vert_right) / 255.0 if vert_right.size > 0 else 0
    
    den_vert_sides = (den_vert_left + den_vert_right) / 2.0
    
    # Ratio
    gap_signal = den_vert_sides / (den_vert_mid + 1e-5)

    return {
        "Zone_TopMid_Density": d_top_mid,
        "Zone_TopSides_Density": d_top_sides,
        "X_Gap_Signal": gap_signal
    }

def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    # Threshold ONLY for Finding the Bounding Box & Shape
    # (We use 30 as a safe lower bound for your events which are ~17-47)
    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea)
    
    # Geometry
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area == 0 or perimeter == 0: return None

    x, y, w, h = cv2.boundingRect(cnt)
    
    # Shape Features (These are ratio-based, so they are robust)
    rect_area = w * h
    aspect_ratio = float(h) / w 
    extent = float(area) / rect_area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = 0 if hull_area == 0 else float(area) / hull_area
    circularity = (4 * math.pi * area) / (perimeter * perimeter)

    # Spatial Features -> NOW USING GRAYSCALE IMG
    spatial_feats = calculate_spatial_features(img, (x, y, w, h))

    return {
        "AspectRatio": aspect_ratio,
        "Extent": extent,
        "Solidity": solidity,
        "Circularity": circularity,
        **spatial_feats 
    }

def process_folders():
    # ... (Same as before) ...
    for folder in TARGET_FOLDERS:
        if not os.path.exists(folder): continue
        output_folder = folder + "_processed"
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
        print(f"Processing {len(files)} in {folder}...")
        
        for filename in files:
            img_path = os.path.join(folder, filename)
            json_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".json")
            data = extract_features(img_path)
            if data:
                data["Label"] = folder 
                data["SourceImage"] = filename
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=4)

if __name__ == "__main__":
    process_folders()