import os
import json
import shutil
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb  # <--- NEW IMPORT
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data_handling.project_paths import DATASET_V1_DIR, MODEL_AUDITS_DIR, XGB_MODEL_PATH

# --- CONFIGURATION ---
PROCESSED_ROOT = os.fspath(DATASET_V1_DIR)
RAW_IMAGE_ROOT = os.fspath(DATASET_V1_DIR)

FOLDERS = ["gesture_data/dataset_v1/positive_processed", "gesture_data/dataset_v1/negative_processed"]

THRESHOLD = 0.6 
MODEL_FILENAME = os.fspath(XGB_MODEL_PATH)

AUDIT_DIRS = {
    "TP": os.fspath(MODEL_AUDITS_DIR / "xgboost" / "true_positives"),
    "TN": os.fspath(MODEL_AUDITS_DIR / "xgboost" / "true_negatives"),
    "FP": os.fspath(MODEL_AUDITS_DIR / "xgboost" / "false_positives"),
    "FN": os.fspath(MODEL_AUDITS_DIR / "xgboost" / "false_negatives"),
}

EXPECTED_FEATURES = [
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

def setup_audit_directories():
    print("Setting up audit directories...")
    for key, path in AUDIT_DIRS.items():
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Warning: Could not delete {path}: {e}")
        if not os.path.exists(path):
            os.makedirs(path)

def load_dataset():
    data = []
    print(f"Scanning for data in {PROCESSED_ROOT}...")
    
    for folder in FOLDERS:
        processed_path = os.path.join(PROCESSED_ROOT, folder)
        
        if not os.path.exists(processed_path): 
            print(f"Skipping '{folder}' (Folder not found)")
            continue
            
        label = 1 if "positive" in folder else 0
        raw_folder_name = folder.replace("_processed", "")
        raw_image_path = os.path.join(RAW_IMAGE_ROOT, raw_folder_name)
        
        try:
            files = [f for f in os.listdir(processed_path) if f.endswith('.json')]
        except FileNotFoundError:
            continue

        print(f"  Found {len(files)} files in '{folder}'")
        
        for filename in files:
            json_full_path = os.path.join(processed_path, filename)
            
            with open(json_full_path, 'r') as f:
                try:
                    payload = json.load(f)
                    if "features" not in payload: continue

                    feature_values = payload["features"]
                    
                    if len(feature_values) != len(EXPECTED_FEATURES):
                        print(f"    Mismatch in {filename}: Found {len(feature_values)}, Expected {len(EXPECTED_FEATURES)}")
                        continue

                    clean = dict(zip(EXPECTED_FEATURES, feature_values))
                    source_img_name = filename.replace(".json", ".jpg")
                    full_img_path = os.path.join(raw_image_path, source_img_name)

                    clean['target'] = label
                    clean['source_path'] = full_img_path 
                    data.append(clean)
                        
                except Exception as e:
                    print(f"    Error reading {filename}: {e}")
                    
    return pd.DataFrame(data)

def audit_images(y_test, y_pred, image_paths):
    print("\n--- GENERATING VISUAL AUDIT ---")
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    image_paths = np.array(image_paths)
    
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for i in range(len(y_test)):
        actual = y_test[i]
        pred = y_pred[i]
        src_path = image_paths[i]
        
        if actual == 1 and pred == 1: cat = "TP"
        elif actual == 0 and pred == 0: cat = "TN"
        elif actual == 0 and pred == 1: cat = "FP"
        elif actual == 1 and pred == 0: cat = "FN"
            
        counts[cat] += 1
        
        if os.path.exists(src_path):
            filename = os.path.basename(src_path)
            dst_path = os.path.join(AUDIT_DIRS[cat], filename)
            try:
                shutil.copy(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {filename}: {e}")

    print(f"Audit Complete. Images copied: {counts}")

def train_and_evaluate():
    setup_audit_directories()

    # 1. Load Data
    df = load_dataset()
    if df.empty: 
        return print("ERROR: No valid data found. Check folder names and JSON content.")
    
    # 2. Separate Data
    X = df[EXPECTED_FEATURES]
    y = df['target']
    paths = df['source_path'] 

    print(f"\nTraining on {len(df)} samples.")

    # 3. Split
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, paths, test_size=0.1, random_state=42
    )

    # --- XGBOOST WEIGHT CALCULATION ---
    # XGBoost uses 'scale_pos_weight' instead of class_weight='balanced'
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    weight_ratio = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"Pos/Neg Ratio: {num_pos}/{num_neg} -> Scale Pos Weight: {weight_ratio:.2f}")

    # 4. Train (XGBoost)
    print("Training XGBoost Classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=5,            # Slightly shallower depth is often better for XGB
        learning_rate=0.1,      # Standard starting rate
        scale_pos_weight=weight_ratio, # Handles class imbalance
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X_train, y_train)

    # 5. Save
    print(f"Saving model to '{MODEL_FILENAME}'...")
    joblib.dump(clf, MODEL_FILENAME)

    # 6. Evaluate
    print(f"Loading model back for verification...")
    loaded_clf = joblib.load(MODEL_FILENAME)
    probas = loaded_clf.predict_proba(X_test)[:, 1]
    y_pred_custom = (probas > THRESHOLD).astype(int)

    # 7. Report
    print(f"\n--- RESULTS (Threshold > {THRESHOLD:.2f}) ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_custom))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_custom, target_names=['Background', 'Peace Sign']))
    
    print("\nFeature Importances:")
    importances = loaded_clf.feature_importances_
    for name, imp in sorted(zip(EXPECTED_FEATURES, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f}")

    # 8. Audit
    audit_images(y_test, y_pred_custom, paths_test)

if __name__ == "__main__":
    train_and_evaluate()
"""Training and model experiment scripts."""
