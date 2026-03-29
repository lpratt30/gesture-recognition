import os
import shutil
from project_paths import DATASET_V1_DIR

# --- CONFIGURATION ---
SOURCE_DIR = os.fspath(DATASET_V1_DIR)

# Destination folder names (will be created inside SOURCE_DIR)
NEG_IMG_FOLDER = "negative"
NEG_JSON_FOLDER = "negative_processed"
# ---------------------

def move_all_to_negative():
    # 1. Verify Source Exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory does not exist:\n{SOURCE_DIR}")
        return

    # 2. Setup Destination Paths
    dest_img_path = os.path.join(SOURCE_DIR, NEG_IMG_FOLDER)
    dest_json_path = os.path.join(SOURCE_DIR, NEG_JSON_FOLDER)

    if not os.path.exists(dest_img_path): os.makedirs(dest_img_path)
    if not os.path.exists(dest_json_path): os.makedirs(dest_json_path)

    print(f"Scanning: {SOURCE_DIR} ...")

    # 3. Get all files
    all_files = os.listdir(SOURCE_DIR)
    
    # Filter for images and JSONs
    # (Exclude directories to avoid trying to move the folders we just created)
    files_to_move = [f for f in all_files if os.path.isfile(os.path.join(SOURCE_DIR, f))]

    count_img = 0
    count_json = 0

    for filename in files_to_move:
        src = os.path.join(SOURCE_DIR, filename)
        lower_name = filename.lower()

        # Move Images
        if lower_name.endswith(('.png', '.jpg', '.jpeg')):
            dst = os.path.join(dest_img_path, filename)
            shutil.move(src, dst)
            count_img += 1
            print(f"Moved Image: {filename}")

        # Move JSONs
        elif lower_name.endswith('.json'):
            dst = os.path.join(dest_json_path, filename)
            shutil.move(src, dst)
            count_json += 1
            print(f"Moved JSON:  {filename}")

    print("-" * 30)
    print(f"COMPLETE.")
    print(f"Images moved to '{NEG_IMG_FOLDER}': {count_img}")
    print(f"JSONs moved to '{NEG_JSON_FOLDER}': {count_json}")

if __name__ == "__main__":
    move_all_to_negative()
