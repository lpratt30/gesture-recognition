import cv2
import os
import shutil
import sys

# --- CONFIGURATION ---
POS_FOLDER_NAME = "positive"
NEG_FOLDER_NAME = "negative"
POS_JSON_SUFFIX = "_processed"
NEG_JSON_SUFFIX = "_processed"
# ---------------------

def get_paths_and_mode():
    """
    Detects context and returns appropriate paths.
    """
    current_dir_name = os.path.basename(os.getcwd())

    if current_dir_name == NEG_FOLDER_NAME:
        print(">>> MODE DETECTED: SUBDIRECTORY (Re-sorting Negative)")
        mode = "SUBDIR"
        root = ".."
        # In subdir, JSONs are not with images; they are in the processed folder up a level
        json_source_dir = os.path.join(root, f"{NEG_FOLDER_NAME}{NEG_JSON_SUFFIX}")
    else:
        print(">>> MODE DETECTED: PARENT (Standard Sort)")
        mode = "PARENT"
        root = "."
        # In parent, JSONs are right here with the images
        json_source_dir = "."

    return mode, root, json_source_dir

def sort_images():
    # 1. Determine Context
    MODE, ROOT, JSON_SOURCE_DIR = get_paths_and_mode()

    # Define Destinations relative to the detected Root
    POS_IMG_DEST = os.path.join(ROOT, POS_FOLDER_NAME)
    NEG_IMG_DEST = os.path.join(ROOT, NEG_FOLDER_NAME)
    POS_JSON_DEST = os.path.join(ROOT, f"{POS_FOLDER_NAME}{POS_JSON_SUFFIX}")
    NEG_JSON_DEST = os.path.join(ROOT, f"{NEG_FOLDER_NAME}{NEG_JSON_SUFFIX}")

    # 2. Create destination folders if they don't exist
    for folder in [POS_IMG_DEST, NEG_IMG_DEST, POS_JSON_DEST, NEG_JSON_DEST]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 3. Get list of images in CURRENT folder (works for both modes)
    try:
        files = os.listdir(".")
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except FileNotFoundError:
        print("Error: Current directory not found.")
        return

    if not images:
        print("No images found in the current directory.")
        return

    # Print Controls based on Mode
    print(f"\n--- CONTROLS ---")
    if MODE == "PARENT":
        print(f" [Y] = Move to Positive")
        print(f" [N] = Move to Negative")
    else:
        print(f" [Y] = CORRECT to Positive (Move out of here)")
        print(f" [N] = KEEP Negative (No Change)")
    print(f" [ESC] = Quit")
    print(f"----------------\n")

    count = 0
    total = len(images)

    for filename in images:
        img_path = os.path.join(".", filename)
        
        # Locate corresponding JSON based on mode logic
        base_name = os.path.splitext(filename)[0]
        json_filename = base_name + ".json"
        json_full_path = os.path.join(JSON_SOURCE_DIR, json_filename)
        
        has_json = os.path.exists(json_full_path)

        # Safety check
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load {filename}, skipping...")
            continue

        # UI Setup
        display_img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_NEAREST)
        
        # Status Text
        status_text = f"{count + 1}/{total} [{MODE}]"
        color = (0, 255, 0) if MODE == "PARENT" else (0, 0, 255) # Green for Parent, Red for Subdir
        
        cv2.putText(display_img, status_text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.putText(display_img, filename, (10, 580), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if has_json:
            cv2.putText(display_img, "+JSON", (500, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Smart Sorter", display_img)
        
        # Wait for key
        move_successful = False
        while not move_successful:
            key = cv2.waitKey(0)
            
            # --- POSITIVE LOGIC (Y) ---
            if key == ord('y'):
                # In BOTH modes, Y moves files to the Positive folders
                try:
                    shutil.move(img_path, os.path.join(POS_IMG_DEST, filename))
                    if has_json:
                        shutil.move(json_full_path, os.path.join(POS_JSON_DEST, json_filename))
                        print(f"[{count+1}/{total}] POSITIVE: Image & JSON moved.")
                    else:
                        print(f"[{count+1}/{total}] POSITIVE: Image moved (No JSON).")
                except Exception as e:
                    print(f"Error moving positive: {e}")
                
                move_successful = True

            # --- NEGATIVE LOGIC (N) ---
            elif key == ord('n'):
                if MODE == "PARENT":
                    # Parent Mode: Move files to Negative folders
                    try:
                        shutil.move(img_path, os.path.join(NEG_IMG_DEST, filename))
                        if has_json:
                            shutil.move(json_full_path, os.path.join(NEG_JSON_DEST, json_filename))
                            print(f"[{count+1}/{total}] NEGATIVE: Image & JSON moved.")
                        else:
                            print(f"[{count+1}/{total}] NEGATIVE: Image moved (No JSON).")
                    except Exception as e:
                        print(f"Error moving negative: {e}")
                
                else: 
                    # Subdir Mode: Do nothing (Keep in Negative)
                    print(f"[{count+1}/{total}] KEPT NEGATIVE.")
                
                move_successful = True
                
            elif key == 27: # ESC
                print("Quitting...")
                cv2.destroyAllWindows()
                return

        count += 1

    print("\nBatch complete!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sort_images()