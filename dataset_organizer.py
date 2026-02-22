import os
import shutil

def transfer_jsons(source_folder):
    # --- CONFIGURATION ---
    # Set to True to test without making changes
    DRY_RUN = False
    # ---------------------

    if not os.path.exists(source_folder):
        print(f"Error: Source folder does not exist: {source_folder}")
        return

    print(f"Processing: {source_folder}")
    print("-" * 40)

    # ---------------------------------------------------------
    # TRANSFER JSONs TO PROCESSED FOLDER
    # ---------------------------------------------------------
    print("Moving JSON files...")

    # Determine sibling directory path
    # e.g., /.../dataset_v1/negative -> /.../dataset_v1/negative_processed
    norm_path = os.path.normpath(source_folder)
    parent_dir = os.path.dirname(norm_path)
    current_folder_name = os.path.basename(norm_path)
    dest_folder_name = f"{current_folder_name}_processed"
    dest_path = os.path.join(parent_dir, dest_folder_name)

    print(f"Destination: {dest_path}")

    # Create directory if it doesn't exist
    if not DRY_RUN:
        os.makedirs(dest_path, exist_ok=True)

    # Get list of files
    all_files = os.listdir(source_folder)
    jsons_to_move = [f for f in all_files if f.lower().endswith('.json')]

    if jsons_to_move:
        print(f"Found {len(jsons_to_move)} JSON files to move.")
        for fname in jsons_to_move:
            src = os.path.join(source_folder, fname)
            dst = os.path.join(dest_path, fname)
            
            if DRY_RUN:
                print(f"[DRY RUN] Would move {fname}")
            else:
                try:
                    shutil.move(src, dst)
                    # print(f"Moved: {fname}") 
                except Exception as e:
                    print(f"Error moving {fname}: {e}")
        if not DRY_RUN:
            print("Move complete.")
    else:
        print("No JSON files found to move.")

if __name__ == "__main__":
    # Change this path as needed
    target_dir = r"C:\Users\pratt\OneDrive\Desktop\gesture_data\dataset_v1\negative"
    transfer_jsons(target_dir)

    target_dir = r"C:\Users\pratt\OneDrive\Desktop\gesture_data\dataset_v1\positive"
    transfer_jsons(target_dir)