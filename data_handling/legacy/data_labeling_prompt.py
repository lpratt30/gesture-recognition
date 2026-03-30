import os
import shutil
import re
import sys

def get_id_from_filename(filename):
    """
    Extracts the numeric ID from a filename. 
    Assumes the blob number is the last sequence of digits before the extension.
    Example: 'image_0045.jpg' -> 45
    """
    # Find all number sequences
    matches = re.findall(r'(\d+)', filename)
    if matches:
        # We usually assume the ID is the last number (or the only number)
        return int(matches[-1])
    return None

def move_batch_by_id():
    # 1. Identify Context
    current_folder_name = os.path.basename(os.getcwd())
    root_dir = ".."

    # Define the map of logic
    if current_folder_name == "positive":
        target_folder_name = "negative"
    elif current_folder_name == "negative":
        target_folder_name = "positive"
    else:
        print("ERROR: You must run this script INSIDE the 'positive' or 'negative' folder.")
        return

    # 2. Define Paths
    src_img_dir = "."
    dest_img_dir = os.path.join(root_dir, target_folder_name)
    
    src_json_dir = os.path.join(root_dir, f"{current_folder_name}_processed")
    dest_json_dir = os.path.join(root_dir, f"{target_folder_name}_processed")

    if not os.path.exists(dest_img_dir): os.makedirs(dest_img_dir)
    if not os.path.exists(dest_json_dir): os.makedirs(dest_json_dir)

    # 3. Build a Map: { ID (int) : Filename (str) }
    print(f"--- MAPPING FILES ---")
    file_map = {}
    
    try:
        files = os.listdir(src_img_dir)
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except Exception as e:
        print(f"Error reading directory: {e}")
        return

    for fname in images:
        blob_id = get_id_from_filename(fname)
        if blob_id is not None:
            file_map[blob_id] = fname

    if not file_map:
        print("No files with extractable numbers found.")
        return

    print(f"Indexed {len(file_map)} files in '{current_folder_name}'.")
    print(f"Target Destination: '{target_folder_name}'")
    print(f"---------------------")

    # 4. Get User Input
    print("\nEnter the BLOB NUMBERS to move (separated by spaces).")
    print("Example: 152 40 99")
    user_input = input("Selection: ")

    try:
        target_ids = [int(x) for x in user_input.replace(',', ' ').split() if x.strip()]
    except ValueError:
        print("Invalid input. Please enter numbers only.")
        return

    # 5. Process Moves
    print("\nProcessing...")
    moved_count = 0
    not_found = []

    for blob_id in target_ids:
        if blob_id in file_map:
            filename = file_map[blob_id]
            
            # Paths
            img_src = os.path.join(src_img_dir, filename)
            img_dest = os.path.join(dest_img_dir, filename)
            
            base_name = os.path.splitext(filename)[0]
            json_filename = base_name + ".json"
            json_src = os.path.join(src_json_dir, json_filename)
            json_dest = os.path.join(dest_json_dir, json_filename)

            try:
                # Move Image
                shutil.move(img_src, img_dest)
                msg = f"[ID: {blob_id}] {filename} -> MOVED"

                # Move JSON
                if os.path.exists(json_src):
                    shutil.move(json_src, json_dest)
                    msg += " (+JSON)"
                else:
                    msg += " (No JSON)"
                
                print(msg)
                moved_count += 1
                
                # Remove from map so we don't try to move it again if user duplicated input
                del file_map[blob_id] 

            except Exception as e:
                print(f"[ID: {blob_id}] Error moving {filename}: {e}")
        else:
            not_found.append(blob_id)

    print(f"\nCompleted. Moved {moved_count} files.")
    if not_found:
        print(f"IDs not found in current folder: {not_found}")

if __name__ == "__main__":
    move_batch_by_id()