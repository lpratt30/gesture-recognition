import os
import re
import sys

# --- CONFIGURATION ---
# Target subdirectories
TARGET_DIRS = ["positive", "positive_processed"]
OFFSET = 1000
# ---------------------
def batch_rename_increment():
    base_path = os.getcwd()
    print(f"Running in: {base_path}")
    print(f"Adding {OFFSET} to all blob numbers in: {TARGET_DIRS}\n")

    for folder in TARGET_DIRS:
        dir_path = os.path.join(base_path, folder)
        
        if not os.path.exists(dir_path):
            print(f"Skipping '{folder}': Directory not found.")
            continue

        print(f"Processing '{folder}'...")
        
        # Get list of files
        files = os.listdir(dir_path)
        renamed_count = 0

        # We process files to ensure we don't accidentally process a file twice 
        # (though typically OS won't list the new name immediately in the same iterator, it's safer to build a list first)
        files_to_process = []
        for f in files:
            # Look for a number in the filename (e.g., blob_531.jpg -> 531)
            # This regex finds the LAST sequence of digits in the name
            matches = re.findall(r'(\d+)', f)
            if matches:
                # Use the last number found (standard for blob_X.ext)
                original_num_str = matches[-1]
                files_to_process.append((f, original_num_str))

        # Sort descending to avoid collisions if we were shifting up by 1 (not strictly necessary for +1000 but good practice)
        files_to_process.sort(key=lambda x: int(x[1]), reverse=True)

        for filename, num_str in files_to_process:
            old_num = int(num_str)
            new_num = old_num + OFFSET
            
            # Replace only the specific number instance to preserve formatting
            # This handles "blob_05.jpg" -> "blob_1005.jpg" correctly
            new_filename = filename.replace(num_str, str(new_num), 1)
            
            src = os.path.join(dir_path, filename)
            dst = os.path.join(dir_path, new_filename)

            if os.path.exists(dst):
                print(f"  [WARNING] Target exists, skipping: {filename} -> {new_filename}")
                continue
            
            try:
                os.rename(src, dst)
                print(f"  {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"  [ERROR] Could not rename {filename}: {e}")

        print(f"Finished '{folder}': {renamed_count} files renamed.\n")

if __name__ == "__main__":
    # Safety confirmation
    print("WARNING: This will rename files in your 'negative' and 'negative_processed' folders.")
    print("Example: blob_5.jpg -> blob_1005.jpg")
    confirm = input("Type 'yes' to proceed: ")
    
    if confirm.lower() == 'yes':
        batch_rename_increment()
    else:
        print("Operation cancelled.")