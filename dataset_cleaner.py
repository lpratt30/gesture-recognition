import os
from project_paths import DATASET_V1_DIR

def clean_unpaired_files(directory):
    # SETTINGS
    # Set this to True to see what would be deleted without actually deleting
    # Set this to False to actually delete the files
    DRY_RUN = False 
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return

    print(f"Scanning: {directory}...")

    # Lists to store the base names (without extension)
    jpg_bases = set()
    json_bases = set()
    
    # helper to track full filenames for deletion
    files_map = {
        'jpg': {},
        'json': {}
    }

    # 1. Scan the directory and populate sets
    try:
        all_files = os.listdir(directory)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return

    for filename in all_files:
        full_path = os.path.join(directory, filename)
        
        # Skip directories
        if os.path.isdir(full_path):
            continue

        base, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext == '.jpg':
            jpg_bases.add(base)
            files_map['jpg'][base] = filename
        elif ext == '.json':
            json_bases.add(base)
            files_map['json'][base] = filename

    # 2. Identify orphans
    # JPGs that don't have a matching JSON
    orphaned_jpgs = jpg_bases - json_bases
    
    # JSONs that don't have a matching JPG
    orphaned_jsons = json_bases - jpg_bases

    files_to_delete = []

    for base in orphaned_jpgs:
        files_to_delete.append(files_map['jpg'][base])
        
    for base in orphaned_jsons:
        files_to_delete.append(files_map['json'][base])

    # 3. Delete files
    if not files_to_delete:
        print("Dataset is clean! All JPGs and JSONs have matches.")
    else:
        print(f"Found {len(files_to_delete)} unpaired files.")
        
        for filename in files_to_delete:
            full_path = os.path.join(directory, filename)
            
            if DRY_RUN:
                print(f"[DRY RUN] Would delete: {filename}")
            else:
                try:
                    os.remove(full_path)
                    print(f"Deleted: {filename}")
                except OSError as e:
                    print(f"Error deleting {filename}: {e}")

if __name__ == "__main__":
    clean_unpaired_files(os.fspath(DATASET_V1_DIR / "positive"))
