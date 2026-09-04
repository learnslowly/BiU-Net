import os
import shutil

def mirror_hla_files(src_root, dst_root):
    for dirpath, _, filenames in os.walk(src_root):
        # Filter files that contain "HLA" in their name
        hla_files = [f for f in filenames if "HLA" in f]
        
        if hla_files:
            # Determine the relative path and the corresponding destination path
            relative_path = os.path.relpath(dirpath, src_root)
            dst_path = os.path.join(dst_root, relative_path)

            # Ensure the destination directory exists
            os.makedirs(dst_path, exist_ok=True)

            # Copy each relevant file
            for file in hla_files:
                src_file = os.path.join(dirpath, file)
                dst_file = os.path.join(dst_path, file)
                shutil.copy2(src_file, dst_file)  # Preserve metadata

# Define source and destination directories
source_directory = "."  # Change this to your actual source folder
destination_directory = "../HLA"  # Change this to your desired destination

# Run the mirroring function
mirror_hla_files(source_directory, destination_directory)

print("Mirroring completed.")
