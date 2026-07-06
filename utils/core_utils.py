import os
import shutil
import hashlib

def copy_config_file():
    src_dir = 'src_infer/model'
    dest_dir = 'src_infer/config'
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_dir, file)
                shutil.copy2(src_file_path, dest_file_path)

def calculate_md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()