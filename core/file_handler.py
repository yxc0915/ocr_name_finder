import os
from PIL import Image
import io

def save_uploaded_files(uploaded_files, upload_folder):
    saved_files = []
    for uploaded_file in uploaded_files:
        # 创建一个唯一的文件名
        file_name = os.path.join(upload_folder, uploaded_file.name)
        counter = 1
        while os.path.exists(file_name):
            name, ext = os.path.splitext(uploaded_file.name)
            file_name = os.path.join(upload_folder, f"{name}_{counter}{ext}")
            counter += 1

        # 保存文件
        with open(file_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        saved_files.append(file_name)
    
    return saved_files
