import os
from PIL import Image
import io

def save_uploaded_files(uploaded_files, upload_folder):
    saved_files = []
    for uploaded_file in uploaded_files:
        # 获取文件名和扩展名
        name, ext = os.path.splitext(uploaded_file.name)
        
        # 创建一个唯一的文件名，使用.png作为新的扩展名
        file_name = os.path.join(upload_folder, f"{name}.png")
        counter = 1
        while os.path.exists(file_name):
            file_name = os.path.join(upload_folder, f"{name}_{counter}.png")
            counter += 1

        # 打开上传的图片
        image = Image.open(io.BytesIO(uploaded_file.getbuffer()))
        
        # 如果图片模式不是RGB，转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 保存为PNG格式
        image.save(file_name, 'PNG')
        
        saved_files.append(file_name)
    
    return saved_files
