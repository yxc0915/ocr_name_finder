import os
import subprocess
import sys
import urllib.request
import tarfile
from tqdm import tqdm

# 定义项目目录和模型目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
CONFIGS_DIR = os.path.join(PROJECT_DIR, 'configs')

# 定义需要下载的模型文件和配置文件
MODELS = {
    'det_server': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar',
    'rec_server': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar',
    'cls': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
    'det': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar',
    'rec': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar'
}

CONFIGS = {
    'det_teacher': 'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml',
    'rec_hgnet': 'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml',
    'det_cml': 'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_cml.yml',
    'rec_distill': 'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_distill.yml',
    'cls': 'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/configs/cls/cls_mv3.yml'
}

def download_file(url, filename):
    """
    下载文件并显示进度条
    """
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename, reporthook=lambda b, bsize, tsize: t.update(bsize))

def extract_tar(filename, extract_dir):
    """
    解压tar文件
    """
    with tarfile.open(filename, 'r') as tar:
        tar.extractall(path=extract_dir)

def install_dependencies():
    """
    安装项目依赖
    """
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # 安装PaddlePaddle GPU版本（假设使用CUDA 12.3）
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paddlepaddle-gpu==3.0.0b1", "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu123/"])
    
    # 安装PaddleOCR
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paddleocr"])

def download_models_and_configs():
    """
    下载并解压模型文件，下载配置文件
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    for model_name, url in MODELS.items():
        tar_filename = os.path.join(MODELS_DIR, f'{model_name}.tar')
        print(f'正在下载{model_name}模型...')
        download_file(url, tar_filename)
        print(f'正在解压{model_name}模型...')
        extract_tar(tar_filename, MODELS_DIR)
        os.remove(tar_filename)

    for config_name, url in CONFIGS.items():
        config_filename = os.path.join(CONFIGS_DIR, f'{config_name}_config.yml')
        print(f'正在下载{config_name}配置文件...')
        download_file(url, config_filename)

if __name__ == '__main__':
    print("正在安装依赖项...")
    install_dependencies()
    
    print("正在下载模型和配置文件...")
    download_models_and_configs()
    
    print("安装完成！")
