import os
import subprocess
import sys
import urllib.request
import tarfile
import time

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
    'rec_distill': 'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_distillation.yml',
    'cls': 'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/configs/cls/cls_mv3.yml'
}

def download_file(url, filename):
    """
    下载文件并显示进度条
    """
    def reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = max(time.time() - start_time, 0.000001)  # 防止除以零
        progress_size = int(count * block_size)
        try:
            speed = int(progress_size / (1024 * duration))
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write(f"\r...{percent}%, {speed} KB/s, {progress_size / (1024 * 1024):.1f} MB")
            sys.stdout.flush()
        except Exception as e:
            print(f"\nError in reporthook: {e}")

    try:
        urllib.request.urlretrieve(url, filename, reporthook)
        print()  # 打印一个换行，使下一行输出在新行上
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        raise

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

def has_nvidia_gpu():
    """
    检测系统是否有NVIDIA GPU
    """
    try:
        subprocess.check_output('nvidia-smi')
        return True
    except:
        return False

def install_paddlepaddle():
    """
    根据系统情况安装PaddlePaddle
    """
    if has_nvidia_gpu():
        print("检测到NVIDIA GPU，正在安装 PaddlePaddle GPU 版本（CUDA 12.3）...")
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "paddlepaddle-gpu==3.0.0b1", 
            "-i", 
            "https://www.paddlepaddle.org.cn/packages/stable/cu123/"
        ])
    else:
        print("未检测到NVIDIA GPU，正在安装 PaddlePaddle CPU 版本...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "paddlepaddle==3.0.0b1",
            "-i",
            "https://www.paddlepaddle.org.cn/packages/stable/cpu/"
        ])

def install_paddleocr():
    """
    安装PaddleOCR
    """
    print("正在安装 PaddleOCR...")
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
    
    print("正在安装 PaddlePaddle...")
    install_paddlepaddle()
    
    print("正在安装 PaddleOCR...")
    install_paddleocr()
    
    print("正在下载模型和配置文件...")
    download_models_and_configs()
    
    print("安装完成！")
