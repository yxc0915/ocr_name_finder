import os
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import numpy as np
from fuzzywuzzy import fuzz
from rich.console import Console
from rich.progress import Progress
import re
import traceback
import yaml
from collections import namedtuple

console = Console()

# 获取项目根目录
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
CONFIGS_DIR = os.path.join(PROJECT_DIR, 'configs')

# 模型和配置文件的路径
DET_SERVER_MODEL_DIR = os.path.join(MODELS_DIR, 'ch_PP-OCRv4_det_server_infer')
REC_SERVER_MODEL_DIR = os.path.join(MODELS_DIR, 'ch_PP-OCRv4_rec_server_infer')
DET_MOBILE_MODEL_DIR = os.path.join(MODELS_DIR, 'ch_PP-OCRv4_det_infer')
REC_MOBILE_MODEL_DIR = os.path.join(MODELS_DIR, 'ch_PP-OCRv4_rec_infer')
CLS_MODEL_DIR = os.path.join(MODELS_DIR, 'ch_ppocr_mobile_v2.0_cls_infer')

DET_SERVER_CONFIG_PATH = os.path.join(CONFIGS_DIR, 'det_teacher_config.yml')
REC_SERVER_CONFIG_PATH = os.path.join(CONFIGS_DIR, 'rec_hgnet_config.yml')
DET_MOBILE_CONFIG_PATH = os.path.join(CONFIGS_DIR, 'det_cml_config.yml')
REC_MOBILE_CONFIG_PATH = os.path.join(CONFIGS_DIR, 'rec_distill_config.yml')
CLS_CONFIG_PATH = os.path.join(CONFIGS_DIR, 'cls_config.yml')

# 定义一个命名元组来存储OCR处理结果
OCRResult = namedtuple('OCRResult', ['processed_images', 'ocr_results'])

def load_yaml(yaml_path):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        console.print(f"[red]加载YAML文件 {yaml_path} 时出错: {str(e)}[/red]")
        return None

def flexible_name_match(user_name, ocr_text, threshold=70):
    user_name = user_name.lower()
    ocr_text = ocr_text.lower()
    
    user_name_chars = list(user_name)
    potential_names = re.findall(r'[\u4e00-\u9fa5]{2,4}', ocr_text)
    
    best_match_ratio = 0
    best_match = ""
    all_matches = []
    
    for name in potential_names:
        ratio = fuzz.ratio(user_name, name)
        char_match = sum(1 for char in user_name_chars if char in name) / len(user_name_chars)
        containment = 1 if user_name in name or name in user_name else 0
        combined_score = (ratio * 0.5 + char_match * 30 + containment * 20)
        
        all_matches.append((name, combined_score))
        
        if combined_score > best_match_ratio:
            best_match_ratio = combined_score
            best_match = name
    
    all_matches.sort(key=lambda x: x[1], reverse=True)
    
    return best_match_ratio >= threshold, best_match, all_matches

def preprocess_image(img, det_limit_side_len=960, det_limit_type='max'):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    orig_w, orig_h = img.size
    if det_limit_type == 'max':
        ratio = det_limit_side_len / max(orig_h, orig_w)
        if ratio < 1:
            img = img.resize((int(orig_w * ratio), int(orig_h * ratio)), Image.LANCZOS)
    elif det_limit_type == 'min':
        ratio = det_limit_side_len / min(orig_h, orig_w)
        if ratio > 1:
            img = img.resize((int(orig_w * ratio), int(orig_h * ratio)), Image.LANCZOS)
    
    return img

def draw_box_around_text(image, matched_positions, matched_name):
    draw = ImageDraw.Draw(image)
    for item in matched_positions:
        box = item['position']
        if isinstance(box, list) and len(box) == 4 and all(isinstance(point, list) and len(point) == 2 for point in box):
            # 计算边框的宽度和高度
            width = max(point[0] for point in box) - min(point[0] for point in box)
            height = max(point[1] for point in box) - min(point[1] for point in box)
            
            # 计算边框的中心点
            center_x = sum(point[0] for point in box) / 4
            center_y = sum(point[1] for point in box) / 4
            
            # 根据文本大小调整边框大小
            scale_factor = max(width, height) / 100  # 可以调整这个值来改变缩放比例
            padding_x = max(10, int(20 * scale_factor))  # 最小10像素，最大根据文本大小调整
            padding_y = max(5, int(10 * scale_factor))   # 最小5像素，最大根据文本大小调整
            
            # 计算新的边框坐标
            new_box = [
                [center_x - width/2 - padding_x, center_y - height/2 - padding_y],
                [center_x + width/2 + padding_x, center_y - height/2 - padding_y],
                [center_x + width/2 + padding_x, center_y + height/2 + padding_y],
                [center_x - width/2 - padding_x, center_y + height/2 + padding_y]
            ]
            
            # 绘制新的边框
            draw.polygon([coord for point in new_box for coord in point], outline=(255, 0, 0), width=max(2, int(scale_factor * 2)))
        elif isinstance(box, list) and len(box) == 8:
            # 如果是8个坐标点的情况，也可以类似处理
            width = max(box[0::2]) - min(box[0::2])
            height = max(box[1::2]) - min(box[1::2])
            center_x = sum(box[0::2]) / 4
            center_y = sum(box[1::2]) / 4
            
            scale_factor = max(width, height) / 100
            padding_x = max(10, int(20 * scale_factor))
            padding_y = max(5, int(10 * scale_factor))
            
            new_box = [
                center_x - width/2 - padding_x, center_y - height/2 - padding_y,
                center_x + width/2 + padding_x, center_y - height/2 - padding_y,
                center_x + width/2 + padding_x, center_y + height/2 + padding_y,
                center_x - width/2 - padding_x, center_y + height/2 + padding_y
            ]
            
            draw.polygon(new_box, outline=(255, 0, 0), width=max(2, int(scale_factor * 2)))
        else:
            console.print(f"[yellow]警告：无法识别的边框格式：{box}[/yellow]")
    return image

def process_images(images, user_name, ocr_lang, use_gpu, gpu_id, name_match_threshold,
                   det_limit_side_len=960, det_limit_type='max',
                   rec_image_shape="3,48,320", rec_batch_num=6,
                   use_angle_cls=True,
                   det_db_thresh=0.3, det_db_box_thresh=0.6, det_db_unclip_ratio=1.5,
                   save_crop_res=False, crop_res_save_dir="./output"):
    
    console.print(f"[cyan]当前工作目录: {os.getcwd()}[/cyan]")
    
    # 根据是否使用GPU选择合适的模型和配置文件
    if use_gpu:
        det_model_dir = DET_SERVER_MODEL_DIR
        rec_model_dir = REC_SERVER_MODEL_DIR
        det_config_path = DET_SERVER_CONFIG_PATH
        rec_config_path = REC_SERVER_CONFIG_PATH
    else:
        det_model_dir = DET_MOBILE_MODEL_DIR
        rec_model_dir = REC_MOBILE_MODEL_DIR
        det_config_path = DET_MOBILE_CONFIG_PATH
        rec_config_path = REC_MOBILE_CONFIG_PATH
    
    console.print(f"[cyan]模型路径:[/cyan]")
    console.print(f"[cyan]  检测模型: {det_model_dir}[/cyan]")
    console.print(f"[cyan]  识别模型: {rec_model_dir}[/cyan]")
    console.print(f"[cyan]  方向分类模型: {CLS_MODEL_DIR}[/cyan]")
    
    console.print(f"[cyan]配置文件路径:[/cyan]")
    console.print(f"[cyan]  检测配置: {det_config_path}[/cyan]")
    console.print(f"[cyan]  识别配置: {rec_config_path}[/cyan]")
    console.print(f"[cyan]  方向分类配置: {CLS_CONFIG_PATH}[/cyan]")
    
    console.print(f"[cyan]模型参数:[/cyan]")
    console.print(f"[cyan]  检测限制边长: {det_limit_side_len}[/cyan]")
    console.print(f"[cyan]  检测限制类型: {det_limit_type}[/cyan]")
    console.print(f"[cyan]  识别图像形状: {rec_image_shape}[/cyan]")
    console.print(f"[cyan]  识别批次大小: {rec_batch_num}[/cyan]")

    det_config = load_yaml(det_config_path)
    rec_config = load_yaml(rec_config_path)
    cls_config = load_yaml(CLS_CONFIG_PATH)
    if det_config is None or rec_config is None or cls_config is None:
        return OCRResult([], [])

    try:
        ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=ocr_lang,
            use_gpu=use_gpu,
            gpu_mem=500,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=CLS_MODEL_DIR,
            det_limit_side_len=det_limit_side_len,
            det_limit_type=det_limit_type,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
            rec_image_shape=rec_image_shape,
            rec_batch_num=rec_batch_num,
            max_text_length=25,
            use_space_char=True,
            show_log=True
        )
    except Exception as e:
        console.print(f"[red]错误：初始化PaddleOCR时出错。{str(e)}[/red]")
        console.print(f"[red]错误详情：\n{traceback.format_exc()}[/red]")
        return OCRResult([], [])

    processed_images = []
    all_ocr_results = []

    with Progress() as progress:
        task = progress.add_task("[cyan]OCR处理中...[/cyan]", total=len(images))

        for idx, img in enumerate(images):
            try:
                if img is None:
                    raise ValueError("图像为空或无效")

                if isinstance(img, str):
                    if not os.path.exists(img):
                        raise FileNotFoundError(f"找不到图像文件：{img}")
                    img = Image.open(img)

                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.uint8(img))

                img = preprocess_image(img, det_limit_side_len, det_limit_type)
                img_array = np.array(img)

                result = ocr.ocr(img_array, cls=use_angle_cls)
                
                console.print(f"[cyan]OCR 原始结果:[/cyan]")
                console.print(result)

                text_with_positions = []
                if result is not None:
                    for item in result:
                        if isinstance(item, list):
                            for line in item:
                                if isinstance(line, list) and len(line) >= 2:
                                    position = line[0]
                                    if isinstance(line[1], tuple) and len(line[1]) >= 2:
                                        text, confidence = line[1][:2]
                                    elif isinstance(line[1], str):
                                        text = line[1]
                                        confidence = line[2] if len(line) > 2 else 1.0
                                    else:
                                        continue
                                    text_with_positions.append({
                                        'text': text,
                                        'position': position,
                                        'confidence': confidence
                                    })
                        elif isinstance(item, dict):
                            text_with_positions.append(item)
                
                full_text = "\n".join([item['text'] for item in text_with_positions])
                
                console.print(f"[cyan]处理后的文本和位置:[/cyan]")
                for item in text_with_positions:
                    console.print(f"  文本: {item['text']}")
                    console.print(f"  位置: {item['position']}")
                    console.print(f"  置信度: {item['confidence']}")

                all_ocr_results.append(text_with_positions)
                console.print(f"[blue]图片 {idx+1} OCR 结果:[/blue]")
                console.print(f"  提取的文本: {full_text}")

                name_found, matched_name, all_matches = flexible_name_match(user_name, full_text, threshold=name_match_threshold)
                
                if name_found:
                    console.print(f"[green]图片 {idx+1} 找到匹配: 用户名 '{user_name}' 被检测为 '{matched_name}'[/green]")
                    console.print(f"[cyan]匹配到的名字位置:[/cyan]")
                    matched_positions = []
                    for item in text_with_positions:
                        if matched_name.lower() in item['text'].lower() or item['text'].lower() in matched_name.lower():
                            console.print(f"  文本: {item['text']}, 位置: {item['position']}")
                            matched_positions.append(item)
                    marked_img = draw_box_around_text(img, matched_positions, matched_name)
                    processed_images.append((marked_img, True, matched_positions))
                else:
                    console.print(f"[yellow]图片 {idx+1} 未找到匹配: 用户名 '{user_name}' 未被检测到[/yellow]")
                    processed_images.append((img, False, []))

                if save_crop_res:
                    for i, item in enumerate(text_with_positions):
                        crop_img = img.crop(item['position'])
                        crop_img.save(os.path.join(crop_res_save_dir, f"crop_{idx}_{i}.jpg"))

            except Exception as e:
                console.print(f"[red]图片 {idx+1} OCR处理时出错: {str(e)}[/red]")
                console.print(f"[red]错误详情：\n{traceback.format_exc()}[/red]")
                processed_images.append((img, False, []))

            progress.update(task, advance=1)

    console.print(f"[bold green]OCR处理完成。处理图片数: {len(processed_images)}[/bold green]")
    
    console.print(f"[cyan]返回值类型:[/cyan]")
    console.print(f"processed_images 类型: {type(processed_images)}")
    console.print(f"processed_images 长度: {len(processed_images)}")
    console.print(f"all_ocr_results 类型: {type(all_ocr_results)}")
    console.print(f"all_ocr_results 长度: {len(all_ocr_results)}")
    
    return OCRResult(processed_images, all_ocr_results)
