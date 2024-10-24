import streamlit as st
from core.file_handler import save_uploaded_files
from core.image_processor import remove_duplicates
from core.ocr_handler import process_images, flexible_name_match
from core.result_handler import download_results
import json
import os
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
import logging
import paddle

# 设置rich日志
console = Console(record=True)
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

log = logging.getLogger("rich")

CONFIG_FILE = "app_config.json"
UPLOAD_FOLDER = "upload"

# 检查CUDA是否可用
use_gpu = paddle.is_compiled_with_cuda()

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def log_step(step_name):
    console.print(Panel(f"[bold blue]{step_name}[/bold blue]", border_style="blue", expand=False))

def log_success(message):
    console.print(Panel(f"[bold green]{message}[/bold green]", border_style="green", expand=False))

def log_error(message):
    console.print(Panel(f"[bold red]{message}[/bold red]", border_style="red", expand=False))

def log_info(message):
    console.print(Panel(f"[cyan]{message}[/cyan]", border_style="cyan", expand=False))

def main():
    st.set_page_config(page_title="综测加分材料自动筛选系统", layout="wide")

    # 确保上传文件夹存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # 加载配置
    config = load_config()

    # 侧边栏配置
    with st.sidebar:
        st.title("配置参数")
        user_name = st.text_input("请输入你的名字", value=config.get('user_name', ''))
        similarity_threshold = st.slider("图片相似度阈值", 80, 100, config.get('similarity_threshold', 95))
        name_match_threshold = st.slider("名字匹配阈值", 60, 100, config.get('name_match_threshold', 80))
        ocr_lang = st.selectbox("OCR 语言", ["ch", "en"], index=0 if config.get('ocr_lang', 'ch') == 'ch' else 1)
        use_gpu_option = st.checkbox("使用 GPU 加速 (如果可用)", value=use_gpu)
        gpu_id = 0
        if use_gpu_option and use_gpu:
            available_gpus = list(range(paddle.device.cuda.device_count()))
            gpu_id = st.selectbox("选择 GPU", available_gpus, index=0)
        
        if st.button("保存配置"):
            new_config = {
                'user_name': user_name,
                'similarity_threshold': similarity_threshold,
                'name_match_threshold': name_match_threshold,
                'ocr_lang': ocr_lang
            }
            save_config(new_config)
            st.success("配置已保存")
        
        st.markdown("---")
        st.subheader("操作步骤")
        step = st.radio("选择步骤", ["1. 上传材料", "2. 处理材料", "3. 查看结果"])

    # 主界面
    st.title("综测加分材料筛选系统")
    st.markdown("---")

    # 存储当前配置到session state
    st.session_state['user_name'] = user_name
    st.session_state['similarity_threshold'] = similarity_threshold
    st.session_state['name_match_threshold'] = name_match_threshold
    st.session_state['ocr_lang'] = ocr_lang
    st.session_state['use_gpu'] = use_gpu_option and use_gpu
    st.session_state['gpu_id'] = gpu_id

    if step == "1. 上传材料":
        show_upload_page()
    elif step == "2. 处理材料":
        show_process_page()
    elif step == "3. 查看结果":
        show_results_page()

def show_upload_page():
    log_step("1. 上传材料")
    st.header("1. 上传材料")
    
    uploaded_files = st.file_uploader("上传图片", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    
    if uploaded_files:
        saved_files = save_uploaded_files(uploaded_files, UPLOAD_FOLDER)
        st.session_state['uploaded_files'] = saved_files
        log_success(f"成功上传并保存了 {len(saved_files)} 个文件到 {UPLOAD_FOLDER} 目录")
        st.success(f"成功上传 {len(saved_files)} 个文件！")
        st.info("请前往 '2. 处理材料' 步骤继续操作。")

def show_process_page():
    log_step("2. 处理材料")
    st.header("2. 处理材料")
    
    if 'uploaded_files' not in st.session_state or not st.session_state['uploaded_files']:
        st.warning("请先上传材料！")
        return

    show_ocr_results = st.checkbox("显示详细的 OCR 结果", value=False)

    if st.button("开始处理", key="process_button"):
        with st.spinner("正在处理中..."):
            progress_bar = st.progress(0)
            
            try:
                # 去重
                log_info("开始去重处理")
                progress_bar.progress(25)
                unique_images = remove_duplicates(
                    st.session_state['uploaded_files'], 
                    "",  # 不再使用文件夹路径
                    st.session_state['similarity_threshold']
                )
                
                # OCR 处理
                log_info("开始OCR处理")
                progress_bar.progress(50)
                ocr_result = process_images(
                    unique_images, 
                    st.session_state['user_name'],
                    st.session_state['ocr_lang'],
                    st.session_state['use_gpu'],
                    st.session_state['gpu_id'],
                    st.session_state['name_match_threshold']
                )
                processed_images = ocr_result.processed_images
                all_ocr_results = ocr_result.ocr_results
                
                # 显示详细的OCR和匹配结果
                if show_ocr_results:
                    st.subheader("OCR 和匹配结果")
                    for idx, (ocr_result, processed_image) in enumerate(zip(all_ocr_results, processed_images)):
                        img, is_matched, _ = processed_image
                        with st.expander(f"图片 {idx+1} {'(匹配)' if is_matched else '(未匹配)'}"):
                            st.image(img, caption=f"图片 {idx+1}", use_column_width=True)
                            st.text_area("OCR 结果", value="\n".join([item['text'] for item in ocr_result]), height=100)
                            
                            # 显示匹配结果
                            full_text = "\n".join([item['text'] for item in ocr_result])
                            is_matched, matched_name, all_matches = flexible_name_match(st.session_state['user_name'], full_text, st.session_state['name_match_threshold'])
                            st.write("匹配结果:")
                            for name, score in all_matches[:5]:  # 只显示前5个最佳匹配
                                st.write(f"- '{name}': {score:.2f}")
                            
                            if is_matched:
                                st.success(f"找到匹配: 用户名 '{st.session_state['user_name']}' 被检测为 '{matched_name}'")
                            else:
                                st.warning(f"未找到匹配: 用户名 '{st.session_state['user_name']}' 未被检测到")
                
                # 分离结果
                log_info("分离处理结果")
                progress_bar.progress(75)
                matched = []
                unmatched = []
                for item in processed_images:
                    if isinstance(item, tuple) and len(item) >= 2:
                        img, is_matched = item[:2]
                        if is_matched:
                            matched.append(img)
                        else:
                            unmatched.append(img)
                    else:
                        log_error(f"Unexpected item format in processed_images: {item}")

                st.session_state['matched'] = matched
                st.session_state['unmatched'] = unmatched
                
                progress_bar.progress(100)
            
                log_success("处理完成！")
                
                # 创建结果表格
                st.subheader("处理结果统计")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("匹配材料数量", len(matched))
                with col2:
                    st.metric("未匹配材料数量", len(unmatched))
                
                st.success("处理完成！")
                st.info("请前往 '3. 查看结果' 步骤查看处理结果。")
            except Exception as e:
                log_error(f"处理过程中出现错误: {str(e)}")
                st.error(f"处理过程中出现错误: {str(e)}")

def show_results_page():
    log_step("3. 查看结果")
    st.header("3. 查看结果")
    
    if 'matched' not in st.session_state or 'unmatched' not in st.session_state:
        st.warning("请先处理材料！")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("匹配的材料数量", len(st.session_state['matched']))
    with col2:
        st.metric("未匹配的材料数量", len(st.session_state['unmatched']))

    st.markdown("---")

    if st.button("下载结果", key="download_button"):
        try:
            download_results(st.session_state['matched'], st.session_state['unmatched'])
            log_success("结果下载成功！")
            st.success("结果下载成功！")
        except Exception as e:
            log_error(f"下载结果时出现错误: {str(e)}")
            st.error(f"下载结果时出现错误: {str(e)}")

    # 预览结果
    st.subheader("结果预览")
    preview_tab1, preview_tab2 = st.tabs(["匹配的材料", "未匹配的材料"])
    
    with preview_tab1:
        show_image_preview(st.session_state['matched'], "匹配")
    
    with preview_tab2:
        show_image_preview(st.session_state['unmatched'], "未匹配")

def show_image_preview(images, category):
    if not images:
        st.info(f"没有{category}的材料")
        return
    
    cols = st.columns(3)
    for idx, img in enumerate(images[:9]):  # 只显示前9张图片
        with cols[idx % 3]:
            if isinstance(img, (tuple, list)):
                img = img[0] if len(img) > 0 else None
            if img is not None:
                st.image(img, caption=f"{category}材料 {idx+1}", use_column_width=True)
            else:
                st.write(f"无法显示 {category}材料 {idx+1}")
    
    if len(images) > 9:
        st.info(f"还有 {len(images) - 9} 张{category}材料未显示")

if __name__ == "__main__":
    main()
