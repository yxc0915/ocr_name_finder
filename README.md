# OCR 图像处理工具

![Python Version](https://img.shields.io/badge/python-3.12.x-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

## 项目简介

这是一个基于 PaddleOCR 的图像处理工具，主要用于识别图像中的文字，并根据用户指定的名称进行匹配。该工具可以处理批量图像，识别文字，并标记出匹配的名称位置。

## 主要功能

- 批量处理图像文件
- 使用 PaddleOCR 进行文字识别
- 灵活的名称匹配算法
- 在图像上标记匹配的文字位置
- 支持 GPU 加速（可选）
- 高度优化的 OCR 参数

## 前提条件

本项目需要满足以下环境要求：

1. CUDA 12.3
   - CUDA（Compute Unified Device Architecture）是NVIDIA推出的并行计算平台和编程模型。
   - 请确保您的系统已正确安装CUDA 12.3版本。

2. cuDNN v9.0.0
   - cuDNN（CUDA Deep Neural Network library）是NVIDIA专门为深度学习开发的GPU加速库。
   - 本项目要求安装cuDNN v9.0.0版本。（或以上版本）

请在开始项目之前，确保您的系统满足上述要求。正确安装和配置这些组件对于项目的顺利运行至关重要。

注意：安装CUDA和cuDNN可能需要特定的硬件支持和操作系统版本。请参考NVIDIA官方文档以获取详细的安装指南和兼容性信息。


## 安装指南

1. 克隆此仓库：
   ```
   git clone https://github.com/yxc0915/ocr_name_finder.git
   cd ocr_name_finder
   ```

2. 运行安装程序以安装依赖以及必要的模型：
   ```
   python install.py
   ```

## 使用方法

1. 准备您要处理的图像文件。

2. 运行主程序：
   ```
   streamlit run app.py
   ```

3. 在 Web 界面中：
   - 上传图像文件
   - 输入要匹配的用户名
   - 设置其他参数（如识别语言、GPU 使用等）
   - 点击"开始处理"按钮

4. 查看处理结果，包括匹配的图像和未匹配的图像。

## 配置说明

- `det_limit_side_len`: 检测模型输入图像的最长边长度
- `det_limit_type`: 限制图像尺寸的方式（'max' 或 'min'）
- `rec_image_shape`: 识别模型输入图像的尺寸
- `rec_batch_num`: 识别模型批处理大小
- `use_angle_cls`: 是否使用方向分类器
- `name_match_threshold`: 名称匹配的阈值

更多配置项可以在 `app.py` 和 `ocr_handler.py` 中找到。

## 注意事项

- 确保您有足够的磁盘空间来存储处理后的图像，本项目需要约300M的磁盘空间。
- 使用 GPU 可以显著提高处理速度，但需要正确配置 CUDA 环境。
- 处理大量图像可能需要较长时间，请耐心等待。

## 贡献指南

欢迎提交 Issues 和 Pull Requests 来帮助改进这个项目。

## 许可证

本项目采用 Apache License 2.0 许可证。有关详细信息，请查看 [LICENSE](LICENSE) 文件。

Apache License 2.0 是一个宽松的、商业友好的开源软件许可证。它允许用户自由地使用、修改和分发本软件，无论是以源代码形式还是以编译形式，但要求在分发时保留原始版权声明和免责声明。

使用本软件时，请确保遵守 Apache License 2.0 的所有条款和条件。


## 致谢

本项目的开发得益于以下开源项目的支持，在此表示衷心的感谢：

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): 一个强大的、领先的OCR工具库，为本项目提供了核心的文字识别功能。

- [Streamlit](https://streamlit.io/): 一个优秀的Python库，用于快速创建数据应用程序，为本项目提供了直观的Web界面。

- [OpenCV](https://opencv.org/): 开源计算机视觉库，在图像处理方面发挥了重要作用。

- [NumPy](https://numpy.org/): 科学计算的基础库，为本项目提供了高效的数组操作。


特别感谢这些项目的开发者和维护者，他们的工作使得本项目成为可能。同时也感谢所有为开源社区做出贡献的人们。



## 联系方式

[QQ](http://wpa.qq.com/msgrd?v=3&uin=2692290472&site=qq&menu=yes)

