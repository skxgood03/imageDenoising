import os
import sys
import time
import random
import logging
import warnings
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import gradio as gr
import requests
from typing import Dict, List, Tuple, Optional, Union, Any

# 取消抑制所有警告
warnings.filterwarnings('default')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('denoising.log', encoding='utf-8')
    ]
)

# 创建必要的目录
os.makedirs('uploads', exist_ok=True)
os.makedirs('noisy_images', exist_ok=True)
os.makedirs('denoised_images', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)


def set_seed(seed: int = 42):
    """设置全局随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """选择计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用Apple Silicon GPU")
    else:
        device = torch.device('cpu')
        print("使用CPU")
        import platform
        import multiprocessing
        print(f"CPU平台: {platform.platform()}")
        print(f"CPU核心数: {multiprocessing.cpu_count()}")
        print(f"Python版本: {platform.python_version()}")
        print(f"PyTorch版本: {torch.__version__}")
    return device


# 定义DnCNN模型架构
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out


# 下载预训练权重
def download_dncnn_weights():
    import requests
    import os

    os.makedirs('checkpoints', exist_ok=True)
    url = 'https://github.com/cszn/KAIR/raw/master/model_zoo/dncnn_color_blind.pth'
    weights_path = 'checkpoints/dncnn_color_blind.pth'

    if not os.path.exists(weights_path):
        print(f"下载预训练权重从 {url}")
        try:
            r = requests.get(url)
            with open(weights_path, 'wb') as f:
                f.write(r.content)
            print(f"权重已保存到 {weights_path}")
        except Exception as e:
            print(f"下载预训练模型失败: {e}")
            return None
    else:
        print(f"使用现有权重: {weights_path}")

    return weights_path


# 初始化和加载DnCNN模型
def load_dncnn_model():
    model = DnCNN()
    weights_path = download_dncnn_weights()

    if weights_path is not None:
        try:
            weights = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(weights)
            print("成功加载预训练模型权重")
        except Exception as e:
            print(f"加载模型权重时出错: {e}")
            print("使用随机初始化权重")
    else:
        print("使用随机初始化权重")

    model.eval()

    # 将模型移至计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, device


# 噪声类型
noise_types = {
    "高斯噪声": "gaussian",
    "椒盐噪声": "salt_pepper",
    "泊松噪声": "poisson",
    "斑点噪声": "speckle"
}


def add_noise(image, noise_type, noise_param=0.1):
    """为图像添加不同类型的噪声"""
    # 确保图像是numpy数组
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    elif isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = np.array(image)

    # 确保图像为float32类型，范围在[0, 1]
    image_np = image_np.astype(np.float32) / 255.0

    if noise_type == "gaussian":
        # 高斯噪声
        noise = np.random.normal(0, noise_param, image_np.shape)
        noisy_image = image_np + noise
    elif noise_type == "salt_pepper":
        # 椒盐噪声
        noisy_image = image_np.copy()
        # 盐噪声 (白点)
        salt = np.random.random(image_np.shape) < noise_param / 2
        noisy_image[salt] = 1.0
        # 椒噪声 (黑点)
        pepper = np.random.random(image_np.shape) < noise_param / 2
        noisy_image[pepper] = 0.0
    elif noise_type == "poisson":
        # 泊松噪声
        noisy_image = np.random.poisson(image_np * 255.0 * noise_param) / (255.0 * noise_param)
    elif noise_type == "speckle":
        # 斑点噪声
        noise = np.random.normal(0, noise_param, image_np.shape)
        noisy_image = image_np + image_np * noise
    else:
        return image

    # 确保像素值在[0,1]范围内
    noisy_image = np.clip(noisy_image, 0.0, 1.0)

    # 转回[0,255]范围的uint8
    return (noisy_image * 255).astype(np.uint8)


# 图像预处理函数
def preprocess_image(image, device):
    """将图像预处理为模型输入格式"""
    # 如果图像是PIL图像，转换为numpy
    if isinstance(image, Image.Image):
        image = np.array(image)

    # 确保图像是RGB格式
    if len(image.shape) == 2:  # 灰度图像
        # 转换灰度为RGB
        image = np.stack([image, image, image], axis=2)
    elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA图像
        image = image[:, :, :3]

    # 标准化到[0,1]
    image = image.astype(np.float32) / 255.0

    # 转换为PyTorch的CHW格式
    image = np.transpose(image, (2, 0, 1))

    # 添加batch维度
    image = np.expand_dims(image, 0)

    return torch.tensor(image, dtype=torch.float32).to(device)


def postprocess_image(tensor):
    """将模型输出转换为可显示图像"""
    # 转换为numpy数组
    image = tensor.cpu().detach().numpy()[0]

    # 将CHW格式转回HWC格式
    image = np.transpose(image, (1, 2, 0))

    # 将[0,1]范围转回[0,255]的uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image


# 添加噪声到图像并保存
def add_noise_to_image(image, noise_type, noise_strength):
    """添加噪声到图像并保存"""
    if image is None:
        return None, "未上传图像"

    try:
        # 转换噪声强度为参数
        noise_param = float(noise_strength) / 50.0  # 将[0-10]映射到[0-0.2]

        # 添加噪声
        noise_type_code = noise_types[noise_type]
        noisy_image_np = add_noise(image, noise_type_code, noise_param)

        # 保存噪声图像
        timestamp = int(time.time())
        filename = f"noisy_images/noisy_{timestamp}.png"

        # 确保是RGB格式
        if len(noisy_image_np.shape) == 2:
            # 灰度转RGB
            noisy_image_pil = Image.fromarray(noisy_image_np, mode='L').convert('RGB')
            noisy_image_pil.save(filename)
        elif len(noisy_image_np.shape) == 3 and noisy_image_np.shape[2] == 4:
            # 去掉alpha通道
            noisy_image_pil = Image.fromarray(noisy_image_np[:, :, :3])
            noisy_image_pil.save(filename)
        else:
            Image.fromarray(noisy_image_np).save(filename)

        return noisy_image_np, filename
    except Exception as e:
        logging.error(f"添加噪声出错: {str(e)}")
        return None, f"处理出错: {str(e)}"


# 使用预训练模型对图像进行去噪处理
def denoise_image(image_path):
    """使用预训练的深度学习模型对图像进行去噪"""
    if image_path is None:
        return None, "未选择图像文件"

    try:
        # 加载DnCNN模型
        model, device = load_dncnn_model()

        # 读取图像
        file_path = image_path.name if hasattr(image_path, 'name') else image_path
        print(f"读取文件：{file_path}")
        image = Image.open(file_path)

        # 预处理
        input_tensor = preprocess_image(image, device)

        # 模型推理
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # 后处理
        denoised_image_np = postprocess_image(output_tensor)

        # 保存去噪后的图像
        timestamp = int(time.time())
        output_path = f"denoised_images/denoised_{timestamp}.png"
        Image.fromarray(denoised_image_np).save(output_path)

        return denoised_image_np, output_path
    except Exception as e:
        logging.error(f"去噪过程出错: {str(e)}")
        import traceback
        traceback.print_exc()

        # 如果神经网络失败，回退到传统方法
        try:
            print("神经网络去噪失败，回退到传统中值滤波方法")
            image = Image.open(file_path)
            image_np = np.array(image)

            # 使用中值滤波
            from scipy import ndimage
            denoised = np.zeros_like(image_np)
            for i in range(3):  # 对RGB三个通道分别处理
                denoised[:, :, i] = ndimage.median_filter(image_np[:, :, i], size=3)

            timestamp = int(time.time())
            output_path = f"denoised_images/denoised_fallback_{timestamp}.png"
            Image.fromarray(denoised).save(output_path)

            return denoised, output_path + " (使用中值滤波)"
        except Exception as fallback_error:
            return None, f"处理出错，无法使用备选方案: {str(e)} -> {str(fallback_error)}"


# Gradio界面
def create_ui():
    with gr.Blocks(title="预训练图像去噪应用") as app:
        gr.Markdown("""
        # 预训练模型图像去噪应用

        ### 说明：
        1. 本应用使用预训练的DnCNN神经网络模型进行图像去噪
        2. 自动下载预训练权重(大约4MB)，请确保网络连接
        3. 如果模型加载失败，将回退到传统的中值滤波方法
        """)

        with gr.Tab("步骤1: 上传并添加噪声"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 上传原始图像")
                    input_image = gr.Image(type="pil", label="原始图像")

                    with gr.Row():
                        noise_type = gr.Dropdown(
                            choices=list(noise_types.keys()),
                            label="选择噪声类型",
                            value="高斯噪声"
                        )
                        noise_strength = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=3,
                            label="噪声强度"
                        )

                    add_noise_btn = gr.Button("添加噪声")

                with gr.Column():
                    gr.Markdown("### 添加噪声后的图像")
                    noisy_image = gr.Image(label="噪声图像")
                    noisy_image_path = gr.Textbox(label="噪声图像保存路径", visible=False)
                    gr.Markdown("噪声图像会自动保存到noisy_images文件夹中")

        with gr.Tab("步骤2: 对噪声图像进行去噪"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 选择要去噪的图像")
                    file_input = gr.File(label="选择噪声图像文件")
                    denoise_btn = gr.Button("开始去噪")

                with gr.Column():
                    gr.Markdown("### 去噪后的图像")
                    denoised_image = gr.Image(label="去噪图像")
                    denoised_image_path = gr.Textbox(label="去噪图像保存路径", visible=False)
                    gr.Markdown("去噪图像会自动保存到denoised_images文件夹中")

        # 连接按钮事件
        add_noise_btn.click(
            fn=add_noise_to_image,
            inputs=[input_image, noise_type, noise_strength],
            outputs=[noisy_image, noisy_image_path]
        )

        denoise_btn.click(
            fn=denoise_image,
            inputs=[file_input],
            outputs=[denoised_image, denoised_image_path]
        )

    return app


if __name__ == "__main__":
    # 下载模型
    print("初始化预训练去噪模型...")
    try:
        model, device = load_dncnn_model()
        print("模型初始化完成")
    except Exception as e:
        print(f"模型初始化失败，将使用备选方法: {e}")

    # 启动UI
    app = create_ui()
    app.launch(share=False)