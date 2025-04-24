import os
import sys
import time
import json
import random
import logging
import warnings
import torch
import numpy as np
from PIL import Image
import gradio as gr
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

# 导入Hugging Face扩散模型组件
from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline, DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionImg2ImgPipeline

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


# ===== 加载Hugging Face预训练扩散模型 =====
# 修改load_huggingface_diffusion_models函数

def load_huggingface_diffusion_models(device):
    """从Hugging Face加载预训练的去噪扩散模型"""
    try:
        # 从Hugging Face加载预训练的模型
        # 用于不同类型的图像
        models = {}
        model_info = {}

        # 1. 添加图像去噪模型 - 使用稳定扩散的img2img
        print("正在加载图像去噪模型...")
        try:
            from diffusers import StableDiffusionImg2ImgPipeline

            # 尝试加载稳定扩散img2img模型用于实际的去噪
            models["denoising"] = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                safety_checker=None,  # 禁用安全检查器以提高速度
                requires_safety_checker=False
            ).to(device)
            model_info["denoising"] = {
                "name": "图像去噪模型",
                "source": "stabilityai/stable-diffusion-2-1",
                "size": 512,
                "type": "img2img"
            }
            print("✓ 图像去噪模型加载成功")
        except Exception as e:
            print(f"× 图像去噪模型加载失败: {e}")

            # 尝试加载备用模型
            try:
                models["denoising"] = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(device)
                model_info["denoising"] = {
                    "name": "备选去噪模型",
                    "source": "runwayml/stable-diffusion-v1-5",
                    "size": 512,
                    "type": "img2img"
                }
                print("✓ 备选去噪模型加载成功")
            except Exception as e:
                print(f"× 备选去噪模型也加载失败: {e}")

        # 2. 生成模型（可选）- 如果需要纯生成功能
        print("正在加载生成模型...")
        try:
            models["generation"] = DiffusionPipeline.from_pretrained(
                "google/ddpm-ema-bedroom-256",
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
            ).to(device)
            model_info["generation"] = {
                "name": "图像生成模型",
                "source": "google/ddpm-ema-bedroom-256",
                "size": 256,
                "type": "ddpm"
            }
            print("✓ 图像生成模型加载成功")
        except Exception as e:
            print(f"× 图像生成模型加载失败: {e}")

        # 如果所有模型都加载失败，返回None
        if len(models) == 0:
            print("所有模型加载失败")
            return None, None

        print(f"成功加载 {len(models)} 个模型")
        return models, model_info

    except Exception as e:
        logging.error(f"加载Hugging Face模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


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


# 使用Hugging Face预训练模型进行图像去噪
def denoise_with_diffusion(image_path, models, model_info, model_type="denoising", steps=50, noise_level=0.5):
    """使用Hugging Face预训练扩散模型对图像进行去噪"""
    if image_path is None:
        return None, "未选择图像文件"

    if models is None or model_info is None:
        return None, "扩散模型未成功加载，请检查日志"

    if model_type not in models:
        available_models = ", ".join(models.keys())
        return None, f"所选模型类型'{model_type}'不可用。可用模型: {available_models}"

    pipe = models[model_type]
    info = model_info[model_type]

    try:
        # 读取图像
        device = pipe.device
        file_path = image_path.name if hasattr(image_path, 'name') else image_path
        print(f"读取文件：{file_path}")
        image = Image.open(file_path)

        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 调整图像大小为模型需要的尺寸
        model_size = info["size"]  # 通常是256、512或768
        image = image.resize((model_size, model_size), Image.LANCZOS)
        print(f"已将图像调整为模型尺寸: {model_size}x{model_size}")

        # 根据模型类型执行不同的处理流程
        if info["type"] == "img2img":
            # 对于图像到图像模型（如Stable Diffusion的img2img）
            print(f"使用图像到图像模型 {info['name']} 开始去噪过程")

            # 使用相应的噪声强度作为强度参数
            # 噪声强度0-1映射到img2img的强度参数
            # 强度越低越保留原始图像内容
            strength = min(0.8, max(0.2, noise_level))  # 限制在0.2-0.8范围

            try:
                generator = torch.manual_seed(42)  # 固定种子以保证可重复性

                # 对于StableDiffusionImg2ImgPipeline，我们需要提供提示词
                # 这里使用中性提示，主要让模型关注于图像修复而不是内容生成
                prompt = "a clear, detailed, high quality image"

                # 执行图像到图像转换
                outputs = pipe(
                    prompt=prompt,
                    image=image,
                    strength=strength,  # 噪声强度，值越大改变越大
                    guidance_scale=7.5,  # 提示词引导强度，可调
                    num_inference_steps=steps,
                    generator=generator
                )

                # 获取处理后的图像
                if hasattr(outputs, "images"):
                    denoised_image = outputs.images[0]  # 取第一张图片
                else:
                    denoised_image = outputs[0]

                # 转为numpy数组
                denoised_image_np = np.array(denoised_image)
                print("图像到图像处理成功")

            except Exception as img2img_error:
                print(f"图像到图像处理失败: {img2img_error}")
                raise img2img_error

        elif info["type"] == "ddpm":
            # 这是生成模型，不是真正的去噪模型
            # 可以保留原代码，但要告知用户这不是真正的去噪
            print("警告：您选择的是生成模型，而非去噪模型。输出图像可能与输入无关。")

            # 调整噪声步数
            num_inference_steps = steps

            # 使用生成模型直接生成图像
            try:
                generator = torch.manual_seed(42)

                # 使用模型直接生成图像
                outputs = pipe(
                    batch_size=1,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                )

                # 获取生成的图像
                if hasattr(outputs, "images"):
                    # 有些模型会返回带images属性的对象
                    generated_image = outputs.images[0]
                else:
                    # 直接返回图像的情况
                    generated_image = outputs[0]

                # 转为numpy数组
                denoised_image_np = np.array(generated_image)
                print("使用生成模型生成图像成功（注意：这不是对输入图像的去噪）")

            except Exception as gen_error:
                print(f"生成图像失败: {gen_error}")
                raise gen_error

        else:
            # 对于其他类型的模型
            return None, f"不支持的模型类型: {info['type']}"

        # 统一保存处理后的图像
        timestamp = int(time.time())
        output_path = f"denoised_images/{model_type}_processed_{timestamp}.png"
        Image.fromarray(denoised_image_np).save(output_path)

        return denoised_image_np, output_path

    except Exception as e:
        logging.error(f"扩散模型处理出错: {str(e)}")
        import traceback
        traceback.print_exc()

        # 如果扩散模型失败，回退到传统方法
        try:
            print("扩散模型处理失败，回退到传统方法")
            return fallback_denoising(image_path)
        except Exception as fallback_error:
            logging.error(f"传统方法也失败了: {fallback_error}")
            return None, f"所有处理方法都失败: {str(e)} -> {str(fallback_error)}"
# 回退到传统去噪方法
def fallback_denoising(image_path):
    """使用传统方法进行图像去噪（回退方案）"""
    from scipy.ndimage import gaussian_filter, median_filter

    try:
        file_path = image_path.name if hasattr(image_path, 'name') else image_path
        image = Image.open(file_path)
        image_np = np.array(image)

        # 使用中值滤波+高斯滤波的组合
        denoised = np.zeros_like(image_np)
        for i in range(min(3, image_np.shape[2])):  # 处理RGB或灰度
            # 先用中值滤波去除椒盐噪声
            channel_med = median_filter(image_np[:, :, i], size=3)
            # 再用高斯滤波去除高斯噪声
            denoised[:, :, i] = gaussian_filter(channel_med, sigma=0.7)

        timestamp = int(time.time())
        output_path = f"denoised_images/traditional_denoised_{timestamp}.png"
        Image.fromarray(denoised).save(output_path)

        return denoised, output_path + " (使用传统滤波)"
    except Exception as e:
        logging.error(f"传统去噪方法出错: {str(e)}")
        return None, f"处理出错: {str(e)}"


# Gradio界面
def create_ui(models, model_info):
    with gr.Blocks(title="Hugging Face预训练扩散模型去噪应用") as app:
        gr.Markdown("""
        # Hugging Face预训练扩散模型去噪应用

        ### 说明：
        1. 本应用使用Hugging Face平台上的预训练扩散模型进行图像去噪
        2. 在"步骤1"中上传原始图像，然后添加不同类型的噪声
        3. 在"步骤2"中选择噪声图像，并使用扩散模型进行去噪
        4. 可以针对不同类型的图像选择不同模型

        ### 注意：
        * 首次使用时需要下载预训练模型（约150MB/个）
        * 去噪过程可能需要一些时间，特别是在CPU上运行时
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

                    # 添加模型选择下拉框
                    available_models = []
                    if models:
                        for model_key in models.keys():
                            if model_key in model_info:
                                name = model_info[model_key]["name"]
                                available_models.append((name, model_key))

                    if not available_models:
                        available_models = [("传统方法", "traditional")]

                    model_choice = gr.Dropdown(
                        choices=[model[0] for model in available_models],
                        value=available_models[0][0] if available_models else None,
                        label="选择去噪模型"
                    )

                    with gr.Row():
                        denoising_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            step=10,
                            value=50,
                            label="去噪步数 (更多步数=更好效果但更慢)"
                        )
                        noise_level_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            step=0.1,
                            value=0.5,
                            label="噪声级别 (估计的噪声强度)"
                        )

                    denoise_btn = gr.Button("开始去噪 (使用选择的模型)")
                    fallback_btn = gr.Button("使用传统方法去噪 (更快)")

                with gr.Column():
                    gr.Markdown("### 去噪后的图像")
                    denoised_image = gr.Image(label="去噪图像")
                    denoised_image_path = gr.Textbox(label="去噪图像保存路径", visible=True)
                    processing_status = gr.Markdown("等待处理...")

        # 连接按钮事件 - 使用Gradio 5.x的方式
        add_noise_btn.click(
            fn=add_noise_to_image,
            inputs=[input_image, noise_type, noise_strength],
            outputs=[noisy_image, noisy_image_path]
        )

        # 为模型选择创建模型名称到键的映射
        model_name_to_key = {model[0]: model[1] for model in available_models}

        # 使用两步方法处理去噪操作
        # 步骤1: 更新状态
        denoise_btn.click(
            fn=lambda: "正在使用所选模型进行去噪，请稍候...",
            inputs=None,
            outputs=processing_status
        )

        # 步骤2: 执行去噪
        # 根据选择的模型名称查找对应的模型key
        def get_model_key(model_name):
            return model_name_to_key.get(model_name, "traditional")

        denoise_btn.click(
            fn=lambda file, model_name, steps, noise_level: (
                denoise_with_diffusion(
                    file,
                    models,
                    model_info,
                    get_model_key(model_name),
                    int(steps),
                    float(noise_level)
                ) if model_name_to_key.get(model_name) != "traditional"
                else fallback_denoising(file)
            ),
            inputs=[file_input, model_choice, denoising_steps, noise_level_slider],
            outputs=[denoised_image, denoised_image_path]
        ).then(
            fn=lambda path: "✅ 去噪完成！" if path else "❌ 去噪失败",
            inputs=[denoised_image_path],
            outputs=processing_status
        )

        # 传统方法去噪按钮
        fallback_btn.click(
            fn=lambda: "正在使用传统方法进行去噪，请稍候...",
            inputs=None,
            outputs=processing_status
        )

        fallback_btn.click(
            fn=fallback_denoising,
            inputs=[file_input],
            outputs=[denoised_image, denoised_image_path]
        ).then(
            fn=lambda output_path: "✅ 传统去噪方法完成！" if output_path else "❌ 去噪失败",
            inputs=[denoised_image_path],
            outputs=processing_status
        )

    return app


if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)

    # 获取计算设备
    device = get_device()

    # 加载多个扩散模型
    print("正在加载Hugging Face扩散模型，请稍候...")
    try:
        models, model_info = load_huggingface_diffusion_models(device)
        if models:
            print("扩散模型加载成功!")
            for key, info in model_info.items():
                print(f"- {info['name']}: {info['source']}")
        else:
            print("扩散模型加载失败，将仅使用传统方法")
    except Exception as e:
        logging.error(f"加载扩散模型失败: {e}")
        models, model_info = None, None
        print("扩散模型加载失败，将仅使用传统方法")

    # 启动UI
    print("启动用户界面...")
    app = create_ui(models, model_info)
    app.launch(share=False)