

##  图像去噪应用: 使用 Hugging Face模型与DnCNN进行高效去噪

这个项目为图像去噪提供了一个简单易用的解决方案，支持使用预训练的 Hugging Face 扩散模型以及 DnCNN 模型进行去噪处理。

**项目特点:**

* **多种去噪方式:** 您可以选择使用 Hugging Face 的预训练扩散模型或 DnCNN 模型进行去噪。
* **多种噪声类型:** 支持添加多种类型的噪声到图像，模拟真实的场景。
* **用户友好:** 使用 Gradio 创建的简洁用户界面，方便您快速体验不同去噪方法。

**快速开始:**

1. **创建环境:**

   ```bash
   conda create -n vnev python=3.10
   conda activate venv
   pip install -r requirements0418.txt
   ```

2. **运行项目:**

   ```bash
   python image_hgfc.py
   ```

3. **打开浏览器:**

   访问 `http://127.0.0.1:7860/` 使用 Gradio 界面进行图像上传和去噪尝试。

**技术细节:**

项目内部集成 Hugging Face 的扩散模型库和 DnCNN 模型库，分别实现两种图像去噪方法以及传统方法。  用户可以选择使用的模型类型，并看到不同方法带来的效果比较。
<img width="1280" alt="798a81fa2b21841756e25ea5c99c267" src="https://github.com/user-attachments/assets/53caa65e-2e6f-411f-a9b4-bd7ee02240bf" />
<img width="1280" alt="c385ed3e746c25309a1685f82849b54" src="https://github.com/user-attachments/assets/a6680e1c-0094-435c-a462-9af9facef0c5" />


**未来发展:**

* 支持更多类型的噪声添加以及去噪模型
* 添加更详细的文档及示例
* 支持图像编辑模式，例如对部分区域进行精准去噪

**贡献:**

开源项目的蓬勃发展依靠每个人的贡献。 鼓励您进行探索、改进和贡献！



