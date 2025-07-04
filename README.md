<p align="center">
  <img src="assets/brand.png" width="65%">
</p>

<p align="center">
  <a href="https://vectorspacelab.github.io/OmniGen2"><img src="https://img.shields.io/badge/Project%20Page-OmniGen2-yellow" alt="project page"></a>
  <a href="https://arxiv.org/abs/2506.18871"><img src="https://img.shields.io/badge/arXiv%20paper-2506.18871-b31b1b.svg" alt="arxiv"></a>
  <a href="https://github.com/VectorSpaceLab/OmniGen2?tab=readme-ov-file#-gradio-demo"><img src="https://img.shields.io/badge/Online%20Demo-🤗-blue" alt="demo"></a>
  <a href="https://huggingface.co/spaces/OmniGen2/OmniGen2"><img src="https://img.shields.io/badge/HF%20Spaces-🤗-lightblue" alt="demo"></a>
  <a href="https://huggingface.co/OmniGen2/OmniGen2"><img src="https://img.shields.io/badge/Model-🤗-yellow" alt="model"></a>
  <a href="https://huggingface.co/datasets/OmniGen2/OmniContext"><img src="https://img.shields.io/badge/Benchmark-🤗-yellow" alt="model"></a>
  <a href=""><img src="https://img.shields.io/badge/Dataset-🤗-yellow" alt="model"></a>
</p>

<h4 align="center">
    <p>
        <a href=#-news>News</a> |
        <a href=#-quick-start>Quick Start</a> |
        <a href=#-usage-tips>Usage Tips</a> |
        <a href=#-limitations-and-suggestions>Limitations</a> |
        <a href=#-gradio-demo>Online Demos</a> |
        <a href=#%EF%B8%8F-citing-us>Citation</a>
    <p>
</h4>

## 🔥 News
- **2025-07-03**: OmniGen2 now supports [TeaCache](https://github.com/ali-vilab/TeaCache) and [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) for faster inference, see (Usage Tips)[https://github.com/VectorSpaceLab/OmniGen2?tab=readme-ov-file#-usage-tips] for details. Thanks @legitnull for great [TeaCache-PR](https://github.com/VectorSpaceLab/OmniGen2/pull/52) and [TaylorSeer-PR](https://github.com/VectorSpaceLab/OmniGen2/pull/76).
- **2025-07-01**: OmniGen2 is supported by [ComfyUI official](https://comfyanonymous.github.io/ComfyUI_examples/omnigen), thanks !!
- **2025-06-30**: Training code is available, see [fine-tuning](docs/FINETUNE.md) for details.
- **2025-06-28**: We release [OmniContext](https://huggingface.co/datasets/OmniGen2/OmniContext) benchmark. The evaluation codes are in [omnicontext](https://github.com/VectorSpaceLab/OmniGen2/tree/main/omnicontext).
- **2025-06-24**: [Technical Report](https://arxiv.org/abs/2506.18871) is available.
- **2025-06-23**: We’ve updated our code and HF model—OmniGen2 now runs *without* `flash-attn`. Users can still install it for optimal performance.
- **2025-06-20**: Updated [resource requirements](#-resources-requirement), adding CPU offload support for devices with limited VRAM.
- **2025-06-16**: [Gradio](https://github.com/VectorSpaceLab/OmniGen2?tab=readme-ov-file#-gradio-demo) and [Jupyter](https://github.com/VectorSpaceLab/OmniGen2/blob/main/example.ipynb) is available. Online Gradio Demo: [Demo1](https://9c4426d27c3b9ecbed.gradio.live); [Chat-Demo1](https://0351497834a4d7226c.gradio.live); see more demo links in [gradio section](https://github.com/VectorSpaceLab/OmniGen2?tab=readme-ov-file#-gradio-demo)
- **2025-06-16**: We release **OmniGen2**, a multimodal generation model, model weights can be accessed in [huggingface](https://huggingface.co/OmniGen2/OmniGen2) and [modelscope](https://www.modelscope.cn/models/OmniGen2/OmniGen2).


## Introduction
**OmniGen2** is a powerful and efficient generative model. Unlike OmniGen v1, OmniGen2 features two distinct decoding pathways for text and image modalities, utilizing unshared parameters and a decoupled image tokenizer. OmniGen2 has competitive performance across four primary capabilities:

- **Visual Understanding**: Inherits the robust ability to interpret and analyze image content from its Qwen-VL-2.5 foundation.
- **Text-to-Image Generation**: Creates high-fidelity and aesthetically pleasing images from textual prompts.
- **Instruction-guided Image Editing**: Executes complex, instruction-based image modifications with high precision, achieving state-of-the-art performance among open-source models.
- **In-context Generation**: A versatile capability to process and flexibly combine diverse inputs—including humans, reference objects, and scenes—to produce novel and coherent visual outputs.

**We will release the training code and dataset. Stay tuned!**

Some good cases of OmniGen2:
<p align="center">
  <img src="assets/teaser.jpg" width="95%">
  <br>
  <em>Demonstrations.</em>
</p>

<p align="center">
  <img src="assets/examples_edit.png" width="95%">
  <br>
  <em> Good demonstrations of OmniGen2's image editing capabilities.</em>
</p>

<p align="center">
  <img src="assets/examples_subject.png" width="95%">
  <br>
  <em> Good demonstrations of OmniGen2's in-context generation capabilities.</em>
</p>



## 📌 TODO
- [x] Technical report.
- [x] Support CPU offload and improve inference efficiency.
- [x] In-context generation benchmark: **OmniContext**.
- [ ] Integration of diffusers.
- [ ] Training datasets.
- [ ] Training data construction pipeline.
- [ ] ComfyUI Demo (**commuity support will be greatly appreciated!**).

## 🚀 Quick Start

### 🛠️ Environment Setup

#### ✅ Recommended Setup

```bash
# 1. Clone the repo
git clone git@github.com:VectorSpaceLab/OmniGen2.git
cd OmniGen2

# 2. (Optional) Create a clean Python environment
conda create -n omnigen2 python=3.11
conda activate omnigen2

# 3. Install dependencies
# 3.1 Install PyTorch (choose correct CUDA version)
pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124

# 3.2 Install other required packages
pip install -r requirements.txt

# Note: Version 2.7.4.post1 is specified for compatibility with CUDA 12.4.
# Feel free to use a newer version if you use CUDA 12.6 or they fixed this compatibility issue.
# OmniGen2 runs even without flash-attn, though we recommend install it for best performance.
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

#### 🌏 For users in Mainland China

```bash
# Install PyTorch from a domestic mirror
pip install torch==2.6.0 torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu124

# Install other dependencies from Tsinghua mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Note: Version 2.7.4.post1 is specified for compatibility with CUDA 12.4.
# Feel free to use a newer version if you use CUDA 12.6 or they fixed this compatibility issue.
# OmniGen2 runs even without flash-attn, though we recommend install it for best performance.
pip install flash-attn==2.7.4.post1 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 🧪 Run Examples

```bash
# Visual Understanding
bash example_understanding.sh

# Text-to-image generation
bash example_t2i.sh

# Instruction-guided image editing
bash example_edit.sh

# In-context generation
bash example_in_context_generation.sh
```

---

### 🌐 Gradio Demo

* **Online Demo**: [HF Spaces](https://huggingface.co/spaces/OmniGen2/OmniGen2). Beyond Hugging Face Spaces, we are *temporarily* allocating additional GPU resources to ensure smooth access to the online demos. If you notice a long queue for a particular link, please try other links:

    [Demo1](https://9c4426d27c3b9ecbed.gradio.live), [Demo2](https://06574c5e62d815f799.gradio.live), [Demo3](https://e0a82fd380d2ff17ac.gradio.live), [Demo4](https://d9c4410ee48ce35051.gradio.live)

    [Chat-Demo1](https://0351497834a4d7226c.gradio.live), [Chat-Demo2](https://032160099388d1d10c.gradio.live), [Chat-Demo3](https://cf9f2797e92cfa2767.gradio.live), [Chat-Demo4](https://b87b82fd14215affc2.gradio.live)

<!-- [Available on Hugging Face Spaces 🚀](https://huggingface.co/spaces/Shitao/OmniGen2) -->

* **Run Locally**:
    ```bash
    # for only generating image
    pip install gradio
    python app.py
    # Optional: Share demo with public link (You need to be able to access huggingface)
    python app.py --share

    # for generating image or text
    pip install gradio
    python app_chat.py
    ```

## 💡 Usage Tips
To achieve optimal results with OmniGen2, you can adjust the following key hyperparameters based on your specific use case.
- `text_guidance_scale`: Controls how strictly the output adheres to the text prompt (Classifier-Free Guidance).
- `image_guidance_scale`: This controls how much the final image should resemble the input reference image.
    - **The Trade-off**: A higher value makes the output more faithful to the reference image's structure and style, but it might ignore parts of your text prompt. A lower value (~1.5) gives the text prompt more influence.
    - **Tip**: For image editing task, we recommend to set it between 1.2 and 2.0; for in-context generateion task, a higher image_guidance_scale will maintian more details in input images, and we recommend to set it between 2.5 and 3.0.
- `max_pixels`: Automatically resizes images when their total pixel count (width × height) exceeds this limit, while maintaining its aspect ratio. This helps manage performance and memory usage.
  - **Tip**: Default value is 1024*1024. You can reduce this value if you encounter memory issues.
- `max_input_image_side_length`: Maximum side length for input images.
- `negative_prompt`: Tell the model what you don't want to see in the image.
    - **Example**: blurry, low quality, text, watermark
    - **Tip**: For the best results, try experimenting with different negative prompts. If you're not sure, just use the default negative prompt.
- `enable_model_cpu_offload`: **Reduces VRAM usage by nearly 50% with a negligible impact on speed**.
  - This is achieved by offloading the model weights to CPU RAM when they are not in use.
  - See: [Model Offloading](https://huggingface.co/docs/diffusers/optimization/memory#model-offloading)
- `enable_sequential_cpu_offload`: Minimizes VRAM usage to less than 3GB, but at the cost of significantly slower performance.
  - This works by offloading the model in submodules and loading them onto the GPU sequentially as needed.
  - See: [CPU Offloading](https://huggingface.co/docs/diffusers/optimization/memory#cpu-offloading)
- `cfg_range_start`, `cfg_range_end`: Define the timestep range where CFG is applied. Per this [paper](https://arxiv.org/abs/2404.07724), reducing `cfg_range_end` can significantly decrease inference time with a negligible impact on quality.
- `scheduler`: Choose between `[euler, dpmsolver++]`. Default is `euler`. For potentially better performance with fewer steps, try `dpmsolver++`.
- `num_inference_step`: Number of discretization steps for the ODE solver. Default is `50`.
- `enable_teacache`: Whether or not enable [teacache](https://github.com/ali-vilab/TeaCache) for faster inference.
- `teacache_rel_l1_thresh`: The threshold for accumulated L1 distance for the timestep embedding-modulated noisy input. It serves as an indicator of whether to cache the model output. You can modify the `teacache_rel_l1_thresh` parameter to achieve your desired trade-off between latency and visual quality. The default value of 0.05 provides approximately a **30% speedup** compared to the baseline. Increasing this value can further reduce latency, but may result in some loss of detail.
- `enable_taylorseer`: Whether or not enable [taylorseer](https://github.com/Shenyi-Z/TaylorSeer) for faster inference. When enabled, inference speed can improve by up to **2X**, with negligible quality loss compared to the baseline.

**Some suggestions for improving generation quality:**
1. Use High-Quality Images
  - Provide clear images, preferably with a resolution **greater than 512×512 pixels**.
  - Small or blurry inputs will result in low-quality outputs.
2. Be Specific with Instructions
  - Clearly describe both **what to change** and **how you want it changed**.

3. Prioritize English
The model currently performs best with **English** prompts.

4. Change instructions to enhance subject consistency.
When the generated image does not align well with the input image, you can try the following methods to improve subject consistency:
  - **Use images with larger size, as well as images in which people occupy a larger proportion of the frame.**
  - **Increase the Image Guidance Scale**, for example to 3.0. The trade-off may be slight overexposure or a greasy look in the image. 
  - **When using a single input image**, you can try to use the following prompt template: "she/he ..., maintaining her/his facial features, hairstyle, and other attributes."
  - **Increase the parameter--Number of images per prompt** to generate more outputs, giving you a better chance to find one with stronger subject consistency and a more satisfactory result. 
  - **Longer prompts generally yield better results than shorter ones.** More detailed descriptions of the scene and character interactions can provide additional benefits.

5. For in-context edit (edit based multiple images), we recommend using the following prompt format: "Edit the first image: add/replace (the [object] with) the [object] from the second image. [descripton for your target image]." 
For example: "Edit the first image: add the man from the second image. The man is talking with a woman in the kitchen". The descition for your target image should be as detailed as possible.

## 🎨 Fine-tune
See [fine-tuning](docs/FINETUNE.md) for details.

## ❌ Limitations and Suggestions
The current model sometimes does not follow instructions. You can increase the "Number of images per prompt" to generate multiple images at once, so you can choose the result you are satisfied with, or try different prompts. In our own experience, being as detailed as possible tends to work better.

The current model cannot decide the output image size by itself; the default size is 1024×1024. You need to set a specific size if you require a different one. When you input an image, we will set the output size to match the input image (this works best for editing tasks). If you want to modify just one image out of several, you should also set the output size to match the image you want to edit; otherwise, it may lead to low-quality outputs.

The in-context generation capability sometimes produces objects that differ from the original ones. Some suggested improvements are: increasing `image_guidance_scale` (it is recommended to set it to 3) can help alleviate this issue; using high-resolution images, increasing the size of the input image, and ensuring that the object to be used occupies a larger proportion of the image; and modifying the prompt. However, there is still a  gap compared to GPT-4o.

Compared to OmniGen 1.0, although OmniGen 2 has made some improvements, many issues still remain. It may take multiple attempts to achieve a satisfactory result. 


## 💻 Resources Requirement
OmniGen2 natively requires an **NVIDIA RTX 3090** or an equivalent GPU with approximately **17GB of VRAM**. For devices with less VRAM, you can enable **CPU Offload** to run the model.

**Performance Tip**: To improve inference speed, consider decreasing the `cfg_range_end` parameter. Within a reasonable range, this has a negligible impact on output quality.

The following table details the inference performance of OmniGen2 on an **A800 GPU**:
<p align="center">
  <img src="assets/efficiency.png" width="95%">
  <br>
  <em>Inference Efficiency of OmniGen2.</em>
</p>

## 🤝 Community Efforts
We’re honored and grateful for the support from the open source community. Here are some unofficial implementations contributed by the community(**Currently, we have not confirmed whether there are no bugs. Please try to use the our official demo as much as possible.**):
- ComfyUI:
  - [ComfyUI Official](https://comfyanonymous.github.io/ComfyUI_examples/omnigen/)
  - [https://github.com/Yuan-ManX/ComfyUI-OmniGen2](https://github.com/Yuan-ManX/ComfyUI-OmniGen2)
  - [https://github.com/neverbiasu/ComfyUI-OmniGen2](https://github.com/neverbiasu/ComfyUI-OmniGen2)
- Quantization:
  - [DFloat11, a lossless compression using 11 bits](https://github.com/LeanModels/OmniGen2-DFloat11)

## ❤️ Citing Us
If you find this repository or our work useful, please consider giving a star ⭐ and citation 🦖, which would be greatly appreciated:

```bibtex
@article{wu2025omnigen2,
  title={OmniGen2: Exploration to Advanced Multimodal Generation},
  author={Chenyuan Wu and Pengfei Zheng and Ruiran Yan and Shitao Xiao and Xin Luo and Yueze Wang and Wanli Li and Xiyan Jiang and Yexin Liu and Junjie Zhou and Ze Liu and Ziyi Xia and Chaofan Li and Haoge Deng and Jiahao Wang and Kun Luo and Bo Zhang and Defu Lian and Xinlong Wang and Zhongyuan Wang and Tiejun Huang and Zheng Liu},
  journal={arXiv preprint arXiv:2506.18871},
  year={2025}
}
```
