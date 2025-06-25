# 🍎 OmniGen2 for Apple Silicon (macOS)

This is a comprehensive guide for running OmniGen2 on Apple Silicon Macs (M1/M2/M3/M4) with MPS (Metal Performance Shaders) acceleration.

## 🎯 What's Modified for Apple Silicon

This fork includes critical optimizations to make OmniGen2 fully compatible with Apple Silicon:

- **✅ MPS Backend Support**: Utilizes Apple's Metal Performance Shaders for GPU acceleration
- **✅ Flash Attention Removal**: Removed CUDA-specific flash attention dependencies
- **✅ Triton Dependencies Eliminated**: Removed all triton operations that are incompatible with MPS
- **✅ Float32 Precision**: Enforced float32 precision for optimal MPS compatibility
- **✅ Enhanced Error Handling**: Added MPS-specific fallbacks for attention operations
- **✅ Improved Gradio UI**: Modern, enhanced user interface with better organization

## 🚀 Quick Start for macOS

### 🛠️ Environment Setup

```bash
# 1. Clone this optimized repository
git clone https://github.com/zettai-seigi/OmniGen2.git
cd OmniGen2

# 2. Create a clean Python environment
conda create -n omnigen2 python=3.11
conda activate omnigen2

# 3. Install PyTorch with MPS support
pip install torch==2.6.0 torchvision torchaudio

# 4. Install other required packages
pip install -r requirements.txt

# Note: Do NOT install flash-attn on Apple Silicon - it's not needed and won't work
```

### 🧪 Run Examples

```bash
# Text-to-image generation (recommended to start with)
python inference.py --model_path "OmniGen2/OmniGen2" --instruction "A beautiful landscape with mountains at sunset"

# Visual Understanding
python inference_chat.py --model_path "OmniGen2/OmniGen2" --instruction "Describe this image" --input_image_path "path/to/your/image.jpg"

# Image editing
python inference.py --model_path "OmniGen2/OmniGen2" --instruction "Add a bird to this scene" --input_image_path "path/to/your/image.jpg"
```

### 🌐 Enhanced Gradio Interface

Launch the beautiful, optimized Gradio interface:

```bash
# Enhanced chat interface (recommended)
python app_chat_enhanced.py

# Standard interfaces
python app.py          # Text-to-image only
python app_chat.py     # Chat interface
```

The enhanced interface features:
- 🎨 Modern gradient design with hover effects
- 📱 Better mobile responsiveness  
- 🔧 Organized settings with collapsible sections
- 💡 Built-in usage tips and guidance
- 💾 Auto-save functionality
- 🍎 Apple Silicon optimization indicators

## ⚙️ Apple Silicon Specific Settings

### Recommended Parameters for Apple Silicon:

```python
# For optimal performance on Apple Silicon
dtype = "fp32"  # Always use float32 on MPS
enable_model_cpu_offload = True  # Recommended for memory management
text_guidance_scale = 5.0
image_guidance_scale = 2.0
num_inference_steps = 50
```

### Memory Management:

- **8GB RAM**: Use `--enable_model_cpu_offload` for better memory efficiency
- **16GB+ RAM**: Can run without offloading for faster performance
- **Unified Memory**: Apple Silicon's unified memory architecture helps with large models

## 🔧 Key Modifications Made

### 1. **Flash Attention Removal** (`omnigen2/utils/import_utils.py`)
```python
def is_flash_attn_available():
    return False  # Disabled for MPS compatibility

def is_triton_available():
    return False  # Disabled for MPS compatibility
```

### 2. **MPS-Safe Attention** (`omnigen2/models/attention_processor.py`)
- Removed `OmniGen2AttnProcessorFlash2Varlen` class
- Enhanced error handling for MPS tensor operations
- Added attention mask validation for MPS devices

### 3. **Float32 Enforcement** (All inference files)
```python
bf16 = False  # Always use float32 on Apple Silicon
weight_dtype = torch.float32
```

### 4. **Manual Component Loading**
- Disabled `trust_remote_code=True` to avoid triton dependencies
- Manual loading of transformer, VAE, scheduler, MLLM, and processor components

## 🚨 Troubleshooting

### Common Issues:

1. **"triton" module not found**
   - ✅ **Fixed**: This fork removes all triton dependencies

2. **MPS tensor dimension errors**
   - ✅ **Fixed**: Added MPS-specific error handling and fallbacks

3. **Flash attention import errors**
   - ✅ **Fixed**: Flash attention is completely disabled

4. **Performance slower than expected**
   - Try enabling model CPU offload: `--enable_model_cpu_offload`
   - Reduce inference steps: `--num_inference_step 30`
   - Lower image resolution if needed

### Performance Tips:

- **Best Performance**: 16GB+ unified memory, no CPU offload
- **Memory Constrained**: Use `--enable_model_cpu_offload`
- **Very Limited Memory**: Use `--enable_sequential_cpu_offload` (slower but uses <3GB VRAM)

## 📊 Performance Benchmarks

On Apple Silicon M3 Max (36GB unified memory):
- **Text-to-Image (1024x1024)**: ~45-60 seconds (50 steps)
- **Image Editing**: ~40-55 seconds (50 steps)  
- **Memory Usage**: ~12-15GB (without offload), ~8-10GB (with model offload)

## 🤝 Contributing

This Apple Silicon optimization is maintained by [zettai-seigi](https://github.com/zettai-seigi). 

Issues specific to Apple Silicon should be reported here: [GitHub Issues](https://github.com/zettai-seigi/OmniGen2/issues)

For general OmniGen2 issues, please refer to the [original repository](https://github.com/VectorSpaceLab/OmniGen2).

## 📄 License

This project maintains the same license as the original OmniGen2 project.

---

**🍎 Optimized for Apple Silicon by [zettai-seigi](https://github.com/zettai-seigi)**