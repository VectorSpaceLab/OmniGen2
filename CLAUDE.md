# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup

#### For CUDA (Linux/Windows with NVIDIA GPU)
```bash
# Setup conda environment
conda create -n omnigen2 python=3.11
conda activate omnigen2

# Install PyTorch with CUDA support
pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

#### For MPS (Mac with Apple Silicon M1/M2/M3/M4)
```bash
# Setup conda environment
conda create -n omnigen2 python=3.11
conda activate omnigen2

# Install PyTorch with MPS support
pip install torch==2.6.0 torchvision

# Install dependencies
pip install -r requirements.txt
```

### Running Examples
```bash
# Text-to-image generation
bash example_t2i.sh

# Image editing with instructions
bash example_edit.sh

# Visual understanding/analysis
bash example_understanding.sh

# In-context generation
bash example_in_context_generation.sh
```

### Gradio Applications
```bash
# Image generation interface
python app.py

# Chat interface (text + image generation)
python app_chat.py

# With public sharing
python app.py --share
```

### Direct Inference
```bash
# Basic inference (uses float32 by default for MPS compatibility)
python inference.py --model_path OmniGen2/OmniGen2 \
  --instruction "your prompt here" \
  --output_image_path outputs/result.png

# Chat inference (multimodal)
python inference_chat.py --model_path OmniGen2/OmniGen2 \
  --instruction "describe this image" \
  --input_image_path path/to/image.jpg

# For Mac with limited memory, enable CPU offloading
python inference.py --model_path OmniGen2/OmniGen2 \
  --enable_model_cpu_offload \
  --instruction "your prompt here" \
  --output_image_path outputs/result.png

# Specify different precision (fp32 is default for MPS compatibility)
python inference.py --model_path OmniGen2/OmniGen2 \
  --dtype fp32 \
  --instruction "your prompt here" \
  --output_image_path outputs/result.png
```

## Architecture Overview

OmniGen2 is a multimodal generative AI model with distinct text and image decoding pathways. The codebase follows a modular pipeline architecture:

### Core Components
- **Pipeline**: `omnigen2/pipelines/omnigen2/` - Main inference pipelines
  - `pipeline_omnigen2.py` - Standard image generation pipeline
  - `pipeline_omnigen2_chat.py` - Chat/multimodal pipeline
- **Models**: `omnigen2/models/` - Core model architectures
  - `transformers/transformer_omnigen2.py` - Main transformer model
  - `embeddings.py` - Text and image embedding layers
  - `attention_processor.py` - Custom attention mechanisms
- **Schedulers**: `omnigen2/schedulers/` - Noise scheduling for diffusion
  - `scheduling_flow_match_euler_discrete.py` - Flow matching scheduler
  - `scheduling_dpmsolver_multistep.py` - DPM solver scheduler
- **Utils**: `omnigen2/utils/` - Utility functions for image processing

### Key Model Features
- Built on Qwen-VL-2.5 foundation for visual understanding
- Dual decoding pathways for text and image generation
- Support for four main capabilities:
  1. Visual Understanding
  2. Text-to-Image Generation
  3. Instruction-guided Image Editing
  4. In-context Generation

### Entry Points
- `app.py` - Gradio web interface for image generation
- `app_chat.py` - Gradio chat interface for multimodal interaction
- `inference.py` - Command-line inference for image generation
- `inference_chat.py` - Command-line multimodal inference

## Key Parameters for Generation

### Essential Hyperparameters
- `text_guidance_scale` (default 4.0-5.0): Controls adherence to text prompts
- `image_guidance_scale` (default 1.2-3.0): Controls reference image fidelity
  - Use 1.2-2.0 for editing tasks
  - Use 2.5-3.0 for in-context generation
- `max_pixels` (default 1024×1024): Automatic image resizing limit
- `num_inference_step` (default 50): Denoising steps
- `enable_model_cpu_offload`: Reduces VRAM usage by ~50%
- `enable_sequential_cpu_offload`: Minimizes VRAM to <3GB (slower)

### Performance Optimization
- Use `cfg_range_start` and `cfg_range_end` to reduce inference time
- Enable CPU offloading for devices with limited memory
- **Model uses float32 precision by default for MPS compatibility**
- **Triton operations disabled for cross-platform compatibility**
- **Uses local implementations instead of remote code to avoid triton dependencies**
- **MPS-specific optimizations: tensor validation, fallback attention, and CPU backup for problematic operations**
- Hardware requirements:
  - NVIDIA: RTX 3090 (17GB VRAM) without CPU offload
  - Mac Apple Silicon: M1/M2/M3/M4 with 16GB+ unified memory recommended

## Model Loading Patterns

The codebase uses a two-stage loading pattern:
1. Load base pipeline from model path with `trust_remote_code=False`
2. Separately load transformer model from subfolder
3. Apply CPU offloading if enabled

This pattern is consistent across `app.py`, `inference.py`, and pipeline implementations. Using `trust_remote_code=False` ensures local implementations are used instead of potentially incompatible remote code.