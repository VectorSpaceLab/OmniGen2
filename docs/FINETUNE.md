Of course! Here is an optimized, all-English version of your GitHub README section. I've focused on creating a clear, step-by-step structure, improving the language for clarity, and ensuring it's easy for any user to follow.

---

## ⚙️ Fine-Tuning OmniGen2

You can fine-tune OmniGen2 to customize its capabilities, enhance its performance on specific tasks, or address potential limitations.

We provide a training script that supports multi-GPU and multi-node distributed training using **PyTorch FSDP (Fully Sharded Data Parallel)**. Both full-parameter fine-tuning and **LoRA (Low-Rank Adaptation)** are supported out of the box.

### 1. Preparation

Before launching the training, you need to prepare the following configuration files.

#### Step 1: Set Up the Training Configuration

This is a YAML file that specifies crucial parameters for your training job, including the model architecture, optimizer, dataset paths, and validation settings.

We provide two templates to get you started:
*   **Full-Parameter Fine-Tuning:** `options/ft.yml`
*   **LoRA Fine-Tuning:** `options/ft_lora.yml`

Copy one of these templates and modify it according to your needs.

#### Step 2: Configure Your Dataset

The data configuration consists of a set of `yaml` and `jsonl` files.
*   The `.yml` file defines the mixing ratios for different data sources.
*   The `.jsonl` files contain the actual data entries, with each line representing a single data sample.

For a practical example, please refer to `data_configs/train/example/mix.yml`.

#### Step 3: Review the Training Scripts

We provide convenient shell scripts to handle the complexities of launching distributed training jobs. You can use them directly or adapt them for your environment.

*   **For Full-Parameter Fine-Tuning:** `scripts/train/ft.sh`
*   **For LoRA Fine-Tuning:** `scripts/train/ft_lora.sh`

---

### 2. 🚀 Launching the Training

Once your configuration is ready, you can launch the training script.

#### Multi-Node / Multi-GPU Training

For distributed training across multiple nodes or GPUs, you need to provide environment variables to coordinate the processes.

```shell
# Example for full-parameter fine-tuning
bash scripts/train/ft.sh --rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --world_size=$WORLD_SIZE
```

#### Single-Node Training

If you are training on a single machine (with one or more GPUs), you can omit the distributed arguments. The script will handle the setup automatically.

```shell
# Example for full-parameter fine-tuning on a single node
bash scripts/train/ft.sh
```

> **⚠️ Note on LoRA Checkpoints:**
> Currently, when training with LoRA, the script saves the entire model's parameters (including the frozen base model weights) in the checkpoint. This is due to a limitation in easily extracting only the LoRA-related parameters when using FSDP. The conversion script in the next step will correctly handle this.

---

### 3. 🖼️ Inference with Your Fine-Tuned Model

After training, you must convert the FSDP-saved checkpoint (`.bin`) into the standard Hugging Face format before you can use it for inference.

#### Step 1: Convert the Checkpoint

We provide a conversion script that automatically handles both full-parameter and LoRA checkpoints.

**For a full fine-tuned model:**
```shell
python convert_ckpt_to_hf_format.py \
  --config_path experiments/train_edit/train_edit.yml \
  --model_path experiments/train_edit/checkpoint-10/pytorch_model_fsdp.bin \
  --save_path experiments/train_edit/checkpoint-10/transformer
```

**For a LoRA fine-tuned model:**
```shell
python convert_ckpt_to_hf_format.py \
  --config_path experiments/train_edit_lora/train_edit_lora.yml \
  --model_path experiments/train_edit_lora/checkpoint-10/pytorch_model_fsdp.bin \
  --save_path experiments/train_edit_lora/checkpoint-10/transformer_lora
```

#### Step 2: Run Inference

Now, you can run the inference script, pointing it to the path of your converted model weights.

**Using a full fine-tuned model:**
Pass the converted model path to the `--transformer_path` argument.

```shell
python inference.py \
  --model_path "OmniGen2/OmniGen2" \
  --num_inference_step 50 \
  --height 1024 \
  --width 1024 \
  --text_guidance_scale 4.0 \
  --instruction "A crystal ladybug on a dewy rose petal in an early morning garden, macro lens." \
  --output_image_path outputs/output_t2i_finetuned.png \
  --num_images_per_prompt 1 \
  --transformer_path experiments/train_edit/checkpoint-10/transformer
```

**Using a LoRA fine-tuned model:**
Pass the converted LoRA weights path to the `--transformer_lora_path` argument.

```shell
python inference.py \
  --model_path "OmniGen2/OmniGen2" \
  --num_inference_step 50 \
  --height 1024 \
  --width 1024 \
  --text_guidance_scale 4.0 \
  --instruction "A crystal ladybug on a dewy rose petal in an early morning garden, macro lens." \
  --output_image_path outputs/output_t2i_lora.png \
  --num_images_per_prompt 1 \
  --transformer_lora_path experiments/train_edit_lora/checkpoint-10/transformer_lora
```