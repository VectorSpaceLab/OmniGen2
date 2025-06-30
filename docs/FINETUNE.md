## 🎨 Fine-Tuning OmniGen2

You can fine-tune OmniGen2 to customize its capabilities, enhance its performance on specific tasks, or address potential limitations.

We provide a training script that supports multi-GPU and multi-node distributed training using **PyTorch FSDP (Fully Sharded Data Parallel)**. Both full-parameter fine-tuning and **LoRA (Low-Rank Adaptation)** are supported out of the box.

### 1. Preparation

Before launching the training, you need to prepare the following configuration files.

#### Step 1: Set Up the Training Configuration

This is a YAML file that specifies crucial parameters for your training job, including the model architecture, optimizer, dataset paths, and validation settings.

We provide two templates to get you started:
*   **Full-Parameter Fine-Tuning:** `options/ft.yml`
*   **LoRA Fine-Tuning:** `options/ft_lora.yml`

Copy one of these templates and modify it according to your needs. Below are some of the most important parameters you may want to adjust:
- `name`: The experiment name. This is used to create a directory for logs and saved model weights (e.g., `experiments/your_exp_name`).
- `data.data_path`: Path to the data configuration file that defines your training data sources and mixing ratios.
- `data.max_output_pixels`: The maximum number of pixels for an output image. Larger images will be downsampled while maintaining their aspect ratio.
- `data.max_input_pixels`: A list specifying the maximum pixel count for input images, corresponding to one, two, three, or more inputs.
- `data.max_side_length`: The maximum side length for any image (input or output). Images exceeding this will be downsampled while maintaining their aspect ratio..
- `train.global_batch_size`: The total batch size across all GPUs. This should equal `batch_size` × `(number of GPUs)`.
- `train.batch_size`: The batch size per GPU.
- `train.max_train_steps`: The total number of training steps to run.
- `train.learning_rate`: The learning rate for the optimizer. **Note:** This often requires tuning based on your dataset size and whether you are using LoRA. We recommend using lower learning rate for full-parameter fine-tuning.
- `logger.log_with`: Specify which loggers to use for monitoring training (e.g., `tensorboard`, `wandb`).

#### Step 2: Configure Your Dataset

The data configuration consists of a set of `yaml` and `jsonl` files.
*   The `.yml` file defines the mixing ratios for different data sources.
*   The `.jsonl` files contain the actual data entries, with each line representing a single data sample.

For a practical example, please refer to `data_configs/train/example/mix.yml`.
Each line in a `.jsonl` file describes a sample, generally following this format:
```json
{
  "task_type": "edit",
  "instruction": "add a hat to the person",
  "input_images": ["/path/to/your/data/edit/input1.png", "/path/to/your/data/edit/input2.png"],
  "output_image": "/path/to/your/data/edit/output.png"
}
```
*Note: The `input_images` field can be omitted for text-to-image (T2I) tasks.*

#### Step 3: Review the Training Scripts

We provide convenient shell scripts to handle the complexities of launching distributed training jobs. You can use them directly or adapt them for your environment.

*   **For Full-Parameter Fine-Tuning:** `scripts/train/ft.sh`
*   **For LoRA Fine-Tuning:** `scripts/train/ft_lora.sh`

---

### 2. 🚀 Launching the Training

Once your configuration is ready, you can launch the training script. All experiment artifacts, including logs and checkpoints, will be saved in `experiments/${experiment_name}`.

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
  --config_path experiments/ft/ft.yml \
  --model_path experiments/ft/checkpoint-10/pytorch_model_fsdp.bin \
  --save_path experiments/ft/checkpoint-10/transformer
```

**For a LoRA fine-tuned model:**
```shell
python convert_ckpt_to_hf_format.py \
  --config_path experiments/ft_lora/ft_lora.yml \
  --model_path experiments/ft_lora/checkpoint-10/pytorch_model_fsdp.bin \
  --save_path experiments/ft_lora/checkpoint-10/transformer_lora
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
  --transformer_path experiments/ft/checkpoint-10/transformer
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
  --transformer_lora_path experiments/ft_lora/checkpoint-10/transformer_lora
```