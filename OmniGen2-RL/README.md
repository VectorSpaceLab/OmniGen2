# 🚀 Advanced Applications of EditScore

Welcome to the examples directory! Here, we demonstrate how to leverage **EditScore** not just as an evaluation metric, but as a powerful component to actively *improve image editing models*.

This guide covers two primary downstream applications:
1. **Best-of-N selection**: A simple, training-free method to instantly boost the output quality of any image editing model.
2. **Reinforcement Learning (RL) Fine-Tuning**: Using EditScore as a high-fidelity reward signal to train models for significantly better performance.

## 🛠️ Setup for Examples
The examples require libraries for RL, data handling, and potentially experiment tracking.
```bash
# Navigate to this directory if you are in the root
cd examples/OmniGen2-RL

# Install the required packages
pip install -r requirements.txt
```

## Application 1: Best-of-N for Superior Outputs
Best-of-N is an elegant and powerful technique. Instead of generating a single output for a given instruction, you generate multiple (N) candidates and then use a highly accurate evaluator—**EditScore**—to select the best one.

This acts as a powerful "reranker" that filters out suboptimal results, significantly improving the perceived quality of the model without any extra training.

### How to Use
We provide ready-to-use scripts to perform a full Best-of-N workflow on the GEdit-Bench benchmark. The following instructions use **OmniGen2** as the base model, but we provide similar scripts for **FLUX-Kontext** and **Qwen-Image-Edit** in the `evaluation/GEdit-Bench/` directory.

**1. Generate Candidates**
```bash
bash evaluation/GEdit-Bench/omnigen2_16samples.sh # default using 8 GPUs
```

> **⚠️ Important Note on Resource Usage**
>
> This process is computationally expensive and slow due to the large number of generations (16 samples per instruction). For reference, completing this step for OmniGen2 takes approximately **3 hours using 64 H100 GPUs**.

<details>
<summary><strong>👉 Click here for tips on the usage of the script</strong></summary>

- **Distributed Inference**: Our scripts natively support multi-machine and multi-GPU execution. To run inference across 4 machines, for example, execute the following commands on each respective machine:
```bash
# On the first machine (rank 0)
bash evaluation/GEdit-Bench/omnigen2_16samples.sh --world_size 4 --rank 0

# On the second machine (rank 1)
bash evaluation/GEdit-Bench/omnigen2_16samples.sh --world_size 4 --rank 1

# ...and so on for ranks 2 and 3.
```

- **Monitoring Progress**: The scripts utilize nohup for background execution. We recommend monitoring the file (specified in the script file) to track the status and progress of the generation process.
</details>

**2. Score and Select**
Next, use EditScore to evaluate all N candidates and identify the one with the highest score.

```bash
bash evaluation/GEdit-Bench/omnigen2_16samples_select_best_editscore_pass1.sh # EditScore-7B, single pass
bash evaluation/GEdit-Bench/omnigen2_16samples_select_best_editscore_pass4.sh # EditScore-7B, Avg@4
```

**3. Evaluate the Final Selections**
Finally, evaluate the performance of the images selected by EditScore on GEdit-Bench to quantify the improvement.
```bash
bash evaluation/GEdit-Bench/omnigen2_16samples_select_best_editscore_pass1_eval.sh
bash evaluation/GEdit-Bench/omnigen2_16samples_select_best_editscore_pass4_eval.sh
```

By comparing these results to the baseline performance of the original model, you will see the benefits of applying EditScore as a reranker.

## Application 2: Reinforcement Fine-Tuning
Beyond evaluation, **EditScore** can be used as a high-quality reward signal to fine-tune your image editing models using Reinforcement Learning (RL), leading to significantly improved performance.

We employ the **FlowGRPO** algorithm, combining its strengths with EditScore's accurate, real-time feedback to create a powerful end-to-end fine-tuning pipeline. This process effectively guides the model toward generating better edits.

### 1. Prepare Training Data
First, set up the dataset for RL fine-tuning.
1. Download the Data
Downlaod the official RL training data from [EditScore-RL-Data](https://huggingface.co/datasets/EditScore/EditScore-RL-Data).
2. Create Meta File
The uploaded dataset uses relative image paths. Run the following script to convert them to absolute paths based on your local environment:
```bash
# Then
python scripts/data/process_jsonl.py --input /path/to/EditScore-RL-Data/rl.jsonl --output /path/to/EditScore-RL-Data/rl_abs.jsonl --base-path /path/to/EditScore-RL-Data

# Due to the limitation of base model (OmniGen2), we discard text change and portrait beautification, as these tasks harm RL training.
python scripts/data/extract_9_tasks.py --input_path /path/to/EditScore-RL-Data/rl_abs.jsonl --output_path /path/to/EditScore-RL-Data/rl_abs_9tasks.jsonl
```
3. Configure the Data Path
Specify the path to your processed `.jsonl` file in the data configuration located at `data_configs/train/example/edit/all.yml`.
For example:
```yaml
ratio_type: inside_ratio

data:
  - 
    path: '/path/to/EditScore-RL-Data/rl_abs_9tasks.jsonl' # <-- Ensure this path is correct
    type: 'edit'
    ratio: !!float 1
```

### 2. Prepare the Base Model (OmniGen2)
```bash
python scripts/misc/extract_bin_from_pipe.py
```

### 3. Launch the Reward Server
RL training requires a live reward signal. Before starting the training process, you must launch the **EditScore Reward Server**. This server will provide real-time scores for the generated images during training.

Our reward server is built with two components: a **proxy** and one or more **reward servers**. The proxy receives requests from the training node, distributes them to the individual reward servers for computation, and then collects the results to send back. This architecture allows for easy scaling across multiple machines.

We provide a convenient script to launch the entire server stack across multiple machines, assuming you have `ssh` access to all reward server nodes.

```bash
# Launch EditScore-7B Reward Server
bash reward_server/start_multi_machines.sh --model_name=editscore_7B --config_path=reward_server/server_configs/editscore_7B.yml

# Launch EditScore-7B (Avg@4) Reward Server
bash reward_server/start_multi_machines.sh --model_name=editscore_7B_pass4 --config_path=reward_server/server_configs/editscore_7B_pass4.yml

# Launch EditScore-72B Reward Server
bash reward_server/start_multi_machines.sh --model_name=editscore_72B --config_path=reward_server/server_configs/editscore_72B.yml
```

> **⚠️ Important Notes**
>
> *   Before running the script, you **must** specify the IP addresses of your reward server machines in the corresponding `.yml` configuration file.
> *   If you cannot use `ssh` to control the nodes, please refer to the logic in `reward_server/start_multi_machines.sh` to manually start the proxy and server processes on each machine.
> *   You can monitor the status of the proxy and servers by checking the log files in the `reward_server/logs/` directory.

## 3.5 (Optional) Reward Server Sanity Check
To ensure the reward server is configured correctly and running as expected, we provide a sanity check script.
```bash
python reward_server/scripts/utils/reward_server_sanity_check.py --config_path=reward_server/server_configs/editscore_7B.yml
```
Once these steps are complete, your environment is ready to begin the reinforcement learning fine-tuning process.

### 4. Start RL Fine-Tuning

**Configure Training Parameters**
Before launching, you may need to adjust key parameters in the configuration file: `options/omnigen2_edit_rl_4machine_editscore7b_avg4.yml`.

Here are some important settings:
- `train.global_batch_size`: The total number of images generated across all GPUs in a single sampling phase before the policy is updated. It is calculated as `num_unique_prompts_per_sampling * num_images_per_prompt`.
- `train.batch_size`: Batch size per GPU (`batch_size_per_forward * gradient_accumulation_steps * num_update_steps_per_sampling`)
- `train.rl.num_images_per_prompt`: The number of candidate images to generate for each unique prompt.
- `train.rl.num_unique_prompts_per_sampling`: The number of unique prompts in a global batch
- `train.rl.num_update_steps_per_sampling`: The number of gradient updates to perform in each sampling phase. Set this to `> 1` to enable off-policy RL, which improves sample efficiency.
- `train.rl.batch_size_per_forward`: Batch size for each forward pass. Together with `num_update_steps_per_sampling`, it defines the total number of samples processed per policy update.

**Launch Distributed Training**
We provide scripts for both single and multi-machine distributed training based on **FSDP**.
```bash
# Single-machine training (8 GPUs) using EditScore-7B as the reward model
bash scripts/train/omnigen2_edit_rl_single_machine_editscore7b.sh

# Multi-machine training (e.g., 4 machines with 8 GPUs each) using EditScore-7B (Avg@4)
bash scripts/train/omnigen2_edit_rl_4machine_editscore7b_avg4.sh
```

### 4. Training Outputs and Monitoring
All training artifacts, including logs and model checkpoints, are saved to the `experiments/` directory.
For transparency and to ensure full reproducibility, we provide the training curves of **OmniGen2-EditScore7B-v1.1** on [Weights & Biases (wandb)](https://wandb.ai/omnigen-rl/OmniGen2-RL/reports/Training-Curves-of-OmniGen2-EditScore7B-v1-1---VmlldzoxNDg2MTI2NQ?accessToken=467cbc4jupu1maluk00pan7z611m6xxmnwviwk8nblml3ydoy3j9fu92el6c0s8i).


### 5. Evaluate your RL Fine-Tuned Model
After training, you must convert the FSDP-saved checkpoint (`.bin`) into the standard Hugging Face format before you can use it for inference.

#### Step 1: Convert the Checkpoint
We provide a script to automatically handle the conversion from the distributed FSDP format to the standard Hugging Face format (`.bin`).
Run the following command, replacing the arguments with your experiment's details:

```shell
bash scripts/misc/convert_dist_ckpt_to_hf_format.sh [EXPERIMENT_NAME] [STEP_NUMBER]
```
- [EXPERIMENT_NAME]: The name of your training experiment (e.g., omnigen2_edit_rl_single_machine_editscore7b).
- [STEP_NUMBER]: The specific training step of the checkpoint you wish to evaluate (e.g., 500).

This will create a new directory containing the converted model weights in the standard format, ready for inference.

#### Step 2: Run Evaluation on GEdit-Bench
Once the checkpoint is converted, you can benchmark its performance. We provide evaluation scripts tailored for GEdit-Bench.

You can use our example scripts as a template. Simply copy one and modify the internal paths to point to your newly converted model checkpoint.
```shell
# Run evaluation for the converted model from step 500
bash evaluation/GEdit-Bench/omnigen2.sh --experiment_name=omnigen2_edit_rl_4machine_editscore7b_avg4 --step=500
bash evaluation/GEdit-Bench/omnigen2_eval.sh --experiment_name=omnigen2_edit_rl_4machine_editscore7b_avg4 --step=500
```
By comparing the results to the baseline model's performance, you can quantify the improvements achieved through RL fine-tuning with EditScore.