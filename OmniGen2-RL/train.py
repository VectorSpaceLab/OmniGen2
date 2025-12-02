import dotenv

dotenv.load_dotenv(override=True)

from typing import Union, List, Optional, Tuple
import time
from contextlib import contextmanager
from copy import deepcopy
import argparse
from collections import defaultdict
import logging
import math
import os
import json
import random
import shutil
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import numpy as np

import matplotlib.pyplot as plt

import torch

import torch.nn.functional as F
import torch.utils.checkpoint

from torchvision.transforms.functional import crop, to_pil_image, to_tensor

from einops import repeat, rearrange

import accelerate
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DataLoaderConfiguration
from accelerate import init_empty_weights
from accelerate.utils import gather_object

import transformers
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLModel as TextEncoder

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from peft import LoraConfig

from omnigen2.training_utils import EMAModel
from omnigen2.utils.logging_utils import TqdmToLogger
from omnigen2.utils.tensor_util import pad_to_length, expand_as
from omnigen2.dataset.omnigen2_train_dataset import OmniGen2TrainDataset, OmniGen2Collator, RepeatedDistributedBatchSampler
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
from omnigen2.grpo.reward_client_edit import evaluate_images
from omnigen2.grpo.utils import forward_logprob, process_grpo_rewards, compute_single_step_ppo_loss
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import FMPipelineOutput


logger = get_logger(__name__)

    
def parse_args(root_path) -> OmegaConf:
    parser = argparse.ArgumentParser(description="OmniGen2 training script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML format)",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=None,
        help="Global batch size.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Data path.",
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    output_dir = os.path.join(root_path, 'experiments', conf.name)
    conf.root_dir = root_path
    conf.output_dir = output_dir
    conf.config_file = args.config

    # Override config with command line arguments
    if args.global_batch_size is not None:
        conf.train.global_batch_size = args.global_batch_size
    
    if args.data_path is not None:
        conf.data.data_path = args.data_path
    return conf

def setup_logging(args: OmegaConf, accelerator: Accelerator) -> None:
    """
    Set up logging configuration for training.
    
    Args:
        accelerator: Accelerator instance
        args: Configuration object
        logging_dir: Directory for log files
    """

    logging_dir = Path(args.output_dir, "logs")
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config_file, args.output_dir)
        
        # Create logging directory and file handler
        os.makedirs(logging_dir, exist_ok=True)
        log_file = Path(logging_dir, f'{time.strftime("%Y%m%d-%H%M%S")}.log')

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.logger.addHandler(file_handler)

    # Configure basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set verbosity for different processes
    log_level = logging.INFO if accelerator.is_local_main_process else logging.ERROR
    transformers.utils.logging.set_verbosity(log_level)
    diffusers.utils.logging.set_verbosity(log_level)


def log_model_info(name: str, model: torch.nn.Module):
    """Logs parameter counts for a given model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"--- {name} ---")
    logger.info(model)
    logger.info(f"Total parameters (M): {total_params / 1e6:.2f}")
    logger.info(f"Trainable parameters (M): {trainable_params / 1e6:.2f}")


def get_qwen2_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    max_sequence_length: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get prompt embeddings from the Qwen2 text encoder.

    Args:
        prompt: The prompt or list of prompts to encode.
        device: The device to place the embeddings on. If None, uses the pipeline's device.
        max_sequence_length: Maximum sequence length for tokenization.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The prompt embeddings tensor
            - The attention mask tensor

    Raises:
        Warning: If the input text is truncated due to sequence length limitations.
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
   
    text_inputs = tokenizer(
        prompt,
        padding="longest",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.to(device)

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because Gemma can only handle sequences up to"
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_attention_mask = text_inputs.attention_mask.to(device)
   
    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_attention_mask,
        output_hidden_states=False,
    ).last_hidden_state

    return prompt_embeds, prompt_attention_mask

@contextmanager
def disabled_adapters(model):
    try:
        # model.disable_adapters()
        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in model.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module._disable_adapters = True
                else:
                    module.disable_adapters = True
        yield
    finally:
        # model.enable_adapters()
        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in model.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module._disable_adapters = False
                else:
                    module.disable_adapters = False


def main(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=Path(args.output_dir, 'logs'))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.train.gradient_accumulation_steps
        * math.ceil(
            args.train.rl.num_inference_step
            * args.train.rl.get("train_timesteps_fraction", 1.0)
        ),
        mixed_precision=args.train.mixed_precision,
        log_with=OmegaConf.to_object(args.logger.log_with),
        project_config=accelerator_project_config,
        dataloader_config=DataLoaderConfiguration(split_batches=True),
    )

    setup_logging(args, accelerator)
    
    # Reproducibility
    if args.seed is not None:
        set_seed(args.seed, device_specific=args.get('device_specific_seed', False))

    # Set performance flags
    if args.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.train.get('benchmark_cudnn', False):
        torch.backends.cudnn.benchmark = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    ema_decay = args.train.get('ema_decay', 0)

    if args.model.pretrained_model_path:
        with init_empty_weights():
            model = OmniGen2Transformer2DModel(**args.model.arch_opt)

        state_dict = torch.load(args.model.pretrained_model_path, mmap=True, weights_only=True)
        missing, unexpect = model.load_state_dict(state_dict, assign=True, strict=False)
    else:
        model = OmniGen2Transformer2DModel(**args.model.arch_opt)
    model.train()
    
    freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
        model.config.axes_dim_rope,
        model.config.axes_lens,
        theta=10000,
    )

    if ema_decay != 0:
        model_ema = deepcopy(model)
        model_ema._requires_grad = False

    processor = AutoProcessor.from_pretrained(args.model.pretrained_text_encoder_model_name_or_path)

    text_tokenizer = processor.tokenizer
    text_tokenizer.padding_side = "right"

    if accelerator.is_main_process:
        text_tokenizer.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))

    text_encoder = TextEncoder.from_pretrained(
        args.model.pretrained_text_encoder_model_name_or_path,
        torch_dtype=weight_dtype,
    )
    if args.model.get('resize_token_embeddings', False):
        text_encoder.resize_token_embeddings(len(text_tokenizer))

    if accelerator.is_main_process:
        text_encoder.save_pretrained(os.path.join(args.output_dir, 'text_encoder'))

    log_model_info("text_encoder", text_encoder)

    vae = AutoencoderKL.from_pretrained(
        args.model.pretrained_vae_model_name_or_path,
        subfolder=args.model.get("vae_subfolder", "vae"),
        local_files_only=True,
    )
    
    logger.info(vae)
    logger.info("***** Move vae, text_encoder to device and cast to weight_dtype *****")
    # Move vae, unet, text_encoder and controlnet_ema to device and cast to weight_dtype
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    args.train.lora_ft = args.train.get('lora_ft', False)
    if args.train.lora_ft:
        model.requires_grad_(False)

        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        lora_config = LoraConfig(
            r=args.train.lora_rank,
            lora_alpha=args.train.lora_alpha,
            lora_dropout=args.train.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model.add_adapter(lora_config)

    if args.train.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.train.scale_lr:
        args.train.learning_rate = (
            args.train.learning_rate * args.train.gradient_accumulation_steps * args.train.batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    log_model_info("transformer", model)

    # Optimizer creation
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    optimizer = optimizer_class(
        trainable_params,
        lr=args.train.learning_rate,
        betas=(args.train.adam_beta1, args.train.adam_beta2),
        weight_decay=args.train.adam_weight_decay,
        eps=args.train.adam_epsilon,
    )

    logger.info("***** Prepare dataset *****")

    with accelerator.main_process_first():
        train_dataset = OmniGen2TrainDataset(
            args.data.data_path,
            tokenizer=text_tokenizer,
            num_workers=args.train.dataloader_num_workers,
            use_chat_template=args.data.use_chat_template,
            prompt_dropout_prob=args.data.get('prompt_dropout_prob', 0.0),
            ref_img_dropout_prob=args.data.get('ref_img_dropout_prob', 0.0),
            max_input_pixels=OmegaConf.to_object(args.data.get('max_input_pixels', 1024 * 1024)),
            max_output_pixels=args.data.get('max_output_pixels', 1024 * 1024),
            max_side_length=args.data.get('max_side_length', 2048),
        )

    logger.info(f"Number of training samples: {len(train_dataset)}")

    if args.seed is not None and args.get("workder_specific_seed", False):
        from omnigen2.utils.reproducibility import worker_init_fn

        worker_init_fn = partial(
            worker_init_fn,
            num_processes=AcceleratorState().num_processes,
            num_workers=args.train.dataloader_num_workers,
            process_index=AcceleratorState().process_index,
            seed=args.seed,
            same_seed_per_epoch=args.get("same_seed_per_epoch", False),
        )
    else:
        worker_init_fn = None

    logger.info("***** Prepare dataLoader *****")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.train.dataloader_num_workers,
        batch_sampler=RepeatedDistributedBatchSampler(
            dataset=train_dataset,
            batch_size=args.train.batch_size,
            num_repeats=args.train.rl.num_images_per_prompt,
            num_replicas=AcceleratorState().num_processes,
            rank=AcceleratorState().process_index,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        ),
        worker_init_fn=worker_init_fn,
        collate_fn=OmniGen2Collator(tokenizer=text_tokenizer, max_token_len=args.data.maximum_text_tokens)
    )

    logger.info(f"{args.train.batch_size=} {args.train.gradient_accumulation_steps=} {accelerator.num_processes=} {args.train.global_batch_size=}")

   
    assert args.train.batch_size % (args.train.rl.batch_size_per_forward * args.train.gradient_accumulation_steps) == 0, f"{args.train.batch_size=} % ({args.train.rl.batch_size_per_forward=} * {args.train.rl.gradient_accumulation_steps=}) != 0"
    assert args.train.batch_size // (args.train.rl.batch_size_per_forward * args.train.gradient_accumulation_steps) == args.train.rl.num_update_steps_per_sampling
    assert args.train.global_batch_size // args.train.rl.num_images_per_prompt == args.train.rl.num_unique_prompts_per_sampling, f"{args.train.global_batch_size=} // {args.train.rl.num_images_per_prompt=} != {args.train.rl.num_unique_prompts_per_sampling=}"
   

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) * args.train.rl.num_update_steps_per_sampling)
    if 'max_train_steps' not in args.train:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.train.lr_scheduler == 'timm_cosine':
        from omnigen2.optim.scheduler.cosine_lr import CosineLRScheduler

        lr_scheduler = CosineLRScheduler(optimizer=optimizer,
                                         t_initial=args.train.t_initial,
                                         lr_min=args.train.lr_min,
                                         cycle_decay=args.train.cycle_decay,
                                         warmup_t=args.train.warmup_t,
                                         warmup_lr_init=args.train.warmup_lr_init,
                                         warmup_prefix=args.train.warmup_prefix,
                                         t_in_epochs=args.train.t_in_epochs)
    elif args.train.lr_scheduler == 'timm_constant_with_warmup':
        from omnigen2.optim.scheduler.step_lr import StepLRScheduler

        lr_scheduler = StepLRScheduler(
            optimizer=optimizer,
            decay_t=1,
            decay_rate=1,
            warmup_t=args.train.warmup_t,
            warmup_lr_init=args.train.warmup_lr_init,
            warmup_prefix=args.train.warmup_prefix,
            t_in_epochs=args.train.t_in_epochs,
        )
    else:
        lr_scheduler = get_scheduler(
            args.train.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.train.lr_warmup_steps,
            num_training_steps=args.train.max_train_steps,
            num_cycles=args.train.lr_num_cycles,
            power=args.train.lr_power,
        )

    logger.info("***** Prepare everything with our accelerator *****")

    if args.train.ema_decay != 0:
        model, model_ema, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, model_ema, optimizer, train_dataloader, lr_scheduler
        )
        model_ema = EMAModel(model_ema.parameters(), decay=ema_decay, model_cls=type(unwrap_model(model)), model_config=model_ema.config)
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) * args.train.rl.num_update_steps_per_sampling)
    if overrode_max_train_steps:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.train.num_train_epochs = math.ceil(args.train.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("OmniGen2-RL", init_kwargs={"wandb": {"name": args.name}})

    # Train!
    total_batch_size = args.train.batch_size * accelerator.num_processes * args.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.train.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.train.max_train_steps}")
    global_step = 0
    first_epoch = 0
        
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.train.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        file=TqdmToLogger(logger, level=logging.INFO)
    )

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                logger.info(f"***** Wandb log dir: {tracker.run.dir} *****")

    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.schedulers.scheduling_flow_match_euler_maruyama_discrete import FlowMatchEulerMaruyamaDiscreteScheduler
    pipeline = OmniGen2Pipeline(
        transformer=model,
        vae=vae,
        scheduler=FlowMatchEulerMaruyamaDiscreteScheduler(
            sigma_coef=args.train.rl.get('sigma_coef', 0.7),
            time_shift_base_res=args.train.rl.get('time_shift_base_res', 320)
        ),
        mllm=None,
        processor=processor,
    )
    pipeline.set_progress_bar_config(disable=True)

    with torch.no_grad():
       
        if args.train.rl.get('use_ori_neg_prompt_template', False):
            negative_prompt = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": args.train.rl.negative_prompt},
            ]
            negative_prompt = pipeline.processor.tokenizer.apply_chat_template(
                negative_prompt, tokenize=False, add_generation_prompt=False
            )
        else:
            negative_prompt = pipeline._apply_chat_template(args.train.rl.negative_prompt)

        negative_prompt_embeds, negative_prompt_attention_mask = get_qwen2_prompt_embeds(
            text_encoder=text_encoder,
            tokenizer=text_tokenizer,
            prompt=negative_prompt,
            device=accelerator.device,
            max_sequence_length=1024,
        )

        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, args.train.rl.batch_size_per_forward, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(args.train.rl.batch_size_per_forward, seq_len, -1)
        negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(args.train.rl.batch_size_per_forward, 1)
        negative_prompt_attention_mask = negative_prompt_attention_mask.view(
            args.train.rl.batch_size_per_forward, -1
        )

        ref_latents_N, ref_img_mask_N, l_effective_ref_img_len_N, ref_img_sizes_N = pipeline.transformer.flat_and_pad_to_seq_ref_img(None, args.train.rl.batch_size_per_forward, weight_dtype, accelerator.device)

    reward_server_config = OmegaConf.load(args.reward_server_config)
    
    for epoch in range(first_epoch, args.train.num_train_epochs):
        if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
            break
        train_dataloader.batch_sampler.batch_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            instruction = batch['instruction']
            input_images = batch['input_images']
            input_images_pil = batch['input_images_pil']
            target_img_size = batch['target_img_size']
            text_mask = batch['text_mask']
            text_input_ids = batch['text_ids']

            total_results = None
            total_text_feats = None

            batch_size_per_forward = args.train.rl.batch_size_per_forward
            for i in range(args.train.batch_size // batch_size_per_forward):
                with torch.no_grad():
                    text_feats = text_encoder(
                        input_ids=text_input_ids[i*batch_size_per_forward:(i+1)*batch_size_per_forward],
                        attention_mask=text_mask[i*batch_size_per_forward:(i+1)*batch_size_per_forward],
                        output_hidden_states=False,
                        ).last_hidden_state
                    
                    results = pipeline(
                        prompt_embeds=text_feats,
                        prompt_attention_mask=text_mask[i*batch_size_per_forward:(i+1)*batch_size_per_forward],
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_prompt_attention_mask=negative_prompt_attention_mask,
                        input_images=input_images[i*batch_size_per_forward:(i+1)*batch_size_per_forward],
                        size=target_img_size[i*batch_size_per_forward:(i+1)*batch_size_per_forward],
                        num_inference_steps=args.train.rl.num_inference_step,
                        max_sequence_length=1024,
                        text_guidance_scale=args.train.rl.text_guidance_scale,
                        image_guidance_scale=args.train.rl.image_guidance_scale,
                        cfg_range=(args.train.rl.cfg_range_start, args.train.rl.cfg_range_end),
                        num_images_per_prompt=1,
                        output_type="pil",
                        enable_parallel_cfg=True,
                        return_middle_statistics=True,
                        mixed_precision=True,
                        do_normalize=False
                    )

                    if i == 0:
                        total_text_feats = text_feats
                        total_results = results
                        for k in ['img_mask', 'ref_latents', 'ref_img_mask', 'middle_latents']:
                            total_results.__dict__[k] = [total_results.__dict__[k]]
                    else:
                        total_text_feats = torch.cat([total_text_feats, text_feats], dim=0)

                        for k in ['images', 'l_effective_img_len', 'img_sizes', 'l_effective_ref_img_len', 'ref_img_sizes']:
                            total_results.__dict__[k].extend(results.__dict__[k])

                        for k in ['img_mask', 'ref_latents', 'ref_img_mask', 'middle_latents']:
                            total_results.__dict__[k].append(results.__dict__[k])

                        for i in range(len(results.log_probs)):
                            total_results.log_probs[i] = torch.cat([total_results.log_probs[i], results.log_probs[i]], dim=0)

            for i in range(len(batch['meta_data'])):
                json_data = json.loads(batch['meta_data'][i])
                json_data['id'] = f"{global_step * args.train.global_batch_size + accelerator.process_index * args.train.batch_size + i}"
                batch['meta_data'][i] = json.dumps(json_data)
            
            local_batch_size = len(input_images_pil)
            gathered_input_images_pil = gather_object(input_images_pil)
            gathered_output_images = gather_object(total_results.images)
            gathered_meta_data = gather_object(batch['meta_data'])

            if accelerator.is_main_process:
                scores, rewards, reasoning, meta_data = evaluate_images(
                    input_images=gathered_input_images_pil,
                    output_image=gathered_output_images,
                    meta_datas=gathered_meta_data,
                    proxy_host=reward_server_config.server.hosts[0],
                    proxy_port=reward_server_config.server.proxy_port,
                    server_type=args.train.rl.get('server_type', 'vlm')
                )

                rewards_to_scatter = [rewards[i:i + local_batch_size] for i in range(0, len(rewards), local_batch_size)]
                reasoning_to_scatter = [reasoning[i:i + local_batch_size] for i in range(0, len(reasoning), local_batch_size)]
                meta_data_to_scatter = [meta_data[i:i + local_batch_size] for i in range(0, len(meta_data), local_batch_size)]
            else:
                rewards_to_scatter = [None for _ in range(accelerator.num_processes)]
                reasoning_to_scatter = [None for _ in range(accelerator.num_processes)]
                meta_data_to_scatter = [None for _ in range(accelerator.num_processes)]

            accelerator.wait_for_everyone()
            # Extract the current process’s own rewards, reasoning, and meta_data.
            rewards = [None]
            reasoning = [None]
            meta_data = [None]
            torch.distributed.scatter_object_list(rewards, rewards_to_scatter)
            torch.distributed.scatter_object_list(reasoning, reasoning_to_scatter)
            torch.distributed.scatter_object_list(meta_data, meta_data_to_scatter)
            rewards = rewards[0]
            reasoning = reasoning[0]
            meta_data = meta_data[0]

            assert len(rewards) == len(reasoning) == len(meta_data) == local_batch_size

            rewards = torch.tensor(rewards, dtype=torch.float32, device=accelerator.device)
        
            advantages, prompt_stats = process_grpo_rewards(
                rewards=rewards,
                prompts=instruction,
                accelerator=accelerator,
                std_level=args.train.rl.get('std_level', 'group'),
            )

            reuse_samples_nums = args.train.rl.reuse_samples_nums  # reuse times of samples 
            clip_range = args.train.rl.clip_range              # PPO clip range
            
            # prepare data for GRPO
            timesteps = pipeline.scheduler._timesteps  # [batch_size, num_timesteps+1] 

            assert reuse_samples_nums == 1
            
            for reuse_step in range(reuse_samples_nums):                    

                logs = defaultdict(list)
                for forward_step in range(args.train.batch_size // batch_size_per_forward):
                    results = FMPipelineOutput(
                        images=[],
                        middle_latents=[],
                        log_probs=[],
                        img_mask=[],
                        l_effective_img_len=[],
                        img_sizes=[],
                        ref_latents=[],
                        ref_img_mask=[],
                        l_effective_ref_img_len=[],
                        ref_img_sizes=[],
                    )

                    for k in ['images', 'l_effective_img_len', 'img_sizes', 'l_effective_ref_img_len', 'ref_img_sizes']:
                        results.__dict__[k] = total_results.__dict__[k][forward_step*batch_size_per_forward:(forward_step+1)*batch_size_per_forward]
                    
                    for k in ['img_mask', 'ref_latents', 'ref_img_mask', 'middle_latents']:
                        results.__dict__[k] = total_results.__dict__[k][forward_step]

                    results.log_probs = [total_results.log_probs[i][forward_step*batch_size_per_forward:(forward_step+1)*batch_size_per_forward] for i in range(len(total_results.log_probs))]

                    text_feats = total_text_feats[forward_step*batch_size_per_forward:(forward_step+1)*batch_size_per_forward]

                    old_log_probs = [total_results.log_probs[i][forward_step*batch_size_per_forward:(forward_step+1)*batch_size_per_forward] for i in range(len(total_results.log_probs))]

                    train_timesteps = list(range(args.train.rl.num_inference_step))
                    sample_steps = math.ceil(args.train.rl.num_inference_step * args.train.rl.get('train_timesteps_fraction', 1.0))
                    train_timesteps = sorted(random.sample(train_timesteps, k=sample_steps))
                    
                    if args.train.rl.policy_loss_reweighting:
                        sigma_ts = []
                        for i in train_timesteps:
                            t = timesteps[0, i]
                            t_next = timesteps[0, i+1]

                            sigma_t = pipeline.scheduler.get_sigma_t(t, t_next if i == 0 else None)  # [batch_size]
                            dt = t_next - t
                            sigma_ts.append(sigma_t * math.sqrt(dt))
                        
                        sigma_ts = torch.stack(sigma_ts)
                        normalize_factor = sigma_ts.mean()

                    for idx, i in enumerate(train_timesteps):
                        with accelerator.accumulate(model):
                            text_guidance_scale = args.train.rl.text_guidance_scale if args.train.rl.cfg_range_start <= i / args.train.rl.num_inference_step <= args.train.rl.cfg_range_end else 1.0
                            image_guidance_scale = args.train.rl.image_guidance_scale if args.train.rl.cfg_range_start <= i / args.train.rl.num_inference_step <= args.train.rl.cfg_range_end else 1.0
                            
                            latents = results.middle_latents[i]
                            latents_next = results.middle_latents[i+1]
                            t = timesteps[:, i]
                            t_next = timesteps[:, i+1]

                            model_kwargs = dict(
                                hidden_states=latents,
                                timestep=t,
                                freqs_cis=freqs_cis,
                                flat_and_pad=False,
                                img_mask=results.img_mask,
                                l_effective_img_len=results.l_effective_img_len,
                                img_sizes=results.img_sizes,
                            )
                            model_pred_kwargs = dict(
                                text_hidden_states=text_feats,
                                text_attention_mask=text_mask[forward_step*batch_size_per_forward:(forward_step+1)*batch_size_per_forward],
                                ref_image_hidden_states=results.ref_latents,
                                ref_img_mask=results.ref_img_mask,
                                l_effective_ref_img_len=results.l_effective_ref_img_len,
                                ref_img_sizes=results.ref_img_sizes,
                            )

                            if image_guidance_scale > 1 and text_guidance_scale > 1:
                                model_kwargs['hidden_states'] = torch.cat([latents, latents, latents], dim=0)
                                model_kwargs['timestep'] = torch.cat([t, t, t], dim=0)
                                model_kwargs['img_mask'] = torch.cat([results.img_mask, results.img_mask, results.img_mask], dim=0)
                                model_kwargs['l_effective_img_len'] = results.l_effective_img_len * 3
                                model_kwargs['img_sizes'] = results.img_sizes * 3
                                
                                model_pred_kwargs['text_hidden_states'] = torch.cat([text_feats, pad_to_length(negative_prompt_embeds, len=text_feats.shape[1]), pad_to_length(negative_prompt_embeds, len=text_feats.shape[1])], dim=0)
                                model_pred_kwargs['text_attention_mask'] = torch.cat([text_mask[forward_step*batch_size_per_forward:(forward_step+1)*batch_size_per_forward], pad_to_length(negative_prompt_attention_mask, len=text_mask.shape[1]), pad_to_length(negative_prompt_attention_mask, len=text_mask.shape[1])], dim=0)
                                model_pred_kwargs['ref_image_hidden_states'] = torch.cat([results.ref_latents, results.ref_latents, pad_to_length(ref_latents_N, len=results.ref_latents.shape[1])], dim=0)
                                model_pred_kwargs['ref_img_mask'] = torch.cat([results.ref_img_mask, results.ref_img_mask, pad_to_length(ref_img_mask_N, len=results.ref_img_mask.shape[1])], dim=0)
                                model_pred_kwargs['l_effective_ref_img_len'] = results.l_effective_ref_img_len * 2 + l_effective_ref_img_len_N
                                model_pred_kwargs['ref_img_sizes'] = results.ref_img_sizes * 2 + ref_img_sizes_N

                            elif text_guidance_scale > 1:
                                model_kwargs['hidden_states'] = torch.cat([latents, latents], dim=0)
                                model_kwargs['timestep'] = torch.cat([t, t], dim=0)
                                model_kwargs['img_mask'] = torch.cat([results.img_mask, results.img_mask], dim=0)
                                model_kwargs['l_effective_img_len'] = results.l_effective_img_len * 2
                                model_kwargs['img_sizes'] = results.img_sizes * 2
                                
                                model_pred_kwargs['text_hidden_states'] = torch.cat([text_feats, pad_to_length(negative_prompt_embeds, len=text_feats.shape[1])], dim=0)
                                model_pred_kwargs['text_attention_mask'] = torch.cat([text_mask[forward_step*batch_size_per_forward:(forward_step+1)*batch_size_per_forward], pad_to_length(negative_prompt_attention_mask, len=text_mask.shape[1])], dim=0)
                                model_pred_kwargs['ref_image_hidden_states'] = torch.cat([results.ref_latents, pad_to_length(ref_latents_N, len=results.ref_latents.shape[1])], dim=0)
                                model_pred_kwargs['ref_img_mask'] = torch.cat([results.ref_img_mask, pad_to_length(ref_img_mask_N, len=results.ref_img_mask.shape[1])], dim=0)
                                model_pred_kwargs['l_effective_ref_img_len'] = results.l_effective_ref_img_len + l_effective_ref_img_len_N
                                model_pred_kwargs['ref_img_sizes'] = results.ref_img_sizes + ref_img_sizes_N

                            step_log_probs, mean_t, sigma_t = forward_logprob(
                                latents=latents,
                                latents_next=latents_next,
                                t=t,
                                t_next=t_next,
                                step_index=i,
                                img_mask=results.img_mask,
                                model=model,
                                model_kwargs=model_kwargs,
                                model_pred_kwargs=model_pred_kwargs,
                                scheduler=pipeline.scheduler,
                                apply_cfg=args.train.rl.apply_cfg_in_training,
                                text_guidance_scale=text_guidance_scale,
                                image_guidance_scale=image_guidance_scale,
                            )

                            if args.train.rl.kl_loss_weight > 0:
                                with torch.no_grad():
                                    with disabled_adapters(unwrap_model(model)):
                                        _, mean_t_ref, _ = forward_logprob(
                                            latents=latents,
                                            latents_next=latents_next,
                                            t=t,
                                            t_next=t_next,
                                            step_index=i,
                                            img_mask=results.img_mask,
                                            model=model,
                                            model_kwargs=model_kwargs,
                                            model_pred_kwargs=model_pred_kwargs,
                                            scheduler=pipeline.scheduler,
                                            apply_cfg=args.train.rl.apply_cfg_in_training,
                                            text_guidance_scale=text_guidance_scale,
                                            image_guidance_scale=image_guidance_scale,
                                        )
                            
                            loss = 0
                          
                            (
                                policy_loss,
                                policy_clip_frac,
                                approx_kl,
                                unclipped_loss,
                                clipped_loss,
                                ratio,
                                ratio_positive,
                                ratio_negative,
                                num_positive,
                                num_negative,
                                ratio_large_than_1,
                                ratio_small_than_1,
                            ) = compute_single_step_ppo_loss(
                                step_log_probs=step_log_probs,
                                old_step_log_probs=old_log_probs[i],
                                advantages=advantages[
                                    forward_step * batch_size_per_forward : (
                                        forward_step + 1
                                    )
                                    * batch_size_per_forward
                                ],
                                clip_range=clip_range,
                                adv_clip_max=args.train.rl.adv_clip_max,
                            )
                            logs['policy_loss'].append(policy_loss.detach())
                            logs['policy_clip_frac'].append(policy_clip_frac.detach())
                            logs['approx_kl'].append(approx_kl.detach())
                            logs['advantages'].append(advantages[forward_step*batch_size_per_forward:(forward_step+1)*batch_size_per_forward].mean().detach())
                            logs['advantages_std'].append(advantages[forward_step*batch_size_per_forward:(forward_step+1)*batch_size_per_forward].std().detach())
                            logs['advantages_min'].append(advantages[forward_step*batch_size_per_forward:(forward_step+1)*batch_size_per_forward].min().detach())
                            logs['advantages_max'].append(advantages[forward_step*batch_size_per_forward:(forward_step+1)*batch_size_per_forward].max().detach())
                            logs['policy_loss_unclipped'].append(unclipped_loss.mean().detach())
                            logs['policy_loss_clipped'].append(clipped_loss.mean().detach())
                            logs['ratio'].append(ratio.mean().detach())
                            logs['ratio_positive'].append(ratio_positive.detach())
                            logs['ratio_negative'].append(ratio_negative.detach())
                            logs['num_positive'].append(num_positive.detach())
                            logs['num_negative'].append(num_negative.detach())
                            logs['ratio_large_than_1'].append(ratio_large_than_1.detach())
                            logs['ratio_small_than_1'].append(ratio_small_than_1.detach())

                            if args.train.rl.policy_loss_reweighting:
                                policy_loss = policy_loss * sample_steps * (sigma_ts[idx] / normalize_factor)

                            loss += policy_loss

                            if args.train.rl.kl_loss_weight > 0:
                                
                                kl_loss = torch.mean(
                                    ((mean_t - mean_t_ref.detach()) ** 2)
                                    .flatten(start_dim=1)
                                    .mean(dim=1)
                                    / (2 * sigma_t)
                                )
                                logs['kl_loss'].append(kl_loss.detach())
                                
                                loss += args.train.rl.kl_loss_weight * kl_loss

                            logs['loss'].append(loss.detach())
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)
                                logs['grad_norm'].append(grad_norm.to(accelerator.device).detach())
                            optimizer.step()
                            if 'timm' in args.train.lr_scheduler:
                                lr_scheduler.step(global_step)
                            else:
                                lr_scheduler.step()
                            optimizer.zero_grad(set_to_none=args.train.set_grads_to_none)
                    
                    # Checks if the accelerator has performed an optimization step behind the scenes

                    if accelerator.sync_gradients:

                        logs = {k: torch.mean(torch.stack(v)) for k, v in logs.items()}
                        logs = accelerator.reduce(logs, reduction="mean")
                        logs = {k: v.item() for k, v in logs.items()}
                        logs.update(
                            {
                                "lr": lr_scheduler.get_last_lr()[0],
                                "rewards_min": np.mean(
                                    [v["min"] for k, v in prompt_stats.items()]
                                ),
                                "rewards_max": np.mean(
                                    [v["max"] for k, v in prompt_stats.items()]
                                ),
                                "rewards_mean": np.mean(
                                    [v["mean"] for k, v in prompt_stats.items()]
                                ),
                                "rewards_std": np.mean(
                                    [v["std"] for k, v in prompt_stats.items()]
                                ),
                                "zero_std_ratio": np.mean(
                                    [
                                        v["std"] == 0 for k, v in prompt_stats.items()
                                    ]
                                )
                            }
                        )

                        if ema_decay != 0:
                            model_ema.step(model.parameters())
                            
                        global_step += 1

                        if global_step % args.logger.checkpointing_steps == 0:
                            if accelerator.is_main_process:
                                if args.logger.checkpoints_total_limit is not None:
                                    checkpoints = os.listdir(args.output_dir)
                                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                                    
                                    if len(checkpoints) >= args.logger.checkpoints_total_limit:
                                        num_to_remove = len(checkpoints) - args.logger.checkpoints_total_limit + 1
                                        removing_checkpoints = checkpoints[0:num_to_remove]

                                        logger.info(
                                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                        )
                                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                        for removing_checkpoint in removing_checkpoints:
                                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                            shutil.rmtree(removing_checkpoint)
                                                
                            accelerator.wait_for_everyone()
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                            
                        if 'train_visualization_interval' in args.val and (global_step - 1) % args.val.train_visualization_interval == 0:
                            num_samples = min(args.val.get('num_train_visualization_samples', 2), args.train.batch_size)

                            if accelerator.is_main_process:
                                target_instruction = instruction[:num_samples]
                            else:
                                target_instruction = [None] * num_samples
                            torch.distributed.broadcast_object_list(target_instruction)

                            rewards_per_instruction = [[] for _ in range(num_samples)]
                            for i in range(len(target_instruction)):
                                for j in range(len(instruction)):
                                    if target_instruction[i] == instruction[j]:
                                        rewards_per_instruction[i].append((rewards[j].item(), accelerator.process_index, j))
                            gathered_rewards_per_instruction = [None for _ in range(accelerator.num_processes)]
                            torch.distributed.all_gather_object(gathered_rewards_per_instruction, rewards_per_instruction)

                            gathered_rewards_per_instruction_flat = [[] for _ in range(num_samples)]
                            for i in range(num_samples):
                                for j in range(accelerator.num_processes):
                                    gathered_rewards_per_instruction_flat[i].extend(gathered_rewards_per_instruction[j][i])

                            p = [{} for _ in range(num_samples)]
                            for i in range(len(target_instruction)):
                                gathered_rewards_per_instruction_flat[i].sort(key=lambda x: x[0])
                                for j in range(len(gathered_rewards_per_instruction_flat[i])):
                                    if gathered_rewards_per_instruction_flat[i][j][1] == accelerator.process_index:
                                        p[i][gathered_rewards_per_instruction_flat[i][j][2]] = j

                            with torch.no_grad():
                                for i in range(len(target_instruction)):
                                    cnt = 0
                                    for j in range(len(instruction)):
                                        if instruction[j] == target_instruction[i]:
                                            if cnt == 0:
                                                for k in range(len(input_images_pil[j])):
                                                    input_images_pil[j][k].save(os.path.join(args.output_dir, f"input_visualization_{global_step}_{i}_input_{k}.png"))

                                            total_results.images[j].save(os.path.join(args.output_dir, f"input_visualization_{global_step}_{i}_{p[i][j]}.png"))
                                            with open(os.path.join(args.output_dir, f"instruction_{global_step}_{i}_{p[i][j]}.txt"), "w", encoding='utf-8') as f:
                                                f.write(f"instruction: {instruction[j]}\nreward: {rewards[j].item()}\nadvantages: {advantages[j].item()}\nreasoning: {reasoning[j]}\nmeta_data: {batch['meta_data'][j]}\ncur_receive_meta_data: {meta_data[j]}")
                                            cnt += 1
                                
                        progress_bar.set_postfix(**logs)
                        progress_bar.update(1)

                        accelerator.log(logs, step=global_step)
                        logs = defaultdict(list)

                    if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
                        break

    checkpoints = os.listdir(args.output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    if len(checkpoints) > 0 and int(checkpoints[-1].split("-")[1]) < global_step:
        if accelerator.is_main_process:
            if args.logger.checkpoints_total_limit is not None:
                if len(checkpoints) >= args.logger.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.logger.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]

                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)

        accelerator.wait_for_everyone()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")


    accelerator.end_training()

if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args(root_path)
    main(args)