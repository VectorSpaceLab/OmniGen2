from typing import List, Dict, Any, Tuple
from collections import defaultdict

import math
import numpy as np
import torch

from accelerate.utils import gather_object

def expand_as(tensor, other):
    """
    Expands a tensor to match the dimensions of another tensor.
    
    If tensor has shape [b] and other has shape [b, c, h, w],
    this function will reshape tensor to [b, 1, 1, 1] to enable broadcasting.
    
    Args:
        tensor (`torch.FloatTensor`): The tensor to expand
        other (`torch.FloatTensor`): The tensor whose shape will be matched
        
    Returns:
        `torch.FloatTensor`: The expanded tensor
    """
    for _ in range(other.ndim - tensor.ndim):
        tensor = tensor.unsqueeze(-1)
    return tensor


def process_grpo_rewards(
    rewards: torch.Tensor,
    prompts: List[str],
    accelerator,
    std_level: str = 'group'
) -> Tuple[torch.Tensor, Dict]:  # Modified return type
    """
    Process GRPO rewards and compute advantages
    Only use group statistics from the same prompt in the current batch
    
    Args:
        rewards: rewards of current batch [batch_size]
        prompts: prompts of current batch [batch_size]
        accelerator: distributed training accelerator
        std_level: 'group' or 'batch'
        
    Returns:
        advantages: computed advantages [batch_size]
        prompt_stats: statistical information for each prompt
    """
    # 1. Gather rewards, text_ids and mask from all processes
    gathered_rewards = accelerator.gather(rewards)  # [world_size * batch_size]
    gathered_prompts = gather_object(prompts) # [world_size * batch_size]

    assert len(gathered_rewards) == len(gathered_prompts), f"{len(gathered_rewards)=} {len(gathered_prompts)=}"
    
    # 3. Group rewards by prompt
    prompt_to_rewards = defaultdict(list)
    for prompt, reward in zip(gathered_prompts, gathered_rewards):
        prompt_to_rewards[prompt].append(reward.item())
    
    # 4. Pre-compute statistical information for each prompt group
    prompt_stats = {}
    for prompt, group_rewards in prompt_to_rewards.items():
        prompt_stats[prompt] = {
            "min": np.min(group_rewards),
            "max": np.max(group_rewards),
            "mean": np.mean(group_rewards),
            "std": np.std(group_rewards),
        }

    if std_level == 'batch':
        batch_std = np.std([reward.item() for reward in gathered_rewards])

    assert set(prompts).issubset(set(prompt_stats.keys())), f"{set(prompts)=} {set(prompt_stats.keys())=}"
    
    advantages = torch.zeros_like(rewards, device=rewards.device)
    for i, prompt in enumerate(prompts):
        stats = prompt_stats[prompt]
        if std_level == 'group':
            advantage = (rewards[i].item() - stats['mean']) / (stats['std'] + 1e-8)
        elif std_level == 'batch':
            advantage = (rewards[i].item() - stats['mean']) / (batch_std + 1e-8)
        advantages[i] = torch.tensor(advantage, device=rewards.device)

    return advantages, prompt_stats


def forward_logprob(
    latents: List[torch.Tensor],
    latents_next: List[torch.Tensor],
    t: torch.Tensor,
    t_next: torch.Tensor,
    step_index: int,
    img_mask,
    model,
    model_kwargs: Dict[str, Any],
    model_pred_kwargs: Dict[str, Any],
    # model_pred_ref_kwargs: Dict[str, Any],
    # model_pred_uncond_kwargs: Dict[str, Any],
    scheduler,
    apply_cfg: bool = True,
    text_guidance_scale: float = 1.0,
    image_guidance_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # cfg
    
    model_pred = model(**model_kwargs, **model_pred_kwargs)
    if apply_cfg:
        if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
            model_pred, model_pred_ref, model_pred_uncond = model_pred.chunk(3)
            model_pred = (
                model_pred_uncond
                + image_guidance_scale * (model_pred_ref - model_pred_uncond)
                + text_guidance_scale * (model_pred - model_pred_ref)
            )
        elif text_guidance_scale > 1.0:
            model_pred, model_pred_uncond = model_pred.chunk(2)
            model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)


    sigma_t = scheduler.get_sigma_t(t, t_next if step_index == 0 else None)  # [batch_size]
    sigma_t = expand_as(sigma_t.unsqueeze(1), latents)  # [batch_size, max_img_len, dim]
    t = expand_as(t.unsqueeze(1), latents)  # [batch_size, max_img_len, dim]
    t_next = expand_as(t_next.unsqueeze(1), latents)  # [batch_size, max_img_len, dim]
    dt = t_next - t
    
    sigma_t = sigma_t.to(dtype=torch.float32)
    t = t.to(dtype=torch.float32)
    t_next = t_next.to(dtype=torch.float32)
    dt = dt.to(dtype=torch.float32)

    prev_sample_mean = (
        latents.to(dtype=torch.float32) * (1 - sigma_t**2 / (2 * (1 - t)) * dt)
        + model_pred * (1 + sigma_t**2 * t / (2 * (1 - t))) * dt
    )

    log_prob = (
        -((latents_next.to(dtype=torch.float32).detach() - prev_sample_mean) ** 2)
        / (2 * (sigma_t**2 * dt))  # Fix: denominator is 2 * σ² * dt
        - torch.log(sigma_t * torch.sqrt(dt))  # Fix: log(σ * √dt)
        - 0.5
        * torch.log(
            2 * torch.as_tensor(math.pi, device=latents.device)
        )  # Fix: 0.5 coefficient
    )

    img_mask = expand_as(img_mask, latents).expand(latents.shape)
    log_prob = (log_prob * img_mask.detach()).sum(
        dim=tuple(range(-log_prob.ndim + 1, 0)), dtype=torch.float32
    ) / img_mask.detach().sum(
        dim=tuple(range(-log_prob.ndim + 1, 0)), dtype=torch.float32
    )
   
    return log_prob, prev_sample_mean, sigma_t**2


def compute_single_step_ppo_loss(
    step_log_probs: torch.Tensor,      # [batch_size] log probability of current time step  
    old_step_log_probs: torch.Tensor,  # [batch_size] log probability of old policy
    advantages: torch.Tensor,          # [batch_size] advantages
    clip_range: Tuple[float, float] = (1e-4, 1e-4),             # PPO clipping range
    adv_clip_max: float = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PPO loss for a single time step
    
    Args:
        step_log_probs: log probability of current policy at this time step [batch_size]
        old_step_log_probs: log probability of old policy at this time step [batch_size] 
        advantages: advantages [batch_size]
        clip_range: PPO clipping range
        
    Returns:
        step_pg_loss: policy loss at this time step (scalar)
        pg_clipfrac_step: fraction of clipped samples (scalar)
        approx_kl_step: approximate KL divergence (scalar)
    """
    # Calculate ratio
    log_ratio = step_log_probs - old_step_log_probs.detach()
    approx_kl_step = (-log_ratio).mean()
    
    # Calculate two types of loss: original and clipped
    ratio = torch.exp(log_ratio)
    advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
    unclipped_loss = -advantages.detach() * ratio  # [batch_size]
    clipped_loss = -advantages.detach() * torch.clamp(
        ratio, 1.0 - clip_range[0], 1.0 + clip_range[1]
    )  # [batch_size]

    # Take maximum value (more conservative loss)
    step_pg_loss = torch.max(unclipped_loss, clipped_loss).mean()


    # Calculate the fraction of clipped samples
    pg_clipfrac_step = (clipped_loss > unclipped_loss).float().mean()

    num_positive = torch.where(advantages > 0, torch.ones_like(ratio), torch.zeros_like(ratio)).sum()
    num_negative = torch.where(advantages < 0, torch.ones_like(ratio), torch.zeros_like(ratio)).sum()
    ratio_positive = torch.where(advantages > 0, ratio, torch.zeros_like(ratio)).sum() / num_positive if num_positive > 0 else torch.tensor(0.0, device=ratio.device)
    ratio_negative = torch.where(advantages < 0, ratio, torch.zeros_like(ratio)).sum() / num_negative if num_negative > 0 else torch.tensor(0.0, device=ratio.device)

    ratio_large_than_1 = torch.where(ratio > 1, torch.ones_like(ratio), torch.zeros_like(ratio)).sum() / len(ratio)
    ratio_small_than_1 = torch.where(ratio < 1, torch.ones_like(ratio), torch.zeros_like(ratio)).sum() / len(ratio)

    return step_pg_loss, pg_clipfrac_step, approx_kl_step, unclipped_loss, clipped_loss, ratio, ratio_positive, ratio_negative, num_positive, num_negative, ratio_large_than_1, ratio_small_than_1


