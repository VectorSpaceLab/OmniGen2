"""
OmniGen2 Diffusion Pipeline

Copyright 2025 BAAI, The OmniGen2 Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from transformers import Qwen2_5_VLForConditionalGeneration

from diffusers.models.autoencoders import AutoencoderKL
from ...models.transformers import OmniGen2Transformer2DModel
from ...models.transformers.repo import OmniGen2RotaryPosEmbed
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from dataclasses import dataclass

from einops import rearrange

import PIL.Image

from diffusers.utils import BaseOutput

from omnigen2.pipelines.image_processor import OmniGen2ImageProcessor
from ..lora_pipeline import OmniGen2LoraLoaderMixin
from ...utils.tensor_util import pad_to_length

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class FMPipelineOutput(BaseOutput):
    """
    Output class for OmniGen2 pipeline.

    Args:
        images (Union[List[PIL.Image.Image], np.ndarray]): 
            List of denoised PIL images of length `batch_size` or numpy array of shape 
            `(batch_size, height, width, num_channels)`. Contains the generated images.
    """
    images: Union[List[PIL.Image.Image], np.ndarray]
    middle_latents: Optional[List[torch.FloatTensor]] = None
    log_probs: Optional[List[torch.FloatTensor]] = None
    img_mask: Optional[torch.FloatTensor] = None
    l_effective_img_len: Optional[List[int]] = None
    img_sizes: Optional[List[Tuple[int, int]]] = None
    ref_latents: Optional[torch.FloatTensor] = None
    ref_img_mask: Optional[torch.FloatTensor] = None
    l_effective_ref_img_len: Optional[List[int]] = None
    ref_img_sizes: Optional[List[Tuple[int, int]]] = None

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class OmniGen2Pipeline(DiffusionPipeline, OmniGen2LoraLoaderMixin):
    """
    Pipeline for text-to-image generation using OmniGen2.

    This pipeline implements a text-to-image generation model that uses:
    - Qwen2.5-VL for text encoding
    - A custom transformer architecture for image generation
    - VAE for image encoding/decoding
    - FlowMatchEulerDiscreteScheduler for noise scheduling

    Args:
        transformer (OmniGen2Transformer2DModel): The transformer model for image generation.
        vae (AutoencoderKL): The VAE model for image encoding/decoding.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for noise scheduling.
        text_encoder (Qwen2_5_VLModel): The text encoder model.
        tokenizer (Union[Qwen2Tokenizer, Qwen2TokenizerFast]): The tokenizer for text processing.
    """

    model_cpu_offload_seq = "mllm->transformer->vae"

    def __init__(
        self,
        transformer: OmniGen2Transformer2DModel,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler,
        mllm: Qwen2_5_VLForConditionalGeneration,
        processor,
    ) -> None:
        """
        Initialize the OmniGen2 pipeline.

        Args:
            transformer: The transformer model for image generation.
            vae: The VAE model for image encoding/decoding.
            scheduler: The scheduler for noise scheduling.
            text_encoder: The text encoder model.
            tokenizer: The tokenizer for text processing.
        """
        super().__init__()

        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm=mllm,
            processor=processor
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_resize=True)
        self.default_sample_size = 128

    def prepare_latents(
        self,
        # batch_size: int,
        num_channels_latents: int,
        size: List[Tuple[int, int]],
        num_images_per_prompt: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        latents: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Prepare the initial latents for the diffusion process.

        Args:
            batch_size: The number of images to generate.
            num_channels_latents: The number of channels in the latent space.
            height: The height of the generated image.
            width: The width of the generated image.
            dtype: The data type of the latents.
            device: The device to place the latents on.
            generator: The random number generator to use.
            latents: Optional pre-computed latents to use instead of random initialization.

        Returns:
            torch.FloatTensor: The prepared latents tensor.
        """
        if latents is None:
            latents = []
            for _size in size:
                width = int(_size[0]) // self.vae_scale_factor
                height = int(_size[1]) // self.vae_scale_factor

                shape = (num_images_per_prompt, num_channels_latents, height, width)
                latent = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                for i in range(num_images_per_prompt):
                    latents.append(latent[i])
        else:
            # for i in range(num_images_per_prompt):
            for i in range(len(latents)):
                latents[i] = latents[i].to(device)
        return latents

    def encode_vae(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """
        Encode an image into the VAE latent space.

        Args:
            img: The input image tensor to encode.

        Returns:
            torch.FloatTensor: The encoded latent representation.
        """
        z0 = self.vae.encode(img.to(dtype=self.vae.dtype)).latent_dist.sample()
        if self.vae.config.shift_factor is not None:
            z0 = z0 - self.vae.config.shift_factor
        if self.vae.config.scaling_factor is not None:
            z0 = z0 * self.vae.config.scaling_factor
        z0 = z0.to(dtype=self.vae.dtype)
        return z0

    def prepare_image(
        self,
        images: Union[List[PIL.Image.Image], PIL.Image.Image],
        batch_size: int,
        num_images_per_prompt: int,
        max_pixels: int,
        max_side_length: int,
        do_normalize: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Optional[torch.FloatTensor]]:
        """
        Prepare input images for processing by encoding them into the VAE latent space.

        Args:
            images: Single image or list of images to process.
            batch_size: The number of images to generate per prompt.
            num_images_per_prompt: The number of images to generate for each prompt.
            device: The device to place the encoded latents on.
            dtype: The data type of the encoded latents.

        Returns:
            List[Optional[torch.FloatTensor]]: List of encoded latent representations for each image.
        """
        latents = []
        for i, img in enumerate(images):
            if img is not None and len(img) > 0:
                ref_latents = []
                for j, img_j in enumerate(img):
                    img_j = self.image_processor.preprocess(img_j, max_pixels=max_pixels, max_side_length=max_side_length, do_normalize=do_normalize)
                    ref_latents.append(self.encode_vae(img_j.to(device=device)).squeeze(0))
            else:
                ref_latents = None
            latents.append(ref_latents)

        latents = [latent for latent in latents for _ in range(num_images_per_prompt)]
        return latents
    
    def _get_qwen2_prompt_embeds(
        self,
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
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        
        text_inputs = self.processor.tokenizer(
            prompt,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        untruncated_ids = self.processor.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.to(device)

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.processor.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because Gemma can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = self.mllm(
            text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]

        if self.mllm is not None:
            dtype = self.mllm.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask
    
    def _apply_chat_template(self, prompt: str):
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates high-quality images based on user instructions.",
            },
            {"role": "user", "content": prompt},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        return prompt

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                Lumina-T2I, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For Lumina-T2I, it's should be the embeddings of the "" string.
            max_sequence_length (`int`, defaults to `256`):
                Maximum sequence length to use for the prompt.
        """
        device = device or self._execution_device

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt = [prompt] if isinstance(prompt, str) else prompt
            prompt = [self._apply_chat_template(_prompt) for _prompt in prompt]

            prompt_embeds, prompt_attention_mask = self._get_qwen2_prompt_embeds(
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)

        # Get negative embeddings for classifier free guidance
        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                negative_prompt = negative_prompt if negative_prompt is not None else ""

                # Normalize str to list
                negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                negative_prompt = [self._apply_chat_template(_negative_prompt) for _negative_prompt in negative_prompt]

                if prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                negative_prompt_embeds, negative_prompt_attention_mask = self._get_qwen2_prompt_embeds(
                    prompt=negative_prompt,
                    device=device,
                    max_sequence_length=max_sequence_length,
                )

            batch_size, seq_len, _ = negative_prompt_embeds.shape
            # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                batch_size * num_images_per_prompt, -1
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
    
    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @property
    def text_guidance_scale(self):
        return self._text_guidance_scale
    
    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale
    
    @property
    def cfg_range(self):
        return self._cfg_range
    
    @property
    def enable_parallel_cfg(self):
        return self._enable_parallel_cfg
    
    @property
    def mixed_precision(self):
        return self._mixed_precision
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        negative_prompt_attention_mask: Optional[torch.LongTensor] = None,
        max_sequence_length: Optional[int] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        input_images: Optional[List[PIL.Image.Image]] = None,
        num_images_per_prompt: int = 1,
        size: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
        max_pixels: int = 1024 * 1024,
        max_input_image_side_length: int = 1024,
        align_res: bool = True,
        num_inference_steps: int = 28,
        text_guidance_scale: float = 4.0,
        image_guidance_scale: float = 1.0,
        cfg_range: Tuple[float, float] = (0.0, 1.0),
        enable_parallel_cfg: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_middle_statistics: bool = False,
        return_dict: bool = True,
        verbose: bool = False,
        step_func=None,
        mixed_precision: bool = False,
        do_normalize: Optional[bool] = None
    ):
        
        size = size or self.default_sample_size * self.vae_scale_factor

        self._text_guidance_scale = text_guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._cfg_range = cfg_range
        self._enable_parallel_cfg = enable_parallel_cfg
        self._attention_kwargs = attention_kwargs
        self._mixed_precision = mixed_precision
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if not isinstance(size, list):
            size = [size] * batch_size

        device = self._execution_device

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            self.text_guidance_scale > 1.0,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
        )

        dtype = self.vae.dtype
        # 3. Prepare control image
        ref_latents = self.prepare_image(
            images=input_images,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            max_pixels=max_pixels,
            max_side_length=max_input_image_side_length,
            do_normalize=do_normalize,
            device=device,
            dtype=dtype,
        )

        for i, _input_images in enumerate(input_images):
            if _input_images is None:
                input_images[i] = []

        ori_size = []
        for i, (_input_images, _size) in enumerate(zip(input_images, size)):
            if len(_input_images) == 1 and align_res:
                size[i] = (ref_latents[i][0].shape[-1] * self.vae_scale_factor, ref_latents[i][0].shape[-2] * self.vae_scale_factor)
                ori_size.append(size[i])
            else:
                ori_size.append(_size)

                cur_pixels = _size[0] * _size[1]
                ratio = (max_pixels / cur_pixels) ** 0.5
                ratio = min(ratio, 1.0)

                width, height = int(_size[0] * ratio) // 16 * 16, int(_size[1] * ratio) // 16 * 16
                size[i] = (width, height)
        
        # 4. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            latent_channels,
            size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            self.transformer.config.axes_dim_rope,
            self.transformer.config.axes_lens,
            theta=10000,
        )
        
        image = self.processing(
            latents=latents,
            ref_latents=ref_latents,
            prompt_embeds=prompt_embeds,
            freqs_cis=freqs_cis,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            generator=generator,
            device=device,
            dtype=dtype,
            verbose=verbose,
            step_func=step_func,
            return_middle_statistics=return_middle_statistics,
        )

        if return_middle_statistics:
            image, middle_latents, log_probs, img_mask, l_effective_img_len, img_sizes, ref_latents, ref_img_mask, l_effective_ref_img_len, ref_img_sizes = image
        
        ori_size = [_ori_size for _ori_size in ori_size for _ in range(num_images_per_prompt)]

        postprocessed_image = []
        for i, (_image, _ori_size) in enumerate(zip(image, ori_size)):
            width, height = _ori_size
            resized_image = F.interpolate(_image.unsqueeze(0), size=(height, width), mode='bilinear')
            postprocessed_image.append(self.image_processor.postprocess(resized_image, output_type=output_type)[0])
        
        image = postprocessed_image

        # Offload all models
        self.maybe_free_model_hooks()
        
        if return_middle_statistics:
            if not return_dict:
                return image, middle_latents, log_probs, img_mask, l_effective_img_len, img_sizes, ref_latents, ref_img_mask, l_effective_ref_img_len, ref_img_sizes
            else:
                return FMPipelineOutput(images=image, middle_latents=middle_latents, log_probs=log_probs, img_mask=img_mask, l_effective_img_len=l_effective_img_len, img_sizes=img_sizes, ref_latents=ref_latents, ref_img_mask=ref_img_mask, l_effective_ref_img_len=l_effective_ref_img_len, ref_img_sizes=ref_img_sizes)
        else:
            if not return_dict:
                return image
            else:
                return FMPipelineOutput(images=image)
            
    def _cfg_predict_sequential(self,
                                t,
                                prompt_embeds,
                                prompt_attention_mask,
                                negative_prompt_embeds,
                                negative_prompt_attention_mask,
                                freqs_cis,
                                latents,
                                img_mask,
                                l_effective_img_len,
                                img_sizes,
                                ref_latents,
                                ref_img_mask,
                                l_effective_ref_img_len,
                                ref_img_sizes,
                                ref_latents_N,
                                ref_img_mask_N,
                                l_effective_ref_img_len_N,
                                ref_img_sizes_N,
                                text_guidance_scale,
                                image_guidance_scale,
                                ):
        model_kwargs = dict(
            freqs_cis=freqs_cis,
            flat_and_pad=False,
            img_mask=img_mask,
            l_effective_img_len=l_effective_img_len,
            img_sizes=img_sizes,
        )
        model_pred_kwargs = dict(
            text_hidden_states=prompt_embeds,
            text_attention_mask=prompt_attention_mask,
            ref_image_hidden_states=ref_latents,
            ref_img_mask=ref_img_mask,
            l_effective_ref_img_len=l_effective_ref_img_len,
            ref_img_sizes=ref_img_sizes,
        )
        model_pred_ref_kwargs = dict(
            text_hidden_states=negative_prompt_embeds,
            text_attention_mask=negative_prompt_attention_mask,
            ref_image_hidden_states=ref_latents,
            ref_img_mask=ref_img_mask,
            l_effective_ref_img_len=l_effective_ref_img_len,
            ref_img_sizes=ref_img_sizes,
        )
        model_pred_uncond_kwargs = dict(
            text_hidden_states=negative_prompt_embeds,
            text_attention_mask=negative_prompt_attention_mask,
            ref_image_hidden_states=ref_latents_N,
            ref_img_mask=ref_img_mask_N,
            l_effective_ref_img_len=l_effective_ref_img_len_N,
            ref_img_sizes=ref_img_sizes_N,
        )

        model_pred = self.predict(
            t=t,
            latents=latents,
            **model_kwargs,
            **model_pred_kwargs,
        )
        if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
            model_pred_ref = self.predict(
                t=t,
                latents=latents,
                **model_kwargs,
                **model_pred_ref_kwargs,
            )
            model_pred_uncond = self.predict(
                t=t,
                latents=latents,
                **model_kwargs,
                **model_pred_uncond_kwargs,
            )
            model_pred = model_pred_uncond + image_guidance_scale * (model_pred_ref - model_pred_uncond) + \
                text_guidance_scale * (model_pred - model_pred_ref)

        elif text_guidance_scale > 1.0:
            model_pred_uncond = self.predict(
                t=t,
                latents=latents,
                **model_kwargs,
                **model_pred_uncond_kwargs,
            )
            model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)
        return model_pred
    
    def _cfg_predict_parallel(self,
                              t,
                              prompt_embeds,
                              prompt_attention_mask,
                              negative_prompt_embeds,
                              negative_prompt_attention_mask,
                              freqs_cis,
                              latents,
                              img_mask,
                              l_effective_img_len,
                              img_sizes,
                              ref_latents,
                              ref_img_mask,
                              l_effective_ref_img_len,
                              ref_img_sizes,
                              ref_latents_N,
                              ref_img_mask_N,
                              l_effective_ref_img_len_N,
                              ref_img_sizes_N,
                              text_guidance_scale,
                              image_guidance_scale,
                              ):
        model_kwargs = dict(
            freqs_cis=freqs_cis,
            flat_and_pad=False,
            img_mask=img_mask,
            l_effective_img_len=l_effective_img_len,
            img_sizes=img_sizes,
        )
        model_pred_kwargs = dict(
            text_hidden_states=prompt_embeds,
            text_attention_mask=prompt_attention_mask,
            ref_image_hidden_states=ref_latents,
            ref_img_mask=ref_img_mask,
            l_effective_ref_img_len=l_effective_ref_img_len,
            ref_img_sizes=ref_img_sizes,
        )

        if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
            latents = torch.cat([latents, latents, latents], dim=0)
            t = torch.cat([t, t, t], dim=0)
            model_kwargs['img_mask'] = torch.cat([img_mask, img_mask, img_mask], dim=0)
            model_kwargs['l_effective_img_len'] = l_effective_img_len * 3
            model_kwargs['img_sizes'] = img_sizes * 3
            
            model_pred_kwargs['text_hidden_states'] = torch.cat([prompt_embeds, pad_to_length(negative_prompt_embeds, len=prompt_embeds.shape[1]), pad_to_length(negative_prompt_embeds, len=prompt_embeds.shape[1])], dim=0)
            model_pred_kwargs['text_attention_mask'] = torch.cat([prompt_attention_mask, pad_to_length(negative_prompt_attention_mask, len=prompt_attention_mask.shape[1]), pad_to_length(negative_prompt_attention_mask, len=prompt_attention_mask.shape[1])], dim=0)
            model_pred_kwargs['ref_image_hidden_states'] = torch.cat([ref_latents, ref_latents, pad_to_length(ref_latents_N, len=ref_latents.shape[1])], dim=0)
            model_pred_kwargs['ref_img_mask'] = torch.cat([ref_img_mask, ref_img_mask, pad_to_length(ref_img_mask_N, len=ref_img_mask.shape[1])], dim=0)
            model_pred_kwargs['l_effective_ref_img_len'] = l_effective_ref_img_len * 2 + l_effective_ref_img_len_N
            model_pred_kwargs['ref_img_sizes'] = ref_img_sizes * 2 + ref_img_sizes_N

        elif text_guidance_scale > 1.0:
            latents = torch.cat([latents, latents], dim=0)
            t = torch.cat([t, t], dim=0)
            model_kwargs['img_mask'] = torch.cat([img_mask, img_mask], dim=0)
            model_kwargs['l_effective_img_len'] = l_effective_img_len * 2
            model_kwargs['img_sizes'] = img_sizes * 2
            
            model_pred_kwargs['text_hidden_states'] = torch.cat([prompt_embeds, pad_to_length(negative_prompt_embeds, len=prompt_embeds.shape[1])], dim=0)
            model_pred_kwargs['text_attention_mask'] = torch.cat([prompt_attention_mask, pad_to_length(negative_prompt_attention_mask, len=prompt_attention_mask.shape[1])], dim=0)
            model_pred_kwargs['ref_image_hidden_states'] = torch.cat([ref_latents, pad_to_length(ref_latents_N, len=ref_latents.shape[1])], dim=0)
            model_pred_kwargs['ref_img_mask'] = torch.cat([ref_img_mask, pad_to_length(ref_img_mask_N, len=ref_img_mask.shape[1])], dim=0)
            model_pred_kwargs['l_effective_ref_img_len'] = l_effective_ref_img_len + l_effective_ref_img_len_N
            model_pred_kwargs['ref_img_sizes'] = ref_img_sizes + ref_img_sizes_N

        model_pred = self.predict(
            t=t,
            latents=latents,
            **model_kwargs,
            **model_pred_kwargs,
        )
        
        if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
            model_pred, model_pred_ref, model_pred_uncond = model_pred.chunk(3)
            model_pred = model_pred_uncond + image_guidance_scale * (model_pred_ref - model_pred_uncond) + \
                text_guidance_scale * (model_pred - model_pred_ref)
        elif text_guidance_scale > 1.0:
            model_pred, model_pred_uncond = model_pred.chunk(2)
            model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)
        
        return model_pred

    def cfg_predict(self,
                    t,
                    prompt_embeds,
                    prompt_attention_mask,
                    negative_prompt_embeds,
                    negative_prompt_attention_mask,
                    freqs_cis,
                    latents,
                    img_mask,
                    l_effective_img_len,
                    img_sizes,
                    ref_latents,
                    ref_img_mask,
                    l_effective_ref_img_len,
                    ref_img_sizes,
                    ref_latents_N,
                    ref_img_mask_N,
                    l_effective_ref_img_len_N,
                    ref_img_sizes_N,
                    text_guidance_scale,
                    image_guidance_scale,):
        if self.enable_parallel_cfg:
            return self._cfg_predict_parallel(
                t=t,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                freqs_cis=freqs_cis,
                latents=latents,
                img_mask=img_mask,
                l_effective_img_len=l_effective_img_len,
                img_sizes=img_sizes,
                ref_latents=ref_latents,
                ref_img_mask=ref_img_mask,
                l_effective_ref_img_len=l_effective_ref_img_len,
                ref_img_sizes=ref_img_sizes,
                ref_latents_N=ref_latents_N,
                ref_img_mask_N=ref_img_mask_N,
                l_effective_ref_img_len_N=l_effective_ref_img_len_N,
                ref_img_sizes_N=ref_img_sizes_N,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
            )
        else:
            return self._cfg_predict_sequential(
                t=t,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                freqs_cis=freqs_cis,
                latents=latents,
                img_mask=img_mask,
                l_effective_img_len=l_effective_img_len,
                img_sizes=img_sizes,
                ref_latents=ref_latents,
                ref_img_mask=ref_img_mask,
                l_effective_ref_img_len=l_effective_ref_img_len,
                ref_img_sizes=ref_img_sizes,
                ref_latents_N=ref_latents_N,
                ref_img_mask_N=ref_img_mask_N,
                l_effective_ref_img_len_N=l_effective_ref_img_len_N,
                ref_img_sizes_N=ref_img_sizes_N,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
            )
        
    def processing(
        self,
        latents,
        ref_latents,
        prompt_embeds,
        freqs_cis,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
        num_inference_steps,
        timesteps,
        generator,
        device,
        dtype,
        verbose,
        step_func=None,
        return_middle_statistics=False,
    ):
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            num_tokens=[latent.shape[-2] * latent.shape[-1] for latent in latents]
        )
        num_warmup_steps = max(len(timesteps[0]) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps[0])
        
        batch_size = len(latents)
        (
            latents,
            img_mask,
            l_effective_img_len,
            img_sizes,
        ) = self.transformer.flat_and_pad_to_seq(latents, batch_size, device)

        (
            ref_latents,
            ref_img_mask,
            l_effective_ref_img_len,
            ref_img_sizes,
        ) = self.transformer.flat_and_pad_to_seq_ref_img(ref_latents, batch_size, dtype, device)
        (
            ref_latents_N,
            ref_img_mask_N,
            l_effective_ref_img_len_N,
            ref_img_sizes_N,
        ) = self.transformer.flat_and_pad_to_seq_ref_img(None, batch_size, dtype, device)

        if return_middle_statistics:
            latents_list = [latents]
            log_probs_list = []
            
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(len(timesteps[0])):
                t = timesteps[:, i]
                text_guidance_scale = self.text_guidance_scale if self.cfg_range[0] <= i / len(timesteps[0]) <= self.cfg_range[1] else 1.0
                image_guidance_scale = self.image_guidance_scale if self.cfg_range[0] <= i / len(timesteps[0]) <= self.cfg_range[1] else 1.0

                model_pred = self.cfg_predict(
                    t=t,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                    freqs_cis=freqs_cis,
                    latents=latents,
                    img_mask=img_mask,
                    l_effective_img_len=l_effective_img_len,
                    img_sizes=img_sizes,
                    ref_latents=ref_latents,
                    ref_img_mask=ref_img_mask,
                    l_effective_ref_img_len=l_effective_ref_img_len,
                    ref_img_sizes=ref_img_sizes,
                    ref_latents_N=ref_latents_N,
                    ref_img_mask_N=ref_img_mask_N,
                    l_effective_ref_img_len_N=l_effective_ref_img_len_N,
                    ref_img_sizes_N=ref_img_sizes_N,
                    text_guidance_scale=text_guidance_scale,
                    image_guidance_scale=image_guidance_scale
                )
                latents_dtype = latents.dtype
                if return_middle_statistics:
                    latents, log_probs = self.scheduler.step(model_pred.to(dtype=torch.float32), t, latents.to(dtype=torch.float32), generator=generator, return_log_prob=True, img_mask=img_mask, mixed_precision=self.mixed_precision, return_dict=False)
                else:
                    latents = self.scheduler.step(model_pred, t, latents, generator=generator, mixed_precision=self.mixed_precision, return_dict=False)[0]

                if latents.dtype != latents_dtype and not self.mixed_precision:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if return_middle_statistics:
                    latents_list.append(latents.clone())
                    log_probs_list.append(log_probs)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update(1)

        image = []
        for latent, img_len, img_size in zip(latents, l_effective_img_len, img_sizes):
            height, width = img_size
            p = self.transformer.config.patch_size
            latent = rearrange(latent[:img_len], '(h w) (p1 p2 c) -> c (h p1) (w p2)', h=height // p, w=width // p, p1=p, p2=p)
            latent = latent.to(dtype=dtype)
            if self.vae.config.scaling_factor is not None:
                latent = latent / self.vae.config.scaling_factor
            if self.vae.config.shift_factor is not None:
                latent = latent + self.vae.config.shift_factor
            image.append(self.vae.decode(latent.unsqueeze(0), return_dict=False)[0].squeeze(0))
        
        if return_middle_statistics:
            return image, latents_list, log_probs_list, img_mask, l_effective_img_len, img_sizes, ref_latents, ref_img_mask, l_effective_ref_img_len, ref_img_sizes
        else:
            return image

    def predict(
        self,
        t,
        latents,
        text_hidden_states,
        text_attention_mask,
        ref_image_hidden_states,
        freqs_cis,
        **model_kwargs,
    ):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        if not self.mixed_precision:
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
        else:
            timestep = t

        # batch_size, num_channels_latents, height, width = latents.shape
        from accelerate.utils import extract_model_from_parallel
        if 'ref_image_hidden_states' in set(inspect.signature(extract_model_from_parallel(self.transformer).forward).parameters.keys()):
            model_kwargs['ref_image_hidden_states'] = ref_image_hidden_states
        
        model_pred = self.transformer(
            latents,
            timestep,
            text_hidden_states,
            text_attention_mask,
            freqs_cis,
            **model_kwargs,
        )
        return model_pred