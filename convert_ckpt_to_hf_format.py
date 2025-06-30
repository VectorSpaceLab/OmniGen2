import dotenv

dotenv.load_dotenv(override=True)

import argparse

from omegaconf import OmegaConf

import torch
from accelerate import init_empty_weights

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline


def main(args):
    config_path = args.config_path
    model_path = args.model_path

    conf = OmegaConf.load(config_path)
    arch_opt = conf.model.arch_opt

    arch_opt = OmegaConf.to_object(arch_opt)
    # Convert lists to tuples in conf.model.arch_opt
    for key, value in arch_opt.items():
        if isinstance(value, list):
            arch_opt[key] = tuple(value)

    with init_empty_weights():
        transformer = OmniGen2Transformer2DModel(**arch_opt)

        if conf.train.get('lora_ft', False):
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

            # now we will add new LoRA weights the transformer layers
            lora_config = LoraConfig(
                r=conf.train.lora_rank,
                lora_alpha=conf.train.lora_rank,
                lora_dropout=conf.train.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            transformer.add_adapter(lora_config)

    state_dict = torch.load(model_path, mmap=True, weights_only=True)
    missing, unexpect = transformer.load_state_dict(
        state_dict, assign=True, strict=False
    )
    print(f"missed parameters: {missing}")
    print(f"unexpected parameters: {unexpect}")

    save_path = args.save_path
    if conf.train.get('lora_ft', False):
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        OmniGen2Pipeline.save_lora_weights(
            save_directory=save_path,
            transformer_lora_layers=transformer_lora_layers,
        )
    else:
        transformer.save_pretrained(save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
