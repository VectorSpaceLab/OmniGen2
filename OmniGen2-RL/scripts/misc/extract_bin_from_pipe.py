import dotenv

dotenv.load_dotenv(override=True)

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel


def main():
    transformer = OmniGen2Transformer2DModel.from_pretrained("OmniGen2/OmniGen2", subfolder="transformer")

    state_dict = transformer.state_dict()

    save_path = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "pretrained_models", "OmniGen2", "transformer/pytorch_model.bin")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(state_dict, save_path)

if __name__ == "__main__":
    main()
