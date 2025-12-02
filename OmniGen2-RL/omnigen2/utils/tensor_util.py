from typing import List

from PIL import Image

from torch.nn import functional as F

def pad_to_length(tensor, len, dim=1):
    return F.pad(tensor, [0] * ((tensor.dim() - 2) * 2) + [0, len - tensor.shape[dim]])

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