import random
import numpy as np

import torch

from diffusers.utils import is_torch_npu_available


def worker_init_fn(worker_id, num_processes, num_workers, process_index, seed, same_seed_per_epoch=False):
    if same_seed_per_epoch:
        worker_seed = seed + num_processes + num_workers * process_index + worker_id
    else:
        worker_seed = torch.initial_seed()

    random.seed(worker_seed)
    np.random.seed(worker_seed % 2**32)
    torch.manual_seed(worker_seed)

    if is_torch_npu_available():
        torch.npu.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed_all(seed)