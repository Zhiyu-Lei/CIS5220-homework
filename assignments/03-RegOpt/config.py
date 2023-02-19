from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 128
    num_epochs = 10
    initial_learning_rate = 0.002
    initial_weight_decay = 0.001

    lrs_kwargs = {
        "init_step_size": 1564,
        "step_size_inc": 2,
        "gamma": 0.5,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose([ToTensor(), Normalize(0.5, 0.5)])
