from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Custom learning rate scheduler
    """

    def __init__(self, optimizer, init_step_size, step_size_inc, gamma, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.
        """
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        self.step_size = init_step_size
        self.step_size_inc = step_size_inc
        self.gamma = gamma
        self.changed_epoch = 0

    def get_lr(self) -> List[float]:
        """
        Get the current learning rates from the scheduler.

        Note to students: You CANNOT change the arguments or return type of
        this function (because it is called internally by Torch)
        """
        if (
            self.last_epoch == 0
            or self.last_epoch - self.changed_epoch < self.step_size
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        self.step_size *= self.step_size_inc
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
