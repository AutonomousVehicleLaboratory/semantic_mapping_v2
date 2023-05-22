# Customized Learning Rate Scheduler


from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_iter

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float): Base learning rate
        max_iter (float): After this step, we stop decreasing learning rate
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_iter, power=0.9):
        assert max_iter > 1, "max_iter should be greater than 1."
        self.lrs = []  # lrs stands for 'Learning Rate(s)'
        self.max_iter = max_iter
        self.power = power
        # A local step counter, starts from 0 index so the first lr of the network is base_lr
        # Note that we did not use self._step_count because it is starts from 1 index
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step >= self.max_iter:
            return self.lrs

        self.lrs = [base_lr * (1 - self.last_step / self.max_iter) ** (self.power) for base_lr in self.base_lrs]
        self.last_step += 1
        return self.lrs


def test_PolyLRDecay():
    import numpy as np
    import torch.optim
    import torch.nn as nn

    max_iter = 30
    base_lr = 0.1
    power = 0.9
    conv2d = nn.Conv2d(3, 5, kernel_size=3)
    optimizer = torch.optim.SGD(conv2d.parameters(), lr=base_lr)
    scheduler = PolynomialLRDecay(optimizer, base_lr=base_lr, max_iter=max_iter, power=power)

    for i in range(max_iter):
        lr = optimizer.param_groups[0]['lr']
        lr_np = base_lr * (1 - i / max_iter) ** power
        assert np.isclose(lr, lr_np), "{} - {}".format(lr, lr_np)

        optimizer.step()
        scheduler.step()


if __name__ == '__main__':
    test_PolyLRDecay()
