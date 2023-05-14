from torch.optim.lr_scheduler import _LRScheduler


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):

    def __init__(self, optimizer, power=1., min_lr=0., max_epoch=200, last_epoch=-1):
        self.power = power
        self.min_lr = min_lr
        self.max_epoch = max_epoch
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        coeff = (1 - min(self.last_epoch, self.max_epoch) / self.max_epoch) ** self.power
        return [(base_lr - self.min_lr) * coeff + self.min_lr for base_lr in self.base_lrs]


def get_warm_up_lr(cls=_LRScheduler):
    class WarmUpLR(cls):
        def __init__(self,  mode="linear", warmup_epochs=5, warmup_ratio=0.2, min_warm_up_lr=1.0e-7, *args, **kwargs):
            self.warmup_mode = mode
            self.warmup_epochs = warmup_epochs
            self.warmup_ratio = warmup_ratio
            self.min_warm_up_lr = min_warm_up_lr
            super(WarmUpLR, self).__init__(*args, **kwargs)

        def get_warm_up_lr(self, cold_lrs):
            if self.warmup_mode == 'linear':
                k = (1 - self.last_epoch / self.warmup_epochs) * (1 - self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in cold_lrs]
            elif self.warmup_mode == 'exp':
                k = self.warmup_ratio ** (1 - self.last_epoch / self.warmup_epochs)
                warmup_lr = [_lr * k for _lr in cold_lrs]
            else:
                warmup_lr = [_lr * self.warmup_ratio for _lr in cold_lrs]
            return warmup_lr

        def get_lr(self):
            cold_lrs = super(WarmUpLR, self).get_lr()

            if self.last_epoch < self.warmup_epochs:
                return self.get_warm_up_lr(cold_lrs)
            return cold_lrs

    return WarmUpLR
