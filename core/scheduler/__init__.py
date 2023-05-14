
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from core.scheduler.schedulers import get_warm_up_lr, ConstantLR, PolynomialLR
from utils.loggers import LoggerManager

key2scheduler = {
    "constant_lr": ConstantLR,
    "poly_lr": PolynomialLR,
    "multi_step": MultiStepLR,
    "cosine_annealing": CosineAnnealingLR,
    "exp_lr": ExponentialLR,
    "escape_plateau": ReduceLROnPlateau
}


def get_scheduler(optimizer, scheduler_dict):
    logger = LoggerManager().get_logger('Scheduler')
    if scheduler_dict is None:
        logger("Using No LR Scheduling")
        return ConstantLR(optimizer)

    s_type = scheduler_dict["name"]
    scheduler_dict.pop("name")

    logger("Using {} scheduler with {} params".format(s_type, scheduler_dict))

    if "warmup_epochs" in scheduler_dict:
        needs = ["warmup_mode", "warmup_epochs", "warmup_ratio", "min_warm_up_lr"]
        defaults = ["linear", 5, 0.2, 1.0e-7]
        # This can be done in a more pythonic way...
        # Sure, I modify it
        warmup_args = [scheduler_dict.get(need, default)
                       for need, default in zip(needs, defaults) ]

        logger(
            "Using Warmup with {} warmup_mode, epochs = {} and warmup_ratio = {} ".format(
                *warmup_args)
            )

        scheduler_dict.pop("warmup_epochs", None)
        scheduler_dict.pop("warmup_mode", None)
        scheduler_dict.pop("warmup_ratio", None)
        scheduler_dict.pop("min_warm_up_lr", None)

        base_scheduler = key2scheduler[s_type]
        return get_warm_up_lr(base_scheduler)(*warmup_args, optimizer=optimizer, **scheduler_dict)

    return key2scheduler[s_type](optimizer, **scheduler_dict)
