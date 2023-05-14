from utils.my_containers import Register

loss_register = Register()

from .contrastive_loss import PlainContrastiveLoss
from .cm_contrastive_loss import PlainDualModalityContrastiveLoss


def get_loss(loss_cfg):
    return loss_register[loss_cfg.name](loss_cfg)
