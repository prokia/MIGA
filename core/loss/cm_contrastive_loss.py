from core.loss import loss_register
from core.loss.contrastive_loss import PlainContrastiveLoss
from core.loss.utils import loss_dict_remake_wrapper


@loss_register
class PlainDualModalityContrastiveLoss(PlainContrastiveLoss):
    def __init__(self, cfg):
        super(PlainDualModalityContrastiveLoss, self).__init__(cfg)

    @loss_dict_remake_wrapper
    def forward(self, embedding_1, embedding_2=None):
        if embedding_2 is None:
            return super(PlainDualModalityContrastiveLoss, self).forward(embedding_1)
        return self.cal_loss_funcs(embedding_1, embedding_2)
