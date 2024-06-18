import os

from torch.cuda.amp import autocast

from core.runner import runner_register, PretrainRunner
from utils.tensor_operator import to_device

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


from dataset.utils.mol_transform import MaskAtom
from dataset.dataloader import DataLoaderDenseMatrix
from utils.loggers import LoggerManager

logger = LoggerManager().get_logger()


@runner_register
class DenseGraphPretrainRunner(PretrainRunner):
    def __init__(self, *args, **kwargs):
        super(DenseGraphPretrainRunner, self).__init__(*args, **kwargs)

    def init_train_loader(self):
        # set up dataset
        dataset = self.train_dataset
        args = self.cfg

        loader = DataLoaderDenseMatrix(dataset, transforms=None, batch_size=args.batch_size, drop_last=True,
                                       shuffle=True, num_workers=args.num_workers)
        return loader

    def init_eval_loader(self):
        args = self.cfg
        dataset = self.test_dataset
        transform = MaskAtom(num_atom_type=119, num_edge_type=5,
                             mask_rate=args.mask_rate, mask_edge=args.mask_edge)

        loader = DataLoaderDenseMatrix(dataset, transforms=transform, batch_size=args.batch_size, shuffle=False,
                                       drop_last=False,
                                       num_workers=args.num_workers)
        return loader

    def train_step(self, data_dict):
        self.model.train()
        batch = to_device(data_dict, self.device)
        self.optimizer.zero_grad()
        fp16 = self.cfg.get('fp16', False)
        with autocast(fp16):
            model_out_dict = self.model(batch)
            self.fp16_backward(model_out_dict['total_loss'])
        if fp16:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        self.update_loss(model_out_dict)
