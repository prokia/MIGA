import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.my_containers import ObjDict


class AutoImgTransformationManager(object):
    def __init__(self, aug_cfg):
        self.cfg = aug_cfg

    def __call__(self, name=None):
        if name is None:
            name = self.cfg.name
        assert name is not None
        if isinstance(name, str):
            return self.__getattribute__(name)()
        aug_seq_list = []
        for each_name in name:
            aug_seq = self.__getattribute__(each_name)()
            aug_seq_list.append(aug_seq)
        return A.Compose(aug_seq_list)

    # def basic_identity(self):
    #     return A.Compose([A.Noop()])

    def base_resize(self):
        return A.Compose([
            A.Resize(height=self.cfg.input_size[0], width=self.cfg.input_size[1])],
            p=1.0)

    def base_toTensorV2(self):
        return A.Compose([ToTensorV2()], p=1.0)

    def classic_miga_aug(self):
        h, w = self.cfg.input_size
        A.Compose([
            A.Resize(height=h, width=w, p=1.0),
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightness(limit=0.2 / 16, p=0.5),
            A.RandomContrast(limit=0.2 / 16, p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.75),
            # A.Cutout(max_h_size=int(img_size*0.7), max_w_size=int(img_size*0.7), num_holes=1, p=0.5),
        ],
            p=1.0)
