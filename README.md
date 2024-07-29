####  Introduction
Source code for Cross-modal Graph Contrastive Learning with Cellular Images.


##### [Paper on Advanced Science](https://onlinelibrary.wiley.com/doi/10.1002/advs.202404845)

####  Environments
MIGA requires anaconda with python 3.7 or later, cudatoolkit=11.1 and below packages
```
torch                     1.7.1+cu110
torch-cluster             1.5.9
torch-geometric           1.6.3
torch-scatter             2.0.7
torch-sparse              0.6.10
torch-spline-conv         1.2.1
torchvision               0.8.2+cu110
```

MIGA has been tested on Ubuntu 18.04, with eight GPUs (Nvidia RTX4090). Installation should take no longer than 20 minutes on a modern server.

Please refer to environment.yml

# Data
CIL dataset originally consists of 919,265 cellular images collected from 30,616 molecular intervention.

The current CIL data in [data](https://drive.google.com/drive/folders/1_JYDE2AUBePDsJ9ux8AmC2ZbdlWJmNEq?usp=sharing) includes 50 molecules and 1270 corresponding images, we would release the full version after being accepted.


# For pre-training
Check the following scripts:
GIN:
```
    python submit.py --config config/miga/miga_gin.yaml
```
Graph Transformer:
```
    python submit.py --config config/miga/miga_graphTrans.yaml
```

# For downstream task (Only support GIN)
Check the following scripts:

```
    finetune_classification.sh
    finetune_regression.sh
    finetune_clinical.sh
```

# MIGA's pretrained model weights
----------------------------------

| Model                     | File Size  |Update Date | Download Link                                                | 
|--------------------------|------------| ------------|--------------------------------------------------------------|
| molecular pretrain (GIN)       | 81MB   | Aug 17 2022 |  [[model weights](https://github.com/prokia/MIGA/blob/main/model/miga_gin_256.pth)]     |
| molecular pretrain (GraphTransformer)          | 96MB   | Feb 05 2023 | [[model weights](https://github.com/prokia/MIGA/blob/main/model/miga_graphtrans_256.pth)]      |



# MIGA representation
## molecule and atoms level representation

```python
import torch
from core.network import MIGA
from dataset import process_data

model = MIGA('graph_transformer', is_eval=True)
model.eval()
checkpoint = torch.load('models/miga_graphtrans_256.pth', map_location='cpu')
model.load_state_dict(checkpoint, strict=False)

smiles = 'c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]'
data = process_data(smiles, 'graph_transformer')
molecule_embeddings = model.get_graph_embedding(data)
```

# Citation
Please cite the following paper if you use this code in your work.

```
@article{zheng2022cross,
  title={Cross-Modal Graph Contrastive Learning with Cellular Images},
  author={Zheng, Shuangjia and Rao, Jiahua and Zhang, Jixian and Zhou, Lianyu and Xie, Jiancong and Cohen, Ethan and Lu, Wei and Li, Chengtao and Yang, Yuedong},
  journal={Advanced Science},
  pages={2404845},
  year={2022},
  publisher={Wiley Online Library}
}
```
