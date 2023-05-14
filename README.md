


# Environments
torch                     1.7.1+cu110
torch-cluster             1.5.9
torch-geometric           1.6.3
torch-scatter             2.0.7
torch-sparse              0.6.10
torch-spline-conv         1.2.1
torchvision               0.8.2+cu110

# Data
CIL dataset originally consists of 919,265 cellular images collected from 30,616 molecular intervention.

The current CIL data in [data](https://drive.google.com/drive/folders/1_JYDE2AUBePDsJ9ux8AmC2ZbdlWJmNEq?usp=sharing) includes 50 molecules and 1270 corresponding images, we would release the full version after being accepted.


# For pre-training
Check the following scripts:
```
    pretrain.sh
```


# For downstream task
Check the following scripts:

```
    finetune_classification.sh
    finetune_regression.sh
    finetune_clinical.sh
```