import sys;
sys.path.extend(['/rhome/lianyu.zhou/code/MIGA', '/rhome/lianyu.zhou/code/MIGA'])
import argparse
import os, random
from collections import OrderedDict

from rdkit.Chem import AllChem
from tqdm import tqdm
import numpy as np
import pandas as pd

from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import roc_auc_score, mean_squared_error

import sys

from core.network.model.encoder.graph_transformer.transformer_model import GraphTransformer
from dataset.dataloader import DataLoaderNoiseMatrix
from dataset.utils.collate_fn import BatchNoiseMatrix
from src_downstream_utils.down_stream_utils import LinearPredictHead, load_pretrain_cfg, resume_pretrain, \
    complete_atom_encoder
from utils.geometric_graph import ATOM_ENCODER
from utils.tensor_operator import to_device
from utils.utils import custom_load_pretrained_dict

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from config import args
from splitters import random_scaffold_split, random_split, scaffold_split

from models import GNN_graphpred
from src_classification.classification_dataset import MoleculeDataset

from utils.loggers import LoggerManager

logger = LoggerManager("/rhome/lianyu.zhou/code/MIGA/run_classification_finetune").get_logger()
logger("Let the games begin :)")


def seed_everything(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, device, loader, optimizer, scheduler, criterion, predict_head=None):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = to_device(batch, device)
        g = batch['graphs']
        node_mask = batch['node_masks']
        pred = model(X=g.X, E=g.E, node_mask=node_mask)
        if predict_head is not None:
            pred = predict_head(pred, node_mask)
        y = batch['y'].view(pred.shape).to(torch.float)

        # # Whether y is non-null or not.
        # is_valid = y ** 2 > 0
        # # Loss matrix
        # loss_mat = criterion(pred.double(), (y + 1) / 2)
        # # loss matrix after removing null target
        # loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        is_valid = ~torch.isnan(y)
        valid_y = y[is_valid]
        valid_pred = pred[is_valid]
        loss_mat = criterion(valid_pred, valid_y)

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()

        optimizer.step()
    scheduler.step()


def eval(args, model, device, loader, metric, criterion, predict_head):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = to_device(batch, device)
        g = batch['graphs']
        node_mask = batch['node_masks']
        with torch.no_grad():
            pred = model(X=g.X, E=g.E, node_mask=node_mask)
            if predict_head is not None:
                pred = predict_head(pred, node_mask)

        y_true.append(batch['y'].view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        if metric == 'roc':
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                is_valid = ~np.isnan(y_true[:, i])
                roc_list.append(roc_auc_score(y_true[is_valid, i], y_scores[is_valid, i]))
        elif metric == 'rmse':
            roc_list.append(mean_squared_error(y_true, y_scores, squared=False))

    if len(roc_list) < y_true.shape[1]:
        logger("Some target is missing!")
        logger("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))
    logger(roc_list)
    return sum(roc_list) / len(roc_list)  # y_true.shape[1]

def main():
    # Training settings
    pretrain_cfg = load_pretrain_cfg(args)
    if args.seed:
        seed = args.seed
        logger(f"Manual seed: {seed}")
    else:
        seed = random.randint(0, 10000)
        logger("Random seed: ", seed)

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
        metric = 'roc'
    elif args.dataset == "hiv":
        num_tasks = 1
        metric = 'roc'
    elif args.dataset == "bbbp":
        num_tasks = 1
        metric = 'roc'
    elif args.dataset == "toxcast":
        num_tasks = 617
        metric = 'roc'
    else:
        raise ValueError("Invalid dataset name.")

    # set up dataset
    ds = "ogbg-mol{}".format(args.dataset)
    # By default, the line above will save data to the path below
    df = pd.read_csv(f"/gxr/shuangjia/github/omics_infomax_jiahua/datasets/{ds}/mapping/mol.csv.gz".replace("-", "_"))
    smiles = df.smiles.values
    complete_atom_encoder(smiles)
    dataset = MoleculeDataset("/gxr/shuangjia/github/omics_infomax_jiahua/datasets/{}".format(args.dataset),
                              dataset=args.dataset)
    dataset = [dataset[n] for n in range(len(dataset))]
    # Read in dataset
    logger("Reading dataset")
    # ! Becareful of whether hrodrogens are included in you pretraining or not
    # * Decide to not addhs as Hydrogen is time consuming
    # dataset = [smiles2graph_addhs(s) for s in smiles]
    # pretrain_cfg.network.molEncoder.input_dims.X = len(ATOM_ENCODER)
    for idx, graph in enumerate(dataset):
        y = dataset[idx].y
        y = y.float()
        if metric == 'roc':
            y[y == 0.] = float('nan')
            y[y == -1.0] = 0.0
            dataset[idx].y = y

    if metric == 'roc':
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        null_value = float('nan')
    elif metric == 'rmse':
        criterion = nn.MSELoss(reduction='none')
        null_value = float('nan')
        logger(dataset[0])

    if args.split == "scaffold":
        smiles_list = smiles  # pd.read_csv('datasets/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=float('nan'),
                                                                    frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                                                                    return_smiles=False)
        logger("scaffold")
    elif args.split == 'ogb_scaffold':
        split_idx = dataset.get_idx_split()
        train_idxs, val_idxs, test_idxs = split_idx["train"], split_idx["valid"], split_idx["test"]
        train_dataset = [dataset[idxs] for idxs in train_idxs]
        valid_dataset = [dataset[idxs] for idxs in val_idxs]
        test_dataset = [dataset[idxs] for idxs in test_idxs]
        logger("ogb_scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=float('nan'), frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
        )
        logger("random")
    elif args.split == "random_scaffold":
        smiles_list = smiles  # pd.read_csv("./dataset-o/" + args.dataset + "/processed/smiles.csv", header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
        )
        logger("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    # logger(train_dataset[0].edge_index.dtype)

    # run multiple times
    best_valid_auc_list = []
    last_epoch_auc_list = []
    for run_idx in range(args.num_run):
        logger(f"Run {run_idx + 1}")
        if args.runseed:
            runseed = args.seed + run_idx
            logger(f"Manual runseed: {runseed}", )
        else:
            runseed = random.randint(0, 10000)
            logger("Random runseed: ", runseed)

        seed_everything(runseed)
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        args.num_workers = 0
        train_loader = DataLoaderNoiseMatrix(train_dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers, transforms=None)
        val_loader = DataLoaderNoiseMatrix(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers, transforms=None)
        test_loader = DataLoaderNoiseMatrix(test_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers, transforms=None)

        # set up model
        if False and args.model_name == "gnn":
            model = GNN_graphpred(
                args.num_layer,
                args.emb_dim,
                num_tasks,
                JK=args.JK,
                drop_ratio=args.dropout_ratio,
                graph_pooling=args.graph_pooling,
                gnn_type=args.gnn_type,
            )
        if 'name' in pretrain_cfg.network.molEncoder:
            pretrain_cfg.network.molEncoder.pop('name')
        model = GraphTransformer(**pretrain_cfg.network.molEncoder)
        predict_head_input_dim = pretrain_cfg.network.molEncoder.hidden_dims.dx
        predict_head = LinearPredictHead(predict_head_input_dim, num_tasks)
        predict_head.to(device)
        # resume_pretrain(pretrain_cfg, model)
        # if not args.input_model_file == "":
        # # ! We force the model to load pretrained model
        #     model.from_pretrained(args.input_model_file)
        #     logger("Successfully load pretrained model")

        model.to(device)

        # set up optimizer
        # different learning rate for different part of GNN
        # model_param_group = []
        # if args.frozen:
        #     model_param_group.append({"params": model.gnn.parameters(), "lr": 0})
        # else:
        #     model_param_group.append({"params": model.gnn.parameters()})
        # if args.graph_pooling == "attention":
        #     model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
        # model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
        # optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

        logger(optimizer)

        scheduler = StepLR(optimizer, step_size=30, gamma=0.3)

        # run fine-tuning
        if metric == "roc":
            best_valid = 0
        elif metric == "rmse":
            best_valid = 1e10
        best_valid_test = 0
        last_epoch_test = 0
        best_test_acc = 0
        for epoch in range(1, args.epochs + 1):
            logger(f"====epoch {str(epoch)} lr: {optimizer.param_groups[-1]['lr']}")

            train(args, model, device, train_loader, optimizer, scheduler, criterion, predict_head)

            logger("====Evaluation")
            if args.eval_train:
                train_acc = eval(args, model, device, train_loader, metric, criterion, predict_head)
            else:
                logger("omit the training accuracy computation")
                train_acc = 0
            val_acc = eval(args, model, device, val_loader, metric, criterion, predict_head)
            test_acc = eval(args, model, device, test_loader, metric, criterion, predict_head)
            best_test_acc = max(best_test_acc, test_acc)
            if metric == 'roc':
                if val_acc > best_valid:
                    best_valid = val_acc
                    best_valid_test = test_acc
            elif metric == 'rmse':
                if val_acc < best_valid:
                    best_valid = val_acc
                    best_valid_test = test_acc
            if epoch > 60 and test_acc < 0.7:
                break
            if epoch > 80 and best_test_acc < 0.73:
                break

            if epoch == args.epochs:
                last_epoch_test = test_acc

            logger("train: %f val: %f test: %f" % (train_acc, val_acc, test_acc))
            logger("")

        best_valid_auc_list.append(best_valid_test)
        last_epoch_auc_list.append(last_epoch_test)

    # summarize results
    best_valid_auc_list = np.array(best_valid_auc_list)
    last_epoch_auc_list = np.array(last_epoch_auc_list)

    # if args.dataset in ["muv", "hiv"]:
    logger(f"best_valid_auc_list: {best_valid_auc_list}")
    logger(f"last_epoch_auc_list: {last_epoch_auc_list}")
    logger("Last epoch:")
    logger("Mean: {}\tStd: {}".format(np.mean(last_epoch_auc_list), np.std(last_epoch_auc_list)))
    logger("Best validation epoch:")
    logger("Mean: {}\tStd: {}".format(np.mean(best_valid_auc_list), np.std(best_valid_auc_list)))


if __name__ == "__main__":
    main()
