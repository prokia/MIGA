import os, random
from tqdm import tqdm
import numpy as np
import pandas as pd

from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import average_precision_score, roc_auc_score, mean_squared_error

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from utils.config_args import args
from utils.splitters import random_scaffold_split, random_split, scaffold_split
from core.network.model.encoder.gnn.gnn import GNN_graphpred
from clinical_dataset import MoleculeDataset


def seed_everything(seed: int=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def train(args, model, device, loader, optimizer, scheduler, criterion):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float)

        is_valid = ~torch.isnan(y)
        valid_y = y[is_valid]
        valid_pred = pred[is_valid]
        loss_mat = criterion(valid_pred, valid_y)

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()

        optimizer.step()
    scheduler.step()


def smiles_txt_to_lst(text):
    """
    "['CN[C@H]1CC[C@@H](C2=CC(Cl)=C(Cl)C=C2)C2=CC=CC=C12', 'CNCCC=C1C2=CC=CC=C2CCC2=CC=CC=C12']" 
    """
    text = text[1:-1]
    lst = [i.strip()[1:-1] for i in text.split(',')]
    return lst 

def eval(args, model, device, loader, metric, criterion):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
 

    roc_list, pr_auc_list = [], []
    for i in range(y_true.shape[1]):
        if metric == 'roc':
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_valid = ~np.isnan(y_true[:,i])
                roc_list.append(roc_auc_score(y_true[is_valid,i] , y_scores[is_valid,i]))
                pr_auc_list.append(average_precision_score(y_true[is_valid,i] , y_scores[is_valid,i]))
        elif metric == 'rmse':
            roc_list.append(mean_squared_error(y_true, y_scores, squared=False))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))
    print(roc_list)
    return (sum(roc_list)/len(roc_list), sum(pr_auc_list)/len(pr_auc_list)) #y_true.shape[1]

def main():

    if args.seed:
        seed = args.seed
        print("Manual seed: ", seed)
    else:
        seed = random.randint(0, 10000)
        print("Random seed: ", seed)

    # Bunch of classification tasks
    if args.dataset == "phaseI":
        num_tasks = 1
        metric = 'roc'
    elif args.dataset == "phaseII":
        num_tasks = 1
        metric = 'roc'
    elif args.dataset == "phaseIII":
        num_tasks = 1
        metric = 'roc'
    else:
        raise ValueError("Invalid dataset name.")
    # set up dataset
    dataset = MoleculeDataset("dataset/{}".format(args.dataset), dataset=args.dataset)
    print(len(dataset))
    dataset =[dataset[n] for n in range(len(dataset))]

    # Read in dataset
    print("Reading dataset")
    # By default, the line above will save data to the path below
    df = pd.read_csv("dataset/{}/processed/smiles.csv".format(args.dataset),header=None)
    
    smiles_list = df[0].tolist()
    smiles = smiles_list
    
    if metric == 'roc':
        criterion = nn.BCEWithLogitsLoss(reduction = "none")
        null_value = float('nan')
    elif metric == 'rmse':
        criterion = nn.MSELoss(reduction = 'none')
        null_value = float('nan')
        print(dataset[0])
    
    if args.split == "scaffold":
        smiles_list = smiles #pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=float('nan'), frac_train=0.8,frac_valid=0.1, frac_test=0.1, return_smiles=False)
        # print(len(train_dataset),len(valid_dataset),len(test_dataset))
        
        print("scaffold")
    elif args.split =='ogb_scaffold':
        split_idx = dataset.get_idx_split()
        train_idxs, val_idxs, test_idxs = split_idx["train"], split_idx["valid"], split_idx["test"]
        train_dataset = [dataset[idxs] for idxs in train_idxs]
        valid_dataset = [dataset[idxs] for idxs in val_idxs]
        test_dataset = [dataset[idxs] for idxs in test_idxs]
        print("ogb_scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset, smiles_splits_list = random_split(
            dataset, null_value=float('nan'), frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed, smiles_list=smiles
        )

        input_df = pd.read_csv(f"dataset/{args.dataset}/df.csv")
        train_smiles, valid_smiles, test_smiles = smiles_splits_list
        # print('smiles', len(train_smiles), len(valid_smiles), len(test_smiles))
        input_df['smiles'] = input_df.smiless.apply(lambda x :smiles_txt_to_lst(x)[0])
        input_df['random_splits'] = 'invalid'
        input_df.loc[input_df.smiles.isin(train_smiles), 'random_splits'] = 'train'
        input_df.loc[input_df.smiles.isin(valid_smiles), 'random_splits'] = 'valid'
        input_df.loc[input_df.smiles.isin(test_smiles), 'random_splits'] = 'test'
        input_df.to_csv(f"dataset/{args.dataset}/split_df.csv")
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = smiles  # pd.read_csv("./dataset-o/" + args.dataset + "/processed/smiles.csv", header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
        )
        print("random scaffold")

    elif args.split == "defined":
        input_df = pd.read_csv(f"dataset/{args.dataset}/df.csv")
        train_idx = input_df.loc[input_df.split=='train'].index.tolist()
        valid_idx = input_df.loc[input_df.split=='valid'].index.tolist()
        test_idx = input_df.loc[input_df.split=='test'].index.tolist()

        train_dataset = [dataset[n] for n in torch.tensor(train_idx)]
        valid_dataset = [dataset[n] for n in torch.tensor(valid_idx)]
        test_dataset = [dataset[n] for n in torch.tensor(test_idx)]


    else:
        raise ValueError("Invalid split option.")

    # print(train_dataset[0].edge_index.dtype)

    # run multiple times
    best_valid_auc_list, best_valid_prauc_list = [], []
    last_epoch_auc_list, last_epoch_prauc_list = [], []
    seeds = []
    for run_idx in range(args.num_run):
        print("\nRun ", run_idx + 1)
        if args.runseed:
            # runseed = seeds[run_idx]
            runseed = args.runseed + 1000 * run_idx
            print("Manual runseed: ", runseed)
        else:
            runseed = random.randint(0, 10000)
            print("Random runseed: ", runseed)
            seeds.append(runseed)

        seed_everything(runseed)
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # set up model
        if args.model_name == "gnn":
            model = GNN_graphpred(
                args.num_layer,
                args.emb_dim,
                num_tasks,
                JK=args.JK,
                drop_ratio=args.dropout_ratio,
                graph_pooling=args.graph_pooling,
                gnn_type=args.gnn_type,
            )
        if not args.input_model_file == "":
            # ! We force the model to load pretrained model
            model.from_pretrained(args.input_model_file)
            print("Successfully load pretrained model")

        model.to(device)

        # set up optimizer
        # different learning rate for different part of GNN
        model_param_group = []
        if args.frozen:
            model_param_group.append({"params": model.gnn.parameters(), "lr": 0})
        else:
            model_param_group.append({"params": model.gnn.parameters()})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        print(optimizer)

        scheduler = StepLR(optimizer, step_size=30, gamma=0.3)

        # run fine-tuning
        if metric == "roc":
            best_valid = 0
        elif metric == "rmse":
            best_valid = 1e10
        best_valid_test, best_valid_test_pr = 0, 0
        last_epoch_test, last_epoch_test_pr = 0, 0
        for epoch in range(1, args.epochs + 1):
            print("====epoch " + str(epoch), " lr: ", optimizer.param_groups[-1]["lr"])

            train(args, model, device, train_loader, optimizer, scheduler, criterion)

            print("====Evaluation")
            if args.eval_train:
                train_acc = eval(args, model, device, train_loader, metric, criterion)
            else:
                print("omit the training accuracy computation")
                train_acc = 0
            val_acc = eval(args, model, device, val_loader, metric, criterion)
            test_acc = eval(args, model, device, test_loader, metric, criterion)
            if metric == 'roc':
                if val_acc[0] > best_valid:
                    best_valid = val_acc[0]
                    best_valid_test = test_acc[0]
                    best_valid_test_pr = test_acc[1]
            elif metric == 'rmse':
                if val_acc < best_valid:
                    best_valid = val_acc
                    best_valid_test = test_acc

            if epoch == args.epochs:
                last_epoch_test = test_acc[0]
                last_epoch_test_pr = test_acc[1]

            print("AUC\t train: %f val: %f test: %f" % (train_acc[0], val_acc[0], test_acc[0]))
            print("PR-AUC\t train: %f val: %f test: %f" % (train_acc[1], val_acc[1], test_acc[1]))
            print("")

        best_valid_auc_list.append(best_valid_test)
        last_epoch_auc_list.append(last_epoch_test)
        best_valid_prauc_list.append(best_valid_test_pr)
        last_epoch_prauc_list.append(last_epoch_test_pr)

    # summarize results
    best_valid_auc_list = np.array(best_valid_auc_list)
    last_epoch_auc_list = np.array(last_epoch_auc_list)
    best_valid_prauc_list = np.array(best_valid_prauc_list)
    last_epoch_prauc_list = np.array(last_epoch_prauc_list)

    print('seeds:', seeds)

    # if args.dataset in ["muv", "hiv"]:
    print("best_valid_auc_list: ", best_valid_auc_list)
    print("last_epoch_auc_list: ", last_epoch_auc_list)
    print("Last epoch:")
    print("Mean: {}\tStd: {}".format(np.mean(last_epoch_auc_list), np.std(last_epoch_auc_list)))
    print("Best validation epoch:")
    print("Mean: {}\tStd: {}".format(np.mean(best_valid_auc_list), np.std(best_valid_auc_list)))


    # if args.dataset in ["muv", "hiv"]:
    print("best_valid_prauc_list: ", best_valid_prauc_list)
    print("last_epoch_prauc_list: ", last_epoch_prauc_list)
    print("Last epoch:")
    print("Mean: {}\tStd: {}".format(np.mean(last_epoch_prauc_list), np.std(last_epoch_prauc_list)))
    print("Best validation epoch:")
    print("Mean: {}\tStd: {}".format(np.mean(best_valid_prauc_list), np.std(best_valid_prauc_list)))


if __name__ == "__main__":
    main()
