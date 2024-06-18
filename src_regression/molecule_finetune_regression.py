import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from os.path import join
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from torch_geometric.data import DataLoader

from core.network.model.encoder.gnn.gnn import GNN_graphpred
from regression_dataset import MoleculeDatasetRegression

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from utils.config_args import args
from utils.splitters import random_scaffold_split, random_split, scaffold_split


import random
def seed_everything(seed: int=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
        y = batch.y.squeeze().float()

        loss = reg_criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval(model, device, loader):
    model.eval()
    y_true, y_pred = [], []

    # for step, batch in enumerate(loader):
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(1)
 
        true = batch.y.view(pred.shape)
        y_true.append(true)
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}, y_true, y_pred


if __name__ == '__main__':

    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')

    num_tasks = 1
    dataset_folder = './dataset/'
    ds = "ogbg_mol{}".format(args.dataset)
    dataset_folder = os.path.join(dataset_folder, ds)
    dataset = MoleculeDatasetRegression(dataset_folder, dataset=args.dataset)
    print('dataset_folder:', dataset_folder)
    print(dataset)

    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, regression=True)
        print('split via scaffold')
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed, regression=True)
        print('randomly split')
    elif args.split == 'random_scaffold':
        smiles_list = pd.read_csv(dataset_folder + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, seed=args.seed, regression=True)
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')


    # run multiple times
    best_valid_rmse_list = []
    last_epoch_rmse_list = []
    seeds = []
    for run_idx in range(args.num_run):

        print("\nRun ", run_idx + 1)
        if args.runseed:
            runseed = args.runseed + 100 * run_idx
            seeds.append(runseed)
            print("Manual runseed: ", runseed)
        else:
            runseed = random.randint(0, 10000)
            seeds.append(runseed)
            print("Random runseed: ", runseed)

        seed_everything(runseed)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # set up model
        args.task_type = 'regression'
        model = GNN_graphpred(args.num_layer,
                              args.emb_dim,
                              num_tasks,
                              JK=args.JK,
                              drop_ratio=args.dropout_ratio,
                              graph_pooling=args.graph_pooling,
                              gnn_type=args.gnn_type)
        if not args.input_model_file == '':
            model.from_pretrained(args.input_model_file)
        model.to(device)
        # print(model)

        model_param_group = [
            {'params': model.gnn.parameters()},
            {'params': model.graph_pred_linear.parameters(), 'lr': args.lr * args.lr_scale}
        ]
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        # reg_criterion = torch.nn.L1Loss()
        reg_criterion = torch.nn.MSELoss()

        train_result_list, val_result_list, test_result_list = [], [], []
        # metric_list = ['RMSE', 'MAE', 'R2']
        metric_list = ['RMSE', 'MAE']
        best_val_rmse, best_val_test_rmse, last_epoch_test, best_val_idx = 1e10, 0, 0, 0

        for epoch in range(1, args.epochs + 1):
            loss_acc = train(model, device, train_loader, optimizer)
            print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

            if args.eval_train:
                train_result, train_target, train_pred = eval(model, device, train_loader)
            else:
                train_result = {'RMSE': 0, 'MAE': 0, 'R2': 0}
            val_result, val_target, val_pred = eval(model, device, val_loader)
            test_result, test_target, test_pred = eval(model, device, test_loader)

            train_result_list.append(train_result)
            val_result_list.append(val_result)
            test_result_list.append(test_result)

            for metric in metric_list:
                print('{} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(metric, train_result[metric], val_result[metric], test_result[metric]))
            print()

            if val_result['RMSE'] < best_val_rmse:
                best_val_rmse = val_result['RMSE']
                best_val_test_rmse = test_result['RMSE']
                best_val_idx = epoch - 1
                if not args.output_model_dir == '':
                    output_model_path = join(args.output_model_dir, 'model_best.pth')
                    saved_model_dict = {
                        'molecule_model': model.gnn.state_dict(),
                        'model': model.state_dict()
                    }
                    torch.save(saved_model_dict, output_model_path)

                    filename = join(args.output_model_dir, 'evaluation_best.pth')
                    np.savez(filename, val_target=val_target, val_pred=val_pred,
                            test_target=test_target, test_pred=test_pred)

            if epoch == args.epochs:
                last_epoch_test = test_result['RMSE']

        for metric in metric_list:
            print('Best (RMSE), {} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(
                metric, train_result_list[best_val_idx][metric], val_result_list[best_val_idx][metric], test_result_list[best_val_idx][metric]))
            
        best_valid_rmse_list.append(best_val_test_rmse)
        last_epoch_rmse_list.append(last_epoch_test)


        if args.output_model_dir is not '':
            output_model_path = join(args.output_model_dir, 'model_final.pth')
            saved_model_dict = {
                'molecule_model': model.gnn.state_dict(),
                'model': model.state_dict()
            }
            torch.save(saved_model_dict, output_model_path)


    # summarize results
    best_valid_rmse_list = np.array(best_valid_rmse_list)
    last_epoch_rmse_list = np.array(last_epoch_rmse_list)

    # if args.dataset in ["muv", "hiv"]:
    print("seeds:", seeds)
    print("best_valid_rmse_list: ", best_valid_rmse_list)
    print("last_epoch_rmse_list: ", last_epoch_rmse_list)
    print("Last epoch:")
    print("Mean: {}\tStd: {}".format(np.mean(last_epoch_rmse_list), np.std(last_epoch_rmse_list)))
    print("Best validation epoch:")
    print("Mean: {}\tStd: {}".format(np.mean(best_valid_rmse_list), np.std(best_valid_rmse_list)))

