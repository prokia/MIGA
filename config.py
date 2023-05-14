
import argparse

# Training settings
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--distributed', default=False, type=bool)


parser.add_argument('--cnn_model_name', type=str, default='resnet34',
                    help='cnn_model_name (default: resnet34)')
parser.add_argument('--gnn_type', type=str, default="gin")
parser.add_argument('--dataset', type=str, default = 'omics', help='root directory of dataset. For now, only classification.')
parser.add_argument('--task_type', type=str, default='classification')

parser.add_argument('--retrieval', type=str, default='image',
                    help='retrieval mode (default: image)')
parser.add_argument('--cDNA', type=int, default=0,
                    help='if use for cDNA (default: 0)')
parser.add_argument('--margin', type=float, default=4.0, help='margin in contrastive loss')

parser.add_argument('--nimages', type=int, default=5,
                    help='input image number of per compound')
parser.add_argument('--batch_size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=101,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='learning rate (default: 0.001)')
parser.add_argument("--lr_scale", type=float, default=1, help="relative learning rate for the feature extraction layer (default: 1)")
parser.add_argument('--decay', type=float, default=0,
                    help='weight decay (default: 0)')
parser.add_argument('--num_layer', type=int, default=5,
                    help='number of GNN message passing layers (default: 5).')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='embedding dimensions (default: 300)')
parser.add_argument("--graph_pooling", type=str, default="mean", help="graph level pooling (sum, mean, max, set2set, attention)")
parser.add_argument('--dropout_ratio', type=float, default=0.2,
                    help='dropout ratio (default: 0.2)')
parser.add_argument('--cnn_lr_scale', type=float, default=1)
parser.add_argument('--gnn_lr_scale', type=float, default=1)


parser.add_argument('--JK', type=str, default="last",
                    help='how the node features across layers are combined. last, sum, max or concat')
parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
parser.add_argument('--num_workers', type=int, default = 32, help='number of workers for dataset loading')

#alpha
parser.add_argument('--alpha_1', type=float, default = 1)
parser.add_argument('--alpha_2', type=float, default = 0)
parser.add_argument('--alpha_3', type=float, default = 0)

#loss
parser.add_argument('--T', type=float, default=0.1)
parser.add_argument('--normalize', dest='normalize', action='store_true')

# for Graph-Image Generator
parser.add_argument('--generator_loss', type=str, default='l2', choices=['l1', 'l2', 'cosine'])
parser.add_argument('--detach_target', dest='detach_target', action='store_true')
parser.add_argument('--no_detach_target', dest='detach_target', action='store_false')
parser.set_defaults(detach_target=True)
parser.add_argument('--beta', type=float, default=1)

# MGM
parser.add_argument('--use_image', type=bool, default=False)
parser.add_argument('--MGM_mode', type=str, default='AM')
parser.add_argument('--mask_rate', type=float, default=0.15)
parser.add_argument('--mask_edge', type=int, default=0)
parser.add_argument('--csize', type=int, default=3)
parser.add_argument('--contextpred_neg_samples', type=int, default=1)


# about if we would print out eval metric for training data
parser.add_argument('--eval_train', dest='eval_train', action='store_true')
parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
parser.set_defaults(eval_train=True)

# about loading and saving
parser.add_argument('--input_model_file', type=str, default='')
parser.add_argument('--output_model_dir', type=str, default='')

# verbosity
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no_verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=False)

# Finetune
parser.add_argument("--num_run", type=int, default=5, help="number of independent runs (default: 5)")
parser.add_argument("--frozen", action="store_true", default=False, help="whether to freeze gnn extractor")
parser.add_argument("--model_name", type=str, default="gnn")
parser.add_argument("--runseed", type=int, default=0, help="Seed for minibatch selection, random initialization.")
parser.add_argument("--split", type=str, default="ogb_scaffold", help="random or scaffold or random_scaffold")

parser.add_argument("--pretrain_model_path", type=str, default="")
parser.add_argument("--pretrain_cfg_path", type=str, default="")
args = parser.parse_args()
