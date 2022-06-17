import os
import argparse
import random
import numpy as np
import torch

from chem_lib.datasets import obatin_train_test_tasks
from chem_lib.utils import init_trial_path
from chem_lib.datasets import obtain_train_test_tasks_ms


def get_parser(root_dir='.'):
    parser = argparse.ArgumentParser(description='Property-Aware Relation Networks for Few-Shot Molecular Property Prediction')
    parser.add_argument('-r', '--root-dir', type=str, default=root_dir, help='root-dir')
    parser.add_argument('-d', '--dataset', type=str, default='all', help='data set name')  # ['tox21','sider','muv','toxcast']
    parser.add_argument('-td', '--test_dataset', type=str, default='tox21',
                        help='test data set name')  # ['tox21','sider','muv','toxcast']
    parser.add_argument('--data_dir', type=str, default=os.path.join(root_dir,'data') +'/', help='data dir')
    parser.add_argument('--preload_train_data', type=bool, default=True)  # 0
    parser.add_argument('--preload_test_data', type=bool, default=True)  # 0
    parser.add_argument("--run_task", type=int, default=-1, help="run on task")

    parser.add_argument("--n_shot_train", type=int, default=10, help="train: number of shot for each class")#choices=[1,10]
    parser.add_argument("--n_shot_test", type=int, default=10, help="test: number of shot for each class")#choices=[1,10]
    parser.add_argument("--n-query", type=int, default=16, help="number of query in few shot learning")

    parser.add_argument("--meta_lr", type=float, default=0.0005,  # 0.003, 0.001, 0.0006
                        help="Training: Meta learning rate")
    parser.add_argument("--agentlr", type=float, default=0.05,  # 0.003, 0.001, 0.0006
                        help="Training: Meta learning rate")
    parser.add_argument("--rl_adv", type=float, default=0.05,  # 0.003, 0.001, 0.0006
                        help="Training: Meta learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-5,
                        help="Training: Meta learning weight_decay")  
    parser.add_argument("--inner_lr", type=float, default=0.005, help="Training: Inner loop learning rate")  # 0.01
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of epochs to train (default: 5000)')  # 5000
    parser.add_argument('--update_step', type=int, default=1)  # 5
    parser.add_argument('--update_step_test', type=int, default=4)  # 10
    parser.add_argument('--inner_update_step', type=int, default=1)  # 10
    parser.add_argument("--meta_warm_step", type=int, default=0, help="meta warp up step for encode")  # 9
    parser.add_argument("--meta_warm_step2", type=int, default=10000, help="meta warp up step 2 for encode")
    parser.add_argument("--second_order", type=int, default=1, help="second order or not")  # 9
    parser.add_argument("--batch_task", type=int, default=9, help="Training: Meta batch size")  # 9
    parser.add_argument("--adapt_weight", type=int, default=5, help="adaptable weights")  # 9
    parser.add_argument("--eval_support", type=int, default=0, help="Training: eval s")

    parser.add_argument('--enc_gnn', type=str, default="gin")  # choices=["gin", "gcn", "gat", "graphsage"]
    parser.add_argument('--enc_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--enc_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--enc_batch_norm', type=int, default=1,
                        help='use batch norm or not')
    parser.add_argument('--pretrained', type=int, default=1, help='pretrained or not')
    parser.add_argument('--pretrained_weight_path', type=str,
                        default=os.path.join(root_dir, 'chem_lib/model_gin/supervised_contextpred.pth'), help='pretrained path')

    parser.add_argument('--map_dim', type=int, default=128, help='map dimensions ')
    parser.add_argument('--map_layer', type=int, default=2, help='map layer ')
    parser.add_argument('--map_pre_fc', type=int, default=0, help='pre fc layer')
    parser.add_argument("--map-dropout", type=float, default=0.1, help="map dropout")
    # relation graph
    parser.add_argument('--ctx_head', type=int, default=2, help='context layer')

    parser.add_argument("--rel-hidden-dim", type=int, default=128, help="hidden dim for relation net")  # 32
    parser.add_argument("--rel-layer", type=int, default=2, help="number of layers for relation net")
    parser.add_argument("--rel-edge-layer", type=int, default=2, help="number of layers for relation edge update")  # 3
    parser.add_argument("--rel-res", type=float, default=0, help="residual weight of mapper and relation")
    parser.add_argument("--batch_norm", type=int, default=0, help="batch_norm or not")
    parser.add_argument("--rel_adj", type=str, default='sim', choices=['dist', 'sim'], help="edge update adjacent")
    parser.add_argument("--rel_act", type=str, default='sigmoid', choices=['sigmoid', 'softmax', 'none'],
                        help="edge update adjacent")
    parser.add_argument('--rel_node_concat', type=int, default=0, help='node concat or not')
    parser.add_argument("--rel-dropout", type=float, default=0, help="rel dropout")
    parser.add_argument("--rel-dropout2", type=float, default=0.2, help="rel dropout2")

    parser.add_argument('--reg_adj', type=float, default=0.0, help='reg adj loss weight')
    parser.add_argument('--seed', type=int, default=5, help="Seed for splitting the dataset.")
    parser.add_argument('--gpu_id', type=int, default=0, help="Choose the number of GPU.")
    parser.add_argument("--result_path", type=str, default=os.path.join(root_dir,'results'), help="result path")
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=2000, help="Training: Number of iterations between checkpoints")
    parser.add_argument("--save_logs", type=int, default=0)
    parser.add_argument("--support_valid", type=int, default=0)

    parser.add_argument('-d_names', '--d_names', help='delimited list input', type=str, default='muv-tox21')
    parser.add_argument('--applyrl', action='store_true',default=False)
    parser.add_argument('--maml', action='store_true',default=False)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--naivedim", type=int, default=27)
    parser.add_argument("--n_query_test", type=int, default=512)
    parser.add_argument("--pi", type=str, default='graph')
    parser.add_argument('--abs', action='store_true',default=False)
    parser.add_argument("--bs", type=str, default='')
    parser.add_argument("--step_gin1", type=int, default=10)
    parser.add_argument("--step_agent", type=int, default=500)
    return parser

def get_args(root_dir='.', is_save=True):
    parser = get_parser(root_dir)
    args = parser.parse_args()

    args.rel_k= args.n_shot_train
    if args.pretrained:
        args.enc_layer = 5
        args.emb_dim = 300
        args.dropout = 0.5 
    if args.enc_layer <= 3:
        args.emb_dim = 200
        args.dropout = 0.1
    alllist = ['muv', 'tox21', 'toxcast', 'sider']

    if args.map_layer <= 0:
        args.map_dim = args.emb_dim
    args = init_trial_path(args, is_save)
    device = "cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu"
    args.device = device

    train_tasks_dict, test_tasks_dict, real_task_map = obtain_train_test_tasks_ms(args.dataset, args.test_dataset, args.d_names)
    args.train_dataset_task_dict = train_tasks_dict
    args.test_dataset_task = test_tasks_dict
    args.real_task_map = real_task_map

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    return args
