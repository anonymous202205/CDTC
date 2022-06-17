# coding:utf-8
import torch
import torch.nn as nn
from chem_lib.models.relation import MLP
from collections import OrderedDict

class ClassIndex(nn.Module):
    def __init__(self, dataset_dict):
        super(ClassIndex, self).__init__()
        self.encode_task_with_index(dataset_dict)

    def encode_task_with_index(self, dataset_dict):
        index = 0
        datadict = OrderedDict(dataset_dict)
        cls_dict = {}

        for k in datadict:
            cls = datadict[k]
            for c in range(len(cls)):
                key = (k, c)
                cls_dict[key] = index
                index += 1
        self.clsIndex = cls_dict

    def get_cls_index(self, onedata, onecls):
        key = (onedata, onecls)
        return self.clsIndex[key]


class ClassProto(nn.Module):
    def __init__(self, inp_dim, hidden_dim, output_dim, num_layers, bef_mlp=False, pos_mlp=False,  batch_norm=False,
                 dropout=0., ctx_head=1):
        super(ClassProto, self).__init__()
        self.bef = bef_mlp
        self.pos = pos_mlp
        if self.bef:
            in_dim = inp_dim
            out_dim = hidden_dim

            self.mlp_proj1 = MLP(inp_dim=in_dim, hidden_dim=out_dim, num_layers=num_layers,
                batch_norm=batch_norm, dropout=dropout)
        if self.bef:
            hidden_dim = inp_dim
        if self.pos:
            in_dim = 2 * hidden_dim
            out_dim = output_dim
            self.mlp_proj2 = MLP(inp_dim=in_dim, hidden_dim=out_dim, num_layers=num_layers,
                                batch_norm=batch_norm, dropout=dropout)

    def forward(self, s_emb):

        if self.bef:
            s_emb = self.mlp_proj1(s_emb)

        n_support = s_emb.size(0)
        n_shot = int(n_support//2)
        pos_emb_rep = s_emb[:n_shot].mean(0).unsqueeze(0)
        neg_emb_rep = s_emb[n_shot:2*n_shot].mean(0).unsqueeze(0)
        all_emb_rep = torch.cat((pos_emb_rep, neg_emb_rep), -1)

        if self.pos:
            all_emb_rep = self.mlp_proj2(all_emb_rep)  # 16 21 600
            return all_emb_rep
        else:
            return all_emb_rep


class TargetPool(nn.Module):
    def __init__(self, inp_dim, hidden_dim, output_dim, num_layers, bef_mlp=False, pos_mlp=False,  batch_norm=False,
                 dropout=0., ctx_head=1):
        super(TargetPool, self).__init__()
        self.encoder = ClassProto(inp_dim, hidden_dim, output_dim, num_layers, bef_mlp=bef_mlp, pos_mlp=pos_mlp,
                                 batch_norm=batch_norm, dropout=dropout, ctx_head=ctx_head,)

    def encode_task_with_index(self, dataset_dict):
        index = 0
        datadict = OrderedDict(dataset_dict)
        cls_dict = {}
        for k in datadict:
            cls = datadict[k]
            for c in range(len(cls)):
                key = (k, c)
                cls_dict[key] = index
                index += 1
        clsIndex = cls_dict
        return clsIndex

    def forward(self, supOfAll, dataset_dict):
        clsIndex = self.encode_task_with_index(dataset_dict)  # get target index of cls(task)
        targetpool = {}
        for key in clsIndex:
            if key not in supOfAll:
                print('no key in support embeddings')
            else:
                emb = supOfAll[key]
                if isinstance(emb, dict):
                    emb = emb['s_data']
                code = self.encoder(emb)
                # todo ap8 skip encoder, use original embedding
                targetpool[key] = code

        return targetpool


class TargetEmb(nn.Module):
    def __init__(self, args):
        super(TargetEmb, self).__init__()
        self.test_dataset = args.test_dataset
        self.target_emb = TargetPool(inp_dim=args.map_dim, hidden_dim=args.map_dim, output_dim=args.map_dim,
                                     num_layers=args.map_layer, batch_norm=args.batch_norm,
                                     dropout=args.map_dropout, bef_mlp=args.map_pre_fc, pos_mlp=args.map_pre_fc,
                                     ctx_head=args.ctx_head)

    def forward(self, supemb, dataset_dict):
        if not isinstance(dataset_dict, dict):
            dataset_dict2 = dict({self.test_dataset: dataset_dict})
        else:
            dataset_dict2 = dataset_dict
        return self.target_emb(supemb, dataset_dict2)



