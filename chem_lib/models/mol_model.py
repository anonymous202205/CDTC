import torch
import torch.nn as nn

from .encoder import GNN_Encoder
from .relation import RawMLP


class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x


class ContextAwareRelationNet(nn.Module):
    def __init__(self, args):
        super(ContextAwareRelationNet, self).__init__()
        # self.rel_layer = args.rel_layer
        self.edge_type = args.rel_adj
        self.edge_activation = args.rel_act
        self.gpu_id = args.gpu_id

        self.mol_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm = args.enc_batch_norm)
        if args.pretrained:
            model_file = args.pretrained_weight_path
            if args.enc_gnn != 'gin':
                temp = model_file.split('/')
                model_file = '/'.join(temp[:-1]) +'/'+args.enc_gnn +'_'+ temp[-1]
            self.mol_encoder.from_pretrained(model_file, self.gpu_id)

        self.encode_projection = RawMLP(inp_dim=args.emb_dim, hidden_dim=args.map_dim, num_layers=2,
                                        batch_norm=args.batch_norm, dropout=args.map_dropout,
                                        pre_fc=args.map_pre_fc, ctx_head=args.ctx_head)

        num_class = 2
        self.cls = nn.Linear(args.map_dim, num_class)

    def to_one_hot(self, class_idx, num_classes=2):
        return torch.eye(num_classes)[class_idx].to(class_idx.device)

    def label2edge(self, label, mask_diag=True):
        num_samples = label.size(1)
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        edge = torch.eq(label_i, label_j).float().to(label.device)

        edge = edge.unsqueeze(1)
        if self.edge_type == 'dist':
            edge = 1 - edge

        if mask_diag:
            diag_mask = 1.0 - torch.eye(edge.size(2)).unsqueeze(0).unsqueeze(0).repeat(edge.size(0), 1, 1, 1).to(edge.device)
            edge = edge*diag_mask
        if self.edge_activation == 'softmax':
            edge = edge / edge.sum(-1).unsqueeze(-1)
        return edge

    def forward(self, s_data, q_data, s_label=None, q_pred_adj=False, stflag=0):
        s_emb, s_node_emb = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        q_emb, q_node_emb = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)

        s_emb_map, q_emb_map = self.encode_projection(s_emb, q_emb)
        s_logits, q_logits = self.cls(s_emb_map), self.cls(q_emb_map)

        adj = None
        return s_logits, q_logits, adj, s_node_emb

    def forward_query_list(self, s_data, q_data_list, s_label=None, q_pred_adj=False, stflag=1):
        s_emb, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        q_emb_list = [self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)[0] for q_data in
                      q_data_list]

        q_logits_list, adj_list = [], []
        for q_emb in q_emb_list:
            s_emb_map, q_emb_map = self.encode_projection(s_emb, q_emb)
            s_logit, q_logit = self.cls(s_emb_map), self.cls(q_emb_map)

            adj = None
            q_logits_list.append(q_logit.detach())
            if adj is not None:
                sim_adj = adj[-1][:, 0].detach()
                q_adj = sim_adj[:, -1]
                adj_list.append(q_adj)

        q_logits = torch.cat(q_logits_list, 0)
        if len(adj_list) > 0:
            adj_list = torch.cat(adj_list, 0)
        return s_logit.detach(), q_logits, adj_list

    def forward_query_loader(self, s_data, q_loader, s_label=None, q_pred_adj=False, stflag=1):
        s_emb, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        y_true_list=[]
        q_logits_list, adj_list = [], []
        for q_data in q_loader:
            q_data = q_data.to(s_emb.device)
            y_true_list.append(q_data.y)
            q_emb,_ = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)
            s_emb_map, q_emb_map = self.encode_projection(s_emb,q_emb)
            s_logit, q_logit = self.cls(s_emb_map), self.cls(q_emb_map)
            adj = None
            q_logits_list.append(q_logit)
            if adj is not None:
                sim_adj = adj[-1].detach()
                adj_list.append(sim_adj)

        q_logits = torch.cat(q_logits_list, 0)
        y_true = torch.cat(y_true_list, 0)
        sup_labels = {'support':s_data.y, 'query':y_true_list}
        return s_logit, q_logits, y_true, adj_list, sup_labels
