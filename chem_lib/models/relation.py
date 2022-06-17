from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers, batch_norm=False, dropout=0.):
        super(MLP, self).__init__()
        layer_list = OrderedDict()
        in_dim = inp_dim
        for l in range(num_layers):
            layer_list['fc{}'.format(l)] = nn.Linear(in_dim, hidden_dim)
            if l < num_layers - 1:
                if batch_norm:
                    layer_list['norm{}'.format(l)] = nn.BatchNorm1d(num_features=hidden_dim)
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
            in_dim = hidden_dim
        if num_layers > 0:
            self.network = nn.Sequential(layer_list)
        else:
            self.network = nn.Identity()

    def forward(self, emb):
        out = self.network(emb)
        return out


class Attention(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    """
    def __init__(self, dim, num_heads=1, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x


class RawMLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers, pre_fc=0, batch_norm=False, dropout=0., ctx_head=1,):
        super(RawMLP, self).__init__()
        self.pre_fc = pre_fc
        self.relu = nn.LeakyReLU()
        self.mlp_proj = MLP(inp_dim=inp_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            batch_norm=batch_norm, dropout=dropout)

    def forward(self, s_emb, q_emb):

        s_emb = self.mlp_proj(s_emb)
        q_emb = self.mlp_proj(q_emb)
        s_emb = self.relu(s_emb)
        q_emb = self.relu(q_emb)

        return s_emb,  q_emb


class Modulate(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers, batch_norm=False, dropout=0.,):
        super(Modulate, self).__init__()

        self.relu = nn.ReLU()

        self.mod = MLP(inp_dim=inp_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                batch_norm=batch_norm, dropout=dropout)

    def forward(self, input_emb):
        emb_m = self.mod(input_emb)
        emb_m = self.relu(emb_m)
        code = torch.mean(emb_m, dim=0)
        return code


class ModulatePN(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers, batch_norm=False, dropout=0.,):
        super(ModulatePN, self).__init__()

        self.relu = nn.ReLU()

        self.mod = MLP(inp_dim=inp_dim * 2, hidden_dim=hidden_dim, num_layers=num_layers,
                batch_norm=batch_norm, dropout=dropout)

    def forward(self, input_emb):
        n = int(input_emb.shape[0]/2)
        neg = input_emb[:n]
        pos = input_emb[n:]
        pos_mean = torch.mean(pos, dim=0)
        neg_mean = torch.mean(neg, dim=0)
        inp = torch.cat((pos_mean, neg_mean), dim=-1)
        emb_m = self.mod(inp)
        emb_m = self.relu(emb_m)
        # code = torch.mean(emb_m, dim=0)

        return emb_m




