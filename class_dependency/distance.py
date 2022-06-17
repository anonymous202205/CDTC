import torch
from collections import OrderedDict
def euclidean_dist(x, y, transform=True):
    bs = x.shape[0]
    if transform:
        num_proto = y.shape[0]
        query_lst = []
        for i in range(bs):
            ext_query = x[i, :].repeat(num_proto, 1)
            query_lst.append(ext_query)
        x = torch.cat(query_lst, dim=0)
        y = y.repeat(bs, 1)

    return torch.pow(x - y, 2).sum(-1)


def dict_dist(dict1, dict2, transform=True):
    key1s = list(OrderedDict(dict1).keys())
    key2s = list(OrderedDict(dict2).keys())

    row = range(len(key1s))
    col = range(len(key2s))
    adj = torch.zeros((len(key1s), len(key2s)))
    # odict_keys' object does not support indexing
    for i in row:
        for j in col:
            key1 = key1s[i]
            key2 = key2s[j]
            value = euclidean_dist(dict1[key1], dict2[key2])
            adj[i, j] = value
    ori1 = key1s
    ori2 = key2s
    return adj, ori1, ori2


def dict_dist2(target1, dict2, transform=True):
    key1s = list(OrderedDict(target1).keys())
    key2s = list(OrderedDict(dict2).keys())

    values1 = list(target1.values())
    values1 = torch.stack(values1, dim=0)
    values1 = torch.mean(values1, dim=0)

    row = range(1)
    col = range(len(key2s))
    adj = torch.zeros((len(row), len(col)))
    for i in row:
        for j in col:
            # key1 = key1s[i]
            key2 = key2s[j]
            value = euclidean_dist(values1, dict2[key2])
            adj[i, j] = value
    ori1 = key1s
    ori2 = key2s
    return adj, ori1, ori2
