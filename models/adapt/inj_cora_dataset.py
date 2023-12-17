from models.utils import load_data
from torch_geometric.utils.convert import to_networkx

import random
from typing import Callable, Optional
import networkx as nx
import torch
import numpy as np
from torch_geometric.utils import dropout_adj
from torch_geometric.utils.convert import (from_networkx)
from torch_geometric.data import (
    Data,
    InMemoryDataset,
)

from scipy.spatial.distance import cosine
import os.path as osp
from sklearn.decomposition import PCA

def similarity (features):
    similarities = np.zeros((features.shape[0], features.shape[0, 1]))
    for i in range(features.shape[0]):
        for j in range(i, features.shape[0]):
            similarities[i, j] = 1 - cosine(features[i], features[j])
            similarities[j, i] = similarities[i, j]
    return similarities

def edge_rewiring(data):
    prob = similarity(data.x)
    edge_index, edge_attr = dropout_adj(edge_index=data.edge_index, edge_attr=data.edge_attr, p=prob, num_nodes=data.x.size(0))
    edge_index, edge_attr = dropout_adj(edge_index=edge_index, edge_attr=edge_attr, p=1-prob, num_nodes=data.x.size(0))
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    return data

def feature_exchange(data):
    prob = similarity(data.x)
    slt = torch.rand(2).reshape(2, 1)
    _, info = torch.sort(torch.cat((prob, slt)), dim=0)
    j = 0
    for i in range(info.shape[0] - 1, -1, -1):
        if info[i] >= 2:
            slt[j] = info[i + 1]
            info[i] = info[i + 1]
            j = j + 1
    choose = torch.randint(slt.shape[0], (2, 1))
    res = torch.zeros_like(slt)
    for i in range(0, 2, 2):
        j = random.randint(0, len(data.x) - 2)
        res[i] = torch.cat((slt[choose[i], 0:j], slt[choose[i + 1], j:len(data.x)]), dim=1)
        res[i + 1] = torch.cat((slt[choose[i + 1], 0:j], slt[choose[i], j:len(data.x)]), dim=1)
    drop_mask = torch.empty(size=(data.x.size(1),), dtype=torch.float32).uniform_(0, 1) < prob
    data.x[:, drop_mask] = 0
    return data



def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def G_PCA(G,nfeat):
    feature = np.zeros((G.number_of_nodes(), nfeat))
    graph_label = np.zeros((G.number_of_nodes(), 1))

    for i, node in enumerate(G.nodes):
        if G.nodes[node]['y'] > 0:
            graph_label[i] = 1

        feature[i] = np.array(G.nodes[node]['x'])

    pca = PCA(n_components=100)  # 实例化 用 mle算出来是614
    pca = pca.fit(feature)  # 拟合模型
    x_new = pca.transform(feature)

    return x_new

# 抽取2阶子图
def find123Nei(G, node, thres):
    nei1_li = []
    nei2_li = []
    nei3_li = []
    if thres == 1:
        return [node]

    for FNs in list(G.neighbors(node)):
        nei1_li.append(FNs)
    if len(nei1_li) > thres:
        nei = set (random.sample(nei1_li,thres-1) + [node])
        return nei

    for n1 in nei1_li:
        for SNs in list(G.neighbors(n1)):
            nei2_li.append(SNs)
        nei2 = nei1_li + nei2_li
        if len(nei2) > thres:
            nei = set(random.sample(nei2, thres - 1) + [node])
            return nei

    for n2 in nei2_li:
        for TNs in nx.neighbors(G, n2):
            nei3_li.append(TNs)


    nei3 = nei1_li + nei2_li + nei3_li
    if len(nei3) > thres:

        nei = set (random.sample(nei3,thres-1) + [node])
    else:
        nei = set( nei3 + [node] )

    return nei


class InjCoraDataset(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        edge_window_size (int, optional): The window size for the existence of
            an edge in the graph sequence since its initial creation.
            (default: :obj:`10`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """


    def __init__(self, root= None, thres: int = 50,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.thres = thres
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'soc-sign-bitcoinotc.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return self.data.edge_index.max().item() + 1


    def process(self):

        # read raw data from Pygod
        raw_data = load_data(name='inj_amazon')  # input a specific domain
        H = to_networkx(raw_data, node_attrs=["x", 'y'])  # x - node_feature; y - node label
        G = H.to_undirected()

        center_nodes = list(G)

        # preprocess data
        data_list = []
        flag = 0
        for n in center_nodes:
            sub_list = find123Nei(G, n, 30)
            u_graph = nx.subgraph(G, sub_list).copy()
            temp = from_networkx(u_graph)
            x = G_PCA(u_graph, 500)
            edge_index = temp.edge_index
            yc = G.nodes[n]['y'] >> 0 & 1
            ys = G.nodes[n]['y'] >> 1 & 1
            label = 1

            y = torch.zeros(1,3)
            if G.nodes[n]['y'] == 0:
                y[0][0] = 1
                label = 0
            elif yc == 1:
                y[0][1] = 1
            elif ys == 1:
                y[0][2] = 1


            # Data Augmentation
            if flag < 3 and flag > 0 and label == 0: # negative sampling
                data = Data(x=x, emb=x, edge_index=edge_index, y=y, label=label, yc=yc, ys=ys, node_id=n)
                data_list.append(data)
                flag += 1
            if label == 1:
                data = Data(x=x, emb = x, edge_index=edge_index, y=y, label = label, yc=yc, ys=ys, node_id = n)
                data_list.append(feature_exchange(data))
                data_list.append(edge_rewiring(data))
                data_list.append(data)
                flag = 1


        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])