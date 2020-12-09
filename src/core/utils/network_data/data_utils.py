# https://github.com/tkipf/gcn/blob/master/gcn/utils.py
import os
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
import torch


from ..generic_utils import *
from ..constants import VERY_SMALL_NUMBER


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data(data_dir, dataset_str, knn_size=None, epsilon=None, knn_metric='cosine', prob_del_edge=None, prob_add_edge=None, seed=1234, sparse_init_adj=False):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name ('cora', 'citeseer', 'pubmed')
    :return: All data input files loaded (as well the training/test data).
    """
    assert (knn_size is None) or (epsilon is None)

    if dataset_str.startswith('ogbn'): # Open Graph Benchmark datasets
        from ogb.nodeproppred import NodePropPredDataset

        dataset = NodePropPredDataset(name=dataset_str)

        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = torch.LongTensor(split_idx["train"]), torch.LongTensor(split_idx["valid"]), torch.LongTensor(split_idx["test"])

        data = dataset[0] # This dataset has only one graph
        features = torch.Tensor(data[0]['node_feat'])
        labels = torch.LongTensor(data[1]).squeeze(-1)

        edge_index = data[0]['edge_index']
        adj = to_undirected(edge_index, num_nodes=data[0]['num_nodes'])
        assert adj.diagonal().sum() == 0 and adj.max() <= 1 and (adj != adj.transpose()).sum() == 0


    else: # datasets: Cora, Citeseer, PubMed

        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(os.path.join(data_dir, 'ind.{}.{}'.format(dataset_str, names[i])), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(os.path.join(data_dir, 'ind.{}.test.index'.format(dataset_str)))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        raw_features = sp.vstack((allx, tx)).tolil()
        raw_features[test_idx_reorder, :] = raw_features[test_idx_range, :]
        features = normalize_features(raw_features)
        raw_features = torch.Tensor(raw_features.todense())
        features = torch.Tensor(features.todense())

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))


        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        # labels = torch.LongTensor(np.where(labels)[1])
        labels = torch.LongTensor(np.argmax(labels, axis=1))

        idx_train = torch.LongTensor(range(len(y)))
        idx_val = torch.LongTensor(range(len(y), len(y) + 500))
        idx_test = torch.LongTensor(test_idx_range.tolist())



    if not knn_size is None:
        print('[ Using KNN-graph as input graph: {} ]'.format(knn_size))
        adj = kneighbors_graph(features, knn_size, metric=knn_metric, include_self=True)
        adj_norm = normalize_sparse_adj(adj)
        if sparse_init_adj:
            adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        else:
            adj_norm = torch.Tensor(adj_norm.todense())

    elif not epsilon is None:
        print('[ Using Epsilon-graph as input graph: {} ]'.format(epsilon))
        feature_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
        attention = torch.mm(feature_norm, feature_norm.transpose(-1, -2))
        mask = (attention > epsilon).float()
        adj = attention * mask
        adj = (adj > 0).float()
        adj = sp.csr_matrix(adj)
        adj_norm = normalize_sparse_adj(adj)
        if sparse_init_adj:
            adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        else:
            adj_norm = torch.Tensor(adj_norm.todense())

    else:
        print('[ Using ground-truth input graph ]')

        if prob_del_edge is not None:
            adj = graph_delete_connections(prob_del_edge, seed, adj.toarray(), enforce_connected=False)
            adj = adj + np.eye(adj.shape[0])
            adj_norm = normalize_adj(torch.Tensor(adj))
            adj_norm = sp.csr_matrix(adj_norm)


        elif prob_add_edge is not None:
            adj = graph_add_connections(prob_add_edge, seed, adj.toarray(), enforce_connected=False)
            adj = adj + np.eye(adj.shape[0])
            adj_norm = normalize_adj(torch.Tensor(adj))
            adj_norm = sp.csr_matrix(adj_norm)

        else:
            adj = adj + sp.eye(adj.shape[0])
            adj_norm = normalize_sparse_adj(adj)


        if sparse_init_adj:
            adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        else:
            adj_norm = torch.Tensor(adj_norm.todense())

    return adj_norm, features, labels, idx_train, idx_val, idx_test


def graph_delete_connections(prob_del, seed, adj, enforce_connected=False):
    rnd = np.random.RandomState(seed)

    del_adj = np.array(adj, dtype=np.float32)
    pre_num_edges = np.sum(del_adj)

    smpl = rnd.choice([0., 1.], p=[prob_del, 1. - prob_del], size=adj.shape) * np.triu(np.ones_like(adj), 1)
    smpl += smpl.transpose()

    del_adj *= smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(del_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                del_adj[k, other_node] = 1
                del_adj[other_node, k] = 1
                add_edges += 1
        print('# ADDED EDGES: ', add_edges)

    cur_num_edges = np.sum(del_adj)
    print('[ Deleted {}% edges ]'.format(100 * (pre_num_edges - cur_num_edges) / pre_num_edges))
    return del_adj


def graph_add_connections(prob_add, seed, adj, enforce_connected=False):
    rnd = np.random.RandomState(seed)

    add_adj = np.array(adj, dtype=np.float32)

    smpl = rnd.choice([0., 1.], p=[1. - prob_add, prob_add], size=adj.shape) * np.triu(np.ones_like(adj), 1)
    smpl += smpl.transpose()

    add_adj += smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(add_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                add_adj[k, other_node] = 1
                add_adj[other_node, k] = 1
                add_edges += 1
        print('# ADDED EDGES: ', add_edges)
    add_adj = (add_adj > 0).astype(float)
    return add_adj

