import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.generic_utils import to_cuda, normalize_adj
from ..utils.constants import VERY_SMALL_NUMBER, INF


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


class AnchorGraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, topk=None, epsilon=None, num_pers=16, metric_type='attention', device=None):
        super(AnchorGraphLearner, self).__init__()
        self.device = device
        self.topk = topk
        self.epsilon = epsilon
        self.metric_type = metric_type
        if metric_type == 'attention':
            self.linear_sims = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False) for _ in range(num_pers)])
            print('[ Multi-perspective {} AnchorGraphLearner: {} ]'.format(metric_type, num_pers))

        elif metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
            print('[ Multi-perspective {} AnchorGraphLearner: {} ]'.format(metric_type, num_pers))


        elif metric_type == 'gat_attention':
            self.linear_sims1 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])

            self.leakyrelu = nn.LeakyReLU(0.2)

            print('[ GAT_Attention AnchorGraphLearner]')

        elif metric_type == 'kernel':
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(input_size, hidden_size)))
        elif metric_type == 'transformer':
            self.linear_sim1 = nn.Linear(input_size, hidden_size, bias=False)
            self.linear_sim2 = nn.Linear(input_size, hidden_size, bias=False)


        elif metric_type == 'cosine':
            pass

        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))

        print('[ Graph Learner metric type: {} ]'.format(metric_type))

    def forward(self, context, anchors, ctx_mask=None, anchor_mask=None):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)

        Returns
        :attention, (batch_size, ctx_size, ctx_size)
        """
        if self.metric_type == 'attention':
            attention = 0
            for _ in range(len(self.linear_sims)):
                context_fc = torch.relu(self.linear_sims[_](context))
                attention += torch.matmul(context_fc, context_fc.transpose(-1, -2))

            attention /= len(self.linear_sims)
            markoff_value = -INF

        elif self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)

            anchors_fc = anchors.unsqueeze(0) * expand_weight_tensor
            anchors_norm = F.normalize(anchors_fc, p=2, dim=-1)

            attention = torch.matmul(context_norm, anchors_norm.transpose(-1, -2)).mean(0)
            markoff_value = 0


        elif self.metric_type == 'transformer':
            Q = self.linear_sim1(context)
            attention = torch.matmul(Q, Q.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
            markoff_value = -INF


        elif self.metric_type == 'gat_attention':
            attention = []
            for _ in range(len(self.linear_sims1)):
                a_input1 = self.linear_sims1[_](context)
                a_input2 = self.linear_sims2[_](context)
                attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

            attention = torch.mean(torch.stack(attention, 0), 0)
            markoff_value = -INF


        elif self.metric_type == 'kernel':
            dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
            attention = self.compute_distance_mat(context, dist_weight)
            attention = torch.exp(-0.5 * attention * (self.precision_inv_dis**2))

            markoff_value = 0

        elif self.metric_type == 'cosine':
            context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
            attention = torch.mm(context_norm, context_norm.transpose(-1, -2)).detach()
            markoff_value = 0


        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if anchor_mask is not None:
            attention = attention.masked_fill_(1 - anchor_mask.byte().unsqueeze(-2), markoff_value)

        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        if self.topk is not None:
            attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)

        return attention

    def build_knn_neighbourhood(self, attention, topk, markoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = to_cuda((markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val), self.device)
        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        return dists


def get_binarized_kneighbors_graph(features, topk, mask=None, device=None):
    assert features.requires_grad is False
    # Compute cosine similarity matrix
    features_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
    attention = torch.matmul(features_norm, features_norm.transpose(-1, -2))

    if mask is not None:
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(1), 0)
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(-1), 0)

    # Extract and Binarize kNN-graph
    topk = min(topk, attention.size(-1))
    _, knn_ind = torch.topk(attention, topk, dim=-1)
    adj = to_cuda(torch.zeros_like(attention).scatter_(-1, knn_ind, 1), device)
    return adj
