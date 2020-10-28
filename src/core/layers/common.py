'''
Created on Nov, 2018

@author: hugo

'''
from typing import List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from ..utils.generic_utils import to_cuda
from ..utils.constants import VERY_SMALL_NUMBER


def dropout(x, drop_prob, shared_axes=[], training=False):
    """
    Apply dropout to input tensor.
    Parameters
    ----------
    input_tensor: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
    Returns
    -------
    output: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)`` with dropout applied.
    """
    if drop_prob == 0 or drop_prob == None or (not training):
        return x

    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1. - drop_prob).div_(1. - drop_prob)
    mask = mask.expand_as(x)
    return x * mask

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, \
        bidirectional=False, num_layers=1, rnn_type='lstm', rnn_dropout=None, device=None):
        super(EncoderRNN, self).__init__()
        if not rnn_type in ('lstm', 'gru'):
            raise RuntimeError('rnn_type is expected to be lstm or gru, got {}'.format(rnn_type))
        if bidirectional:
            print('[ Using {}-layer bidirectional {} encoder ]'.format(num_layers, rnn_type))
        else:
            print('[ Using {}-layer {} encoder ]'.format(num_layers, rnn_type))
        if bidirectional and hidden_size % 2 != 0:
            raise RuntimeError('hidden_size is expected to be even in the bidirectional mode!')
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout
        self.device = device
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_directions = 2 if bidirectional else 1
        model = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.model = model(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """x: [batch_size * max_length * emb_dim]
           x_len: [batch_size]
        """
        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        x = pack_padded_sequence(x[indx], sorted_x_len.data.tolist(), batch_first=True)

        h0 = to_cuda(torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size), self.device)
        if self.rnn_type == 'lstm':
            c0 = to_cuda(torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size), self.device)
            packed_h, (packed_h_t, packed_c_t) = self.model(x, (h0, c0))
        else:
            packed_h, packed_h_t = self.model(x, h0)

        if self.num_directions == 2:
            packed_h_t = torch.cat((packed_h_t[-1], packed_h_t[-2]), 1)
            if self.rnn_type == 'lstm':
                packed_c_t = torch.cat((packed_c_t[-1], packed_c_t[-2]), 1)
        else:
            packed_h_t = packed_h_t[-1]
            if self.rnn_type == 'lstm':
                packed_c_t = packed_c_t[-1]

        # restore the sorting
        _, inverse_indx = torch.sort(indx, 0)

        hh, _ = pad_packed_sequence(packed_h, batch_first=True)
        restore_hh = hh[inverse_indx]
        restore_hh = dropout(restore_hh, self.rnn_dropout, shared_axes=[-2], training=self.training)
        restore_hh = restore_hh.transpose(0, 1) # [max_length, batch_size, emb_dim]

        restore_packed_h_t = packed_h_t[inverse_indx]
        restore_packed_h_t = dropout(restore_packed_h_t, self.rnn_dropout, training=self.training)
        restore_packed_h_t = restore_packed_h_t.unsqueeze(0) # [1, batch_size, emb_dim]

        if self.rnn_type == 'lstm':
            restore_packed_c_t = packed_c_t[inverse_indx]
            restore_packed_c_t = dropout(restore_packed_c_t, self.rnn_dropout, training=self.training)
            restore_packed_c_t = restore_packed_c_t.unsqueeze(0) # [1, batch_size, emb_dim]
            rnn_state_t = (restore_packed_h_t, restore_packed_c_t)
        else:
            rnn_state_t = restore_packed_h_t
        return restore_hh, rnn_state_t
