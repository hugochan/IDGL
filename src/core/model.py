import os
import random
import numpy as np
from collections import Counter
from sklearn.metrics import r2_score


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from .models.graph_clf import GraphClf
from .models.text_graph import TextGraphRegression, TextGraphClf
from .utils.text_data.vocab_utils import VocabModel
from .utils import constants as Constants
from .utils.generic_utils import to_cuda, create_mask
from .utils.constants import INF
from .utils.radam import RAdam


class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """
    def __init__(self, config, train_set=None):
        self.config = config
        if self.config['model_name'] == 'GraphClf':
            self.net_module = GraphClf
        elif self.config['model_name'] == 'TextGraphRegression':
            self.net_module = TextGraphRegression
        elif self.config['model_name'] == 'TextGraphClf':
            self.net_module = TextGraphClf
        else:
            raise RuntimeError('Unknown model_name: {}'.format(self.config['model_name']))
        print('[ Running {} model ]'.format(self.config['model_name']))

        if config['data_type'] == 'text':
            saved_vocab_file = os.path.join(config['data_dir'], '{}_seed{}.vocab'.format(config['dataset_name'], config.get('data_seed', 1234)))
            self.vocab_model = VocabModel.build(saved_vocab_file, train_set, self.config)

        if config['task_type'] == 'regression':
            assert config['out_predictions']
            self.criterion = F.mse_loss
            self.score_func = r2_score
            self.metric_name = 'r2'
        elif config['task_type'] == 'classification':
            self.criterion = F.nll_loss
            self.score_func = accuracy
            self.metric_name = 'acc'
        else:
            self.criterion = F.nll_loss
            self.score_func = None
            self.metric_name = None



        if self.config['pretrained']:
            self.init_saved_network(self.config['pretrained'])
        else:
            # Building network.
            self._init_new_network()

        num_params = 0
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()

        print('#Parameters = {}\n'.format(num_params))
        self._init_optimizer()


    def init_saved_network(self, saved_dir):
        _ARGUMENTS = ['word_embed_dim', 'hidden_size', 'f_qem', 'f_pos', 'f_ner',
                      'word_dropout', 'rnn_dropout',
                      'ctx_graph_hops', 'ctx_graph_topk',
                      'score_unk_threshold', 'score_yes_threshold',
                      'score_no_threshold']

        # Load all saved fields.
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        self.state_dict = saved_params['state_dict']
        # for k in _ARGUMENTS:
        #     if saved_params['config'][k] != self.config[k]:
        #         print('Overwrite {}: {} -> {}'.format(k, self.config[k], saved_params['config'][k]))
        #         self.config[k] = saved_params['config'][k]

        if self.config['data_type'] == 'text':
            w_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'])
            self.network = self.net_module(self.config, w_embedding, self.vocab_model.word_vocab)
        else:
            self.network = self.net_module(self.config)

        # Merge the arguments
        if self.state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in self.state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

    def _init_new_network(self):
        if self.config['data_type'] == 'text':
            w_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'],
                                               pretrained_vecs=self.vocab_model.word_vocab.embeddings)
            self.network = self.net_module(self.config, w_embedding, self.vocab_model.word_vocab)
        else:
            self.network = self.net_module(self.config)

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters, lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'radam':
            self.optimizer = RAdam(parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['lr_reduce_factor'], \
                    patience=self.config['lr_patience'], verbose=True)

    def _init_embedding(self, vocab_size, embed_size, pretrained_vecs=None):
        """Initializes the embeddings
        """
        return nn.Embedding(vocab_size, embed_size, padding_idx=0,
                            _weight=torch.from_numpy(pretrained_vecs).float()
                            if pretrained_vecs is not None else None)

    def save(self, dirname):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'config': self.config,
            'dir': dirname,
        }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')


    def clip_grad(self):
        # Clip gradients
        if self.config['grad_clipping']:
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])

def train_batch(batch, network, vocab, criterion, forcing_ratio, rl_ratio, config, wmd=None):
    network.train(True)

    with torch.set_grad_enabled(True):
        ext_vocab_size = batch['oov_dict'].ext_vocab_size if batch['oov_dict'] else None

        network_out = network(batch, batch['targets'], criterion,
                forcing_ratio=forcing_ratio, partial_forcing=config['partial_forcing'], \
                sample=config['sample'], ext_vocab_size=ext_vocab_size, \
                include_cover_loss=config['show_cover_loss'])

        if rl_ratio > 0:
            batch_size = batch['context'].shape[0]
            sample_out = network(batch, saved_out=network_out, criterion=criterion, \
                    criterion_reduction=False, criterion_nll_only=True, \
                    sample=True, ext_vocab_size=ext_vocab_size)
            baseline_out = network(batch, saved_out=network_out, visualize=False, \
                                    ext_vocab_size=ext_vocab_size)

            sample_out_decoded = sample_out.decoded_tokens.transpose(0, 1)
            baseline_out_decoded = baseline_out.decoded_tokens.transpose(0, 1)

            neg_reward = []
            for i in range(batch_size):
                scores = eval_batch_output([batch['target_src'][i]], vocab, batch['oov_dict'],
                                       [sample_out_decoded[i]], [baseline_out_decoded[i]])

                greedy_score = scores[1][config['rl_reward_metric']]
                reward_ = scores[0][config['rl_reward_metric']] - greedy_score

                if config['rl_wmd_ratio'] > 0:
                    # Add word mover's distance
                    sample_seq = batch_decoded_index2word([sample_out_decoded[i]], vocab, batch['oov_dict'])[0]
                    greedy_seq = batch_decoded_index2word([baseline_out_decoded[i]], vocab, batch['oov_dict'])[0]

                    sample_wmd = -wmd.distance(sample_seq, batch['target_src'][i]) / max(len(sample_seq.split()), 1)
                    greedy_wmd = -wmd.distance(greedy_seq, batch['target_src'][i]) / max(len(greedy_seq.split()), 1)
                    wmd_reward_ = sample_wmd - greedy_wmd
                    wmd_reward_ = max(min(wmd_reward_, config['max_wmd_reward']), -config['max_wmd_reward'])
                    reward_ += config['rl_wmd_ratio'] * wmd_reward_

                neg_reward.append(reward_)
            neg_reward = to_cuda(torch.Tensor(neg_reward), network.device)


            # if sample > baseline, the reward is positive (i.e. good exploration), rl_loss is negative
            rl_loss = torch.sum(neg_reward * sample_out.loss) / batch_size
            rl_loss_value = torch.sum(neg_reward * sample_out.loss_value).item() / batch_size
            loss = (1 - rl_ratio) * network_out.loss + rl_ratio * rl_loss
            loss_value = (1 - rl_ratio) * network_out.loss_value + rl_ratio * rl_loss_value

            metrics = eval_batch_output(batch['target_src'], vocab, \
                            batch['oov_dict'], baseline_out.decoded_tokens)[0]

        else:
            loss = network_out.loss
            loss_value = network_out.loss_value
            metrics = eval_batch_output(batch['target_src'], vocab, \
                            batch['oov_dict'], network_out.decoded_tokens)[0]

    return loss, loss_value, metrics

def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item()
    return correct / len(labels)
