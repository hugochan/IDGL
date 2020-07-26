# The code below are borrowed from https://github.com/lucfra/LDS-GNN/blob/master/
import os
import pickle
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import kneighbors_graph

import torch
from ..generic_utils import *


class Config:
    """ Base class of a configuration instance; offers keyword initialization with easy defaults,
    pretty printing and grid search!
    """
    def __init__(self, **kwargs):
        self._version = 1
        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise AttributeError('This config does not include attribute: {}'.format(k) +
                                     '\n Available attributes with relative defaults are\n{}'.format(
                                         str(self.default_instance())))

    def __str__(self):
        _sting_kw = lambda k, v: '{}={}'.format(k, v)

        def _str_dict_pr(obj):
            return [_sting_kw(k, v) for k, v in obj.items()] if isinstance(obj, dict) else str(obj)

        return self.__class__.__name__ + '[' + '\n\t'.join(
            _sting_kw(k, _str_dict_pr(v)) for k, v in sorted(self.__dict__.items())) + ']\n'

    @classmethod
    def default_instance(cls):
        return cls()

    @classmethod
    def grid(cls, **kwargs):
        """Builds a mesh grid with given keyword arguments for this Config class.
        If the value is not a list, then it is considered fixed"""

        class MncDc:
            """This is because np.meshgrid does not always work properly..."""

            def __init__(self, a):
                self.a = a  # tuple!

            def __call__(self):
                return self.a

        sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
        for k, v in sin.items():
            copy_v = []
            for e in v:
                copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
            sin[k] = copy_v

        grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
        return [cls(**far.utils.merge_dicts(
            {k: v for k, v in kwargs.items() if not isinstance(v, list)},
            {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
        )) for vv in grd]



class ConfigData(Config):
    def __init__(self, **kwargs):
        self.seed = 0
        self.f1 = 'load_data_del_edges'
        self.dataset_name = 'cora'
        self.kwargs_f1 = {}
        self.f2 = 'reorganize_data_for_es'
        self.kwargs_f2 = {}
        super().__init__(**kwargs)

    def load(self):
        res = eval(self.f1)(seed=self.seed, dataset_name=self.dataset_name, **self.kwargs_f1)
        if self.f2:
            res = eval(self.f2)(res, **self.kwargs_f2, seed=self.seed)
        return res



class UCI(ConfigData):

    def __init__(self, **kwargs):
        self.n_train = None
        self.n_val = None
        super().__init__(**kwargs)

    def load(self, data_dir=None, knn_size=None, epsilon=None, knn_metric='cosine'):
        assert (knn_size is None) or (epsilon is None)
        if self.dataset_name == 'iris':
            data = datasets.load_iris()
            scale_ = False
        elif self.dataset_name == 'wine':
            data = datasets.load_wine()
            scale_ = True
        elif self.dataset_name == 'breast_cancer':
            data = datasets.load_breast_cancer()
            scale_ = True
        elif self.dataset_name == 'digits':
            data = datasets.load_digits()
            scale_ = True
        elif self.dataset_name == 'fma':
            data = np.load(os.path.join(data_dir, 'fma.npz'))
            scale_ = False
        elif self.dataset_name == '20news10':
            scale_ = False
            # from sklearn.datasets import fetch_20newsgroups
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.feature_extraction.text import TfidfTransformer
            # categories = ['alt.atheism',
            #               'comp.sys.ibm.pc.hardware',
            #               'misc.forsale',
            #               'rec.autos',
            #               'rec.sport.hockey',
            #               'sci.crypt',
            #               'sci.electronics',
            #               'sci.med',
            #               'sci.space',
            #               'talk.politics.guns']
            # data = fetch_20newsgroups(subset='all', categories=categories)
            # pickle.dump(data, open(os.path.join(data_dir, '20news10.pkl'), 'wb'))
            data = pickle.load(open(os.path.join(data_dir, '20news10.pkl'), 'rb'))
            vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
            X_counts = vectorizer.fit_transform(data.data).toarray()
            transformer = TfidfTransformer(smooth_idf=False)
            features = transformer.fit_transform(X_counts).todense()
        else:
            raise AttributeError('dataset not available')

        if self.dataset_name != 'fma':
            from sklearn.preprocessing import scale
            if self.dataset_name != '20news10':
                if scale_:
                    features = scale(data.data)
                else:
                    features = data.data
            y = data.target
        else:
            features = data['X']
            y = data['y']
        ys = LabelBinarizer().fit_transform(y)
        if ys.shape[1] == 1:
            ys = np.hstack([ys, 1 - ys])
        n = features.shape[0]
        from sklearn.model_selection import train_test_split
        train, test, y_train, y_test = train_test_split(np.arange(n), y, random_state=self.seed,
                                                        train_size=self.n_train + self.n_val,
                                                        test_size=n - self.n_train - self.n_val,
                                                        stratify=y)
        train, val, y_train, y_val = train_test_split(train, y_train, random_state=self.seed,
                                                    train_size=self.n_train, test_size=self.n_val,
                                                    stratify=y_train)

        features = torch.Tensor(features)
        labels = torch.LongTensor(np.argmax(ys, axis=1))
        idx_train = torch.LongTensor(train)
        idx_val = torch.LongTensor(val)
        idx_test = torch.LongTensor(test)


        if not knn_size is None:
            print('[ Using KNN-graph as input graph: {} ]'.format(knn_size))
            adj = kneighbors_graph(features, knn_size, metric=knn_metric, include_self=True)
            adj_norm = normalize_sparse_adj(adj)
            adj_norm = torch.Tensor(adj_norm.todense())
        elif not epsilon is None:
            print('[ Using Epsilon-graph as input graph: {} ]'.format(epsilon))
            feature_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
            attention = torch.mm(feature_norm, feature_norm.transpose(-1, -2))
            mask = (attention > epsilon).float()
            adj = attention * mask
            adj = (adj > 0).float()
            adj_norm = normalize_adj(adj)
        else:
            adj_norm = None

        return adj_norm, features, labels, idx_train, idx_val, idx_test

