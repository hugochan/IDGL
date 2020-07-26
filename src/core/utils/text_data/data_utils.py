import os
import re
import codecs
import string
from collections import defaultdict
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from sklearn import preprocessing

from .vocab_utils import VocabModel


tokenize = lambda s: wordpunct_tokenize(re.sub('[%s]' % re.escape(string.punctuation), ' ', s))

def load_data(config):
    data_split = [float(x) for x in config['data_split_ratio'].replace(' ', '').split(',')]
    if config['dataset_name'] == 'mrd':
        file_path = os.path.join(config['data_dir'], 'mrd.txt')
        train_set, dev_set, test_set = load_mrd_data(file_path, data_split, config.get('data_seed', 1234))
    elif config['dataset_name'] == '20news':
        train_set, dev_set, test_set = load_20news_data(config['data_dir'], data_split, config.get('data_seed', 1234))
    else:
        raise ValueError('Unknown dataset_name: {}'.format(config['dataset_name']))

    return train_set, dev_set, test_set

def load_mrd_data(file_path, data_split, seed):
    '''Loads the Movie Review Data (https://www.cs.cornell.edu/people/pabo/movie-review-data/).'''

    all_instances = []
    all_seq_len = []
    with open(file_path, 'r') as fp:
        for line in fp:
            idx, rating, subj = line.split('\t')
            word_list = tokenize(subj.lower())
            all_instances.append([word_list, float(rating)])
            all_seq_len.append(len(word_list))

    print('[ Max seq length: {} ]'.format(np.max(all_seq_len)))
    print('[ Min seq length: {} ]'.format(np.min(all_seq_len)))
    print('[ Mean seq length: {} ]'.format(int(np.mean(all_seq_len))))

    # Random data split
    train_ratio, dev_ratio, test_ratio = data_split
    assert train_ratio + dev_ratio + test_ratio == 1
    n_train = int(len(all_instances) * train_ratio)
    n_dev = int(len(all_instances) * dev_ratio)
    n_test = len(all_instances) - n_train - n_dev

    random = np.random.RandomState(seed)
    random.shuffle(all_instances)

    train_instances = all_instances[:n_train]
    dev_instances = all_instances[n_train: n_train + n_dev]
    test_instances = all_instances[-n_test:]
    return train_instances, dev_instances, test_instances


def load_20news_data(data_dir, data_split, seed):
    train_dev_instances, train_dev_seq_len, train_dev_labels = data_load_helper(os.path.join(data_dir, '20news-bydate-train'))
    test_instances, test_seq_len, test_labels = data_load_helper(os.path.join(data_dir, '20news-bydate-test'))

    all_seq_len = train_dev_seq_len + test_seq_len
    print('[ Max seq length: {} ]'.format(np.max(all_seq_len)))
    print('[ Min seq length: {} ]'.format(np.min(all_seq_len)))
    print('[ Mean seq length: {} ]'.format(int(np.mean(all_seq_len))))

    le = preprocessing.LabelEncoder()
    le.fit(train_dev_labels + test_labels)
    nclass = len(list(le.classes_))
    print('[# of classes: {}] '.format(nclass))

    train_dev_labels = le.transform(train_dev_labels)
    test_labels = le.transform(test_labels)
    train_dev_instances = list(zip(train_dev_instances, train_dev_labels))
    test_instances = list(zip(test_instances, test_labels))


    # Random data split
    train_ratio, dev_ratio = data_split
    assert train_ratio + dev_ratio == 1
    n_train = int(len(train_dev_instances) * train_ratio)

    random = np.random.RandomState(seed)
    random.shuffle(train_dev_instances)

    train_instances = train_dev_instances[:n_train]
    dev_instances = train_dev_instances[n_train:]
    return train_instances, dev_instances, test_instances

def data_load_helper(data_dir):
    all_instances = []
    all_seq_len = []
    all_labels = []
    files = get_all_files(data_dir, recursive=True)
    for filename in files:
        # with open(filename, 'r') as fp:
        with codecs.open(filename, 'r', encoding='UTF-8', errors='ignore') as fp:
            text = fp.read().lower()
            word_list = tokenize(text)

            parent_name, child_name = os.path.split(filename)
            doc_name = os.path.split(parent_name)[-1] + '_' + child_name
            label = doc_name.split('_')[0]

            all_instances.append(word_list)
            all_seq_len.append(len(word_list))
            all_labels.append(label)

    return all_instances, all_seq_len, all_labels


def get_all_files(corpus_path, recursive=False):
    if recursive:
        return [os.path.join(root, file) for root, dirnames, filenames in os.walk(corpus_path) for file in filenames if os.path.isfile(os.path.join(root, file)) and not file.startswith('.')]
    else:
        return [os.path.join(corpus_path, filename) for filename in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, filename)) and not filename.startswith('.')]
