# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 16:47:13 2017
@author: Chin-Wei
"""

PARAM_EXTENSION = 'params'

import cPickle as pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
floatX = 'float32'

import scipy.io
from torch.utils.data import Dataset as Dataset

import external_maf as maf

def load_bmnist_image():

    # Larochelle 2011
    path_tr = 'dataset/binarized_mnist_train.amat'
    path_va = 'dataset/binarized_mnist_valid.amat'
    path_te = 'dataset/binarized_mnist_test.amat'
    train_x = np.loadtxt(path_tr).astype('float32').reshape(50000,784)
    valid_x = np.loadtxt(path_va).astype('float32').reshape(10000,784)
    test_x = np.loadtxt(path_te).astype('float32').reshape(10000,784)
    
    
    return train_x, valid_x, test_x


def load_mnist_image(n_validation=1345, state=123):
    filenames = ['train-images-idx3-ubyte', 't10k-images-idx3-ubyte']
    path_tr = 'dataset/{}'.format(filenames[0])
    path_te = 'dataset/{}'.format(filenames[1])
    
    tr = np.loadtxt(path_tr).astype('float32')
    te = np.loadtxt(path_te).astype('float32')
    
    return tr[:50000], tr[50000:], te
    

def load_cifar10_image(labels=False):
    f = lambda d:d.astype(floatX)
    filename = 'dataset/cifar10.pkl'
    tr_x, tr_y, te_x, te_y = pickle.load(open(filename,'r'))
    if tr_x.max() == 255:
        tr_x = tr_x / 256.
        te_x = te_x / 256.
        
    if labels:
        enc = OneHotEncoder(10)
        tr_y = enc.fit_transform(tr_y).toarray().reshape(50000,10).astype(int)
        te_y = enc.fit_transform(te_y).toarray().reshape(10000,10).astype(int)
        
        return (f(d) for d in [tr_x, tr_y, te_x, te_y])   
    else:
        return (f(d) for d in [tr_x, te_x])
    
    
def load_omniglot_image(n_validation=1345, state=123):
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
    path = 'dataset/omniglot.amat'
    omni_raw = scipy.io.loadmat(path)

    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))

    n = train_data.shape[0]
    
    ind_va = np.random.RandomState(
        state).choice(n, n_validation, replace=False)
    
    ind_tr = np.delete(np.arange(n), ind_va)
    
    return train_data[ind_tr], train_data[ind_va], test_data



class DatasetWrapper(Dataset):

    def __init__(self, dataset, transform=None):

        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        
        sample = self.dataset[ind]
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class InputOnly(Dataset):

    def __init__(self, dataset):

        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        return self.dataset[ind][0]





def load_maf_data(name):

    
    if name == 'mnist':
        return maf.MNIST(logit=True, dequantize=True)

    elif name == 'bsds300':
        return maf.BSDS300()

    elif name == 'cifar10':
        return maf.CIFAR10(logit=True, flip=True, dequantize=True)

    elif name == 'power':
        return maf.POWER()

    elif name == 'gas':
        return maf.GAS()

    elif name == 'hepmass':
        return maf.HEPMASS()

    elif name == 'miniboone':
        return maf.MINIBOONE()

    else:
        raise ValueError('Unknown dataset')












