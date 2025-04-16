# Copyright 2023 Andrew Lehr
# The MIT License

import os
import os.path as path
import _pickle as cPickle
import numpy as np

class Analysis:
    def __init__(self, exp_data_dir=''):
        pass
        #self.root = path.abspath(path.join(__file__ ,"../../../..")) + '/'
        #self.data_dir = self.root + 'data/'
        #self.exp_dir = self.data_dir + exp_data_dir
        #self.param_space = self.load_metadata()
    # eigenvalue spectrum, real-imag
    # eigenvalue magnitudes
    # first K eigenvectors, real-imag
    # dominant frequency of eigenvectors
    
    # pca
    # explained variance
    # principal angles
    
    # plots
    # single neurons
    # pca 2d projection
    
    #def load_metadata(self):
    #    name = self.exp_dir + 'param_space.pkl'
    #    with open(name, "rb") as f:
    #        return cPickle.load(f)
    
    #def load_data(self):

    def compute_eigendecomposition(self, net):
        # extract weight matrix and P matrix
        W = net.W
        P = net.P

        # compute PW
        PW = np.diag(P)@W

        # make reduced row-column matrix
        W_reduced = W[P.astype(bool), :]
        W_reduced = W_reduced[:, P.astype(bool)]

        # compute eigendecomposition
        vals_W, vecs_W = np.linalg.eig(W)
        vals_PW, vecs_PW = np.linalg.eig(PW)
        vals_Wr, vecs_Wr = np.linalg.eig(W_reduced)

        # largest complex eigenvalue 
        val_W_max = np.max(np.abs(vals_W))
        val_PW_max = np.max(np.abs(vals_PW))
        val_Wr_max = np.max(np.abs(vals_Wr))

        # index of largest eigenvalue, for getting proper eigenvector below
        val_W_argmax = np.argmax(np.abs(vals_W))
        val_PW_argmax = np.argmax(np.abs(vals_PW))
        val_Wr_argmax = np.argmax(np.abs(vals_Wr))

        # largest real part
        val_W_max_real = np.max(np.abs(vals_W.real))
        val_PW_max_real = np.max(np.abs(vals_PW.real))
        val_Wr_max_real = np.max(np.abs(vals_Wr.real))

        # largest imaginary part
        val_W_max_imag = np.max(np.abs(vals_W.imag))
        val_PW_max_imag = np.max(np.abs(vals_PW.imag))
        val_Wr_max_imag = np.max(np.abs(vals_Wr.imag))

        # eigenvector corresponding to largest magnitude complex eigenvalue
        vec_W_max = vecs_W[:, val_W_argmax]
        vec_PW_max = vecs_PW[:, val_PW_argmax]
        vec_Wr_max = vecs_Wr[:, val_Wr_argmax]

        eigendecomposition_data = {    
            # eigenvalues
            'evals_W': vals_W,
            'evals_PW': vals_PW, 
            'evals_Wr': vals_Wr, 
            # max complex eigenvalue
            'val_W_max': val_W_max, 
            'val_PW_max': val_PW_max,
            'val_Wr_max': val_Wr_max,
            # index of largest eigenvalue
            'val_W_argmax': val_W_argmax,
            'val_PW_argmax': val_PW_argmax,
            'val_Wr_argmax': val_Wr_argmax,
            # largest real part
            'val_W_max_real': val_W_max_real,
            'val_PW_max_real': val_PW_max_real,
            'val_Wr_max_real': val_Wr_max_real,
            # largest imaginary part
            'val_W_max_imag': val_W_max_imag,
            'val_PW_max_imag': val_PW_max_imag,
            'val_Wr_max_imag': val_Wr_max_imag,
            # eigenvector corresponding to largest magnitude complex eigenvalue
            'vec_W_max': vec_W_max, 
            'vec_PW_max': vec_PW_max, 
            'vec_Wr_max': vec_PW_max
        }
        return eigendecomposition_data
        
        