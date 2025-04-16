# Copyright 2023 Andrew Lehr
# The MIT License

import os
import os.path as path
import _pickle as cPickle
from collections import defaultdict
import numpy as np

class DataManager:
    def __init__(self, exp_data_dir):
        self.root = path.abspath(path.join(__file__ ,"../../..")) + '/'
        self.data_dir = self.root + 'data/'
        self.exp_dir = self.data_dir + exp_data_dir + '/'
        self.meta_dir = self.exp_dir + 'metadata/'
        self.sim_dir = self.exp_dir + 'simulation/'  
        self.eig_dir = self.exp_dir + 'eigendecomposition/' 
        self.bump_dir = self.exp_dir + 'bump_statistics/' 
        self.load_metadata()
    
    def load_metadata(self):
        with open(self.meta_dir + 'param_space.pkl', "rb") as f:
            self.param_space = cPickle.load(f)
        with open(self.meta_dir + 'params_to_set.pkl', "rb") as f:
            self.params_to_set = cPickle.load(f)
        with open(self.meta_dir + 'params_to_iterate.pkl', "rb") as f:
            self.params_to_iterate = cPickle.load(f)
        with open(self.meta_dir + 'keys.pkl', "rb") as f:
            self.keys = cPickle.load(f)
    
    def load_data(self, parameter_setting, type='sim'):
        filename = self.param_space.index(parameter_setting)
        if type == 'sim':
            name = self.sim_dir + str(filename) + '.pkl'
            with open(name, "rb") as f:
                return cPickle.load(f)
        elif type == 'eig':
            name = self.eig_dir + str(filename) + '.pkl'
            with open(name, "rb") as f:
                return cPickle.load(f)
            
    def load(self, filename, location):
        name = str(location) + str(filename) + '.pkl'
        with open(name, "rb") as f:
            return cPickle.load(f)


    def load_eigendecomposition_data(self):
        # NOTE: this function is not nicely extendable as is, it needs refactoring to be generally applicable
        #       but it works fine to reproduce the paper
        
        # get the network size N and P matrix
        static_parameters = tuple(val[0] for val in self.params_to_set.values())
        parameter_setting = (True, 0, 0, 0) + static_parameters
        net = self.load_data(parameter_setting)
        N = net.params.N
        P = net.params.P

        rescale = self.params_to_iterate['rescale']
        perc_sel = self.params_to_iterate['p_inh']
        shifts = self.params_to_iterate['shift_percent']
        seeds = self.params_to_iterate['seed']
        
        n_rescale = len(rescale)
        n_perc_sel = len(perc_sel)
        n_shifts = len(shifts)
        n_seeds = len(seeds)
        
        # initialize dicts/arrays
        evals_W = defaultdict(dict) 
        evals_PW = defaultdict(dict) 
        evals_Wr = defaultdict(dict) 
        
        val_W_max = np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds))
        val_PW_max = np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds))
        val_Wr_max = np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds))
        
        val_W_argmax = np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds), dtype=int)
        val_PW_argmax = np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds), dtype=int)
        val_Wr_argmax = np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds), dtype=int)
        
        val_W_max_real = np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds))
        val_PW_max_real = np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds))
        val_Wr_max_real = np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds))
        
        val_W_max_imag = np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds))
        val_PW_max_imag = np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds))
        val_Wr_max_imag = np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds))
        
        vec_W_max = 1j * np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds, N))
        vec_PW_max = 1j * np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds, N))
        vec_Wr_max = 1j * np.zeros((n_rescale, n_perc_sel, n_shifts, n_seeds, N))

        # load data and store into data structures
        for l, t in enumerate(rescale):
            for k, p in enumerate(perc_sel):
                for i, s in enumerate(shifts):
                    for j, seed in enumerate(seeds):
                        # parameter setting to load
                        parameter_setting = (t, p, seed, s) + static_parameters
                        eig_data = self.load_data(parameter_setting, type='eig')
        
                        # store eigenvalues
                        evals_W[l,k,i,j] = eig_data['evals_W']
                        evals_PW[l,k,i,j] = eig_data['evals_PW']
                        evals_Wr[l,k,i,j] = eig_data['evals_Wr']
        
                        # largest complex eigenvalue 
                        val_W_max[l,k,i,j] = eig_data['val_W_max']
                        val_PW_max[l,k,i,j] = eig_data['val_PW_max']
                        val_Wr_max[l,k,i,j] = eig_data['val_Wr_max']
        
                        # index of largest eigenvalue, for getting proper eigenvector below
                        val_W_argmax[l,k,i,j] = eig_data['val_W_argmax']
                        val_PW_argmax[l,k,i,j] = eig_data['val_PW_argmax']
                        val_Wr_argmax[l,k,i,j] = eig_data['val_Wr_argmax']
        
                        # largest real part
                        val_W_max_real[l,k,i,j] = eig_data['val_W_max_real']
                        val_PW_max_real[l,k,i,j] = eig_data['val_PW_max_real']
                        val_Wr_max_real[l,k,i,j] = eig_data['val_Wr_max_real']
        
                        # largest imaginary part
                        val_W_max_imag[l,k,i,j] = eig_data['val_W_max_imag']
                        val_PW_max_imag[l,k,i,j] = eig_data['val_PW_max_imag']
                        val_Wr_max_imag[l,k,i,j] = eig_data['val_Wr_max_imag']
        
                        # eigenvector corresponding to largest magnitude complex eigenvalue
                        vec_W_max[l,k,i,j,:] = eig_data['vec_W_max']
                        vec_PW_max[l,k,i,j,:] = eig_data['vec_PW_max']
                        vec_Wr_max[l,k,i,j,P.astype(bool)] = eig_data['vec_Wr_max']

        # package as dict
        eigendecomposition_data = {    
            # eigenvalues
            'evals_W': evals_W,
            'evals_PW': evals_PW, 
            'evals_Wr': evals_Wr, 
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
                
        
    def save(self, data, filename, location):
        name = str(location) + str(filename) + '.pkl'
        with open(name, "wb") as f:
            return cPickle.dump(data, f)
        
        
        
        