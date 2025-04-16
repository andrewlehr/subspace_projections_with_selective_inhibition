# Copyright 2023 Andrew Lehr
# The MIT License

import numpy as np
import itertools
from sklearn.linear_model import Ridge

class RingNetwork:
    def __init__(self, params):
        self.params = params
        self._built = False
        self.weights = None
        self.I_E = None
        self.I_I = None
        self.P = None
        self.R = None
        
    def _connect(self):

        # determine center of gaussian kernel for neuron j
        c = self.params.shift % self.params.N

        # compute distance (counter clockwise) from neuron j to each of the other neurons i
        d = abs(self.params.x - c)

        # distance on circle is minimum of clockwise and counterclockwise distance
        dx = np.minimum(d, self.params.N-d)

        # compute the weights with gaussian kernel, parameters: w_E, w_I, shift, sigma
        weights = self.params.w_E*np.exp(-0.5 * dx**2/self.params.sigma**2) - self.params.w_I    
        
        # rescale weights by weight factor, computed within parameter class
        weights = self.params.weight_factor * weights
        
        # store weights
        self.weights = weights

    @property
    def W(self):
        W = np.zeros((self.params.N,self.params.N))
        for i in range(self.params.N):
            W[:, i] = np.roll(self.weights, i)
        return W
        
    def _set_inputs(self):   
        self.I_E = self.params.I_E
        self.I_I = self.params.I_I
        self.P = self.params.P
    
    def _after_run(self):
        pass
    
    def _build(self):
        self._connect()
        self._set_inputs()
        self._built = True
        
    def run(self):
        if self._built == False:
            self._build()
        
        W = self.W    
        r_store = np.zeros((self.params.N, self.params.T))
        r = self.params.initial_r
        
        for t in range(self.params.T):
            r_store[:, t] = r
            r = self.P * (W @ r + self.I_E - self.I_I)
            r[r<0] = 0
        
        self.R = r_store
    
        self._after_run()


class RingNetworkDynamicSelection:
    def __init__(self, params, tau=1, tau_I=1, w_ei_1=None, w_ei_2=None, w_ie=1, w_ii=1, I_ext_1=None, I_ext_2=None):
        self.params = params      # parameter object 
        self._built = False       # flag whether network is already built
        self.weights = None       # weight vector for ring network recurrent weights
        self.I_E = None           # excitatory input to ring network
        self.I_I = None           # inhibitory input to ring network
        self.P = None             # projection matrix
        self.R = None             # firing rate matrix NxT
        self.tau = tau            # time constant ring network
        self.tau_I = tau_I        # time contstant selective inhibition
        self.w_ie = w_ie          # weight from ring neurons to selective inhibition ensembles, constant
        self.w_ii = w_ii          # weight between selective inhibition ensembles, constant
        self.I_ext_1 = I_ext_1    # top down external input to selective inhibition ensemble 1, 1xT
        self.I_ext_2 = I_ext_2    # top down external input to selective inhibition ensemble 2, 1xT
        self.w_ei_1 = w_ei_1      # weight vector from selective inhibitory ensemble 1 to ring network, Nx1 
        self.w_ei_2 = w_ei_2      # weight vector from selective inhibitory ensemble 2 to ring network, Nx1
        
    def _connect(self):

        # determine center of gaussian kernel for neuron j
        c = self.params.shift % self.params.N

        # compute distance (counter clockwise) from neuron j to each of the other neurons i
        d = abs(self.params.x - c)

        # distance on circle is minimum of clockwise and counterclockwise distance
        dx = np.minimum(d, self.params.N-d)

        # compute the weights with gaussian kernel, parameters: w_E, w_I, shift, sigma
        weights = self.params.w_E*np.exp(-0.5 * dx**2/self.params.sigma**2) - self.params.w_I    
        
        # rescale weights by weight factor, computed within parameter class
        weights = self.params.weight_factor * weights
        
        # store weights
        self.weights = weights

    @property
    def W(self):
        W = np.zeros((self.params.N,self.params.N))
        for i in range(self.params.N):
            W[:, i] = np.roll(self.weights, i)
        return W
    
    # activation function
    def F(self, x):
        if hasattr(x, "__len__"):
            x[x<0] = 0
            x[x>1] = 1
            return x
        else:
            return 1 if x>1 else 0 if x<0 else x
        
    def _set_inputs(self):   
        self.I_E = self.params.I_E
        self.I_I = self.params.I_I
        self.P = self.params.P
    
    def _after_run(self):
        pass
    
    def _build(self):
        self._connect()
        self._set_inputs()
        self._built = True
        
    def run(self):
        if self._built == False:
            self._build()
        
        # generate the weight matrix from the stored weight vector
        W = self.W
        
        # initialize storage for rates of ring network neurons and the inhibitory ensembles
        r_E_store = np.zeros((self.params.N, self.params.T))  # N x T
        r_I_1_store = np.zeros(self.params.T)                 # 1 x T
        r_I_2_store = np.zeros(self.params.T)                 # 1 x T

        # initial values for firing rates
        r_E = self.params.initial_r      # ring network initialized as bump
        r_I_1 = 0                        # both inhibitory ensembles initialized to zero
        r_I_2 = 0
        
        # initial values for synaptic currents
        I = self.params.initial_r        # synaptic currents of ring network initialized to bump
        I_1 = 0                          # synaptic currents of both inhibitory ensembles initilized to zero
        I_2 = 0
            
        # loop over time steps
        for t in range(self.params.T):
            
            # store firing rates
            r_E_store[:, t] = r_E   # of ring
            r_I_1_store[t] = r_I_1  # of inhibitory ensemble 1
            r_I_2_store[t] = r_I_2  # of inhibitory ensemble 2
            
            # update synaptic inputs and rates of ring network neurons
            I = I + (1/self.tau) * (-I + self.P * (W @ r_E - self.w_ei_1 * r_I_1 - self.w_ei_2 * r_I_2))
            r_E = self.F(I)
            
            # update synaptic inputs and rate of inhibtory ensemble 1
            I_1 = I_1 + (1/self.tau_I) * (-I_1 + np.sum(self.w_ie * r_E) - self.w_ii * r_I_2 + self.I_ext_1[t])
            r_I_1 = self.F(I_1)
            
            # update synaptic inputs and rate of inhibtory ensemble 2
            I_2 = I_2 + (1/self.tau_I) * (-I_2 + np.sum(self.w_ie * r_E) - self.w_ii * r_I_1 + self.I_ext_2[t])
            r_I_2 = self.F(I_2)
            
        # after simulation store rates of ring network and inhibitory ensembles as properties of class    
        self.R = r_E_store
        self.I_1 = r_I_1_store
        self.I_2 = r_I_2_store
    
        # call an after run function, so far not implemented
        self._after_run()