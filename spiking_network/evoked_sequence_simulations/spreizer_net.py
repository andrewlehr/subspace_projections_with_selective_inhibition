# Copyright 2021 Leo Hiselius
# The MIT License
# adapted by Andrew Lehr, 2024

from brian2 import *
import numpy as np
from scipy import interpolate
import warnings
from perlin import generate_perlin
from torus_distance import torus_distance
import params

class SpreizerNet:  

    def __init__(self):
        self.network = Network()
        self.neuron_groups = {'e' : None, 'i' : None}
        self.synapses = {'ee' : None, 'ie' : None, 'ei' : None, 'ii' : None}
        self.spike_monitors = {'e' : None, 'i' : None}
        self.state_monitors = {'e' : None, 'i' : None}
        self.spike_generator = None
        self.spike_generator_synapses = None
      
    def set_seed(self, seed_value=0):
        """Sets the seed for any stochastic elements of the simulation.

        Args:
            seed_value (int, optional): The seed value. Defaults to 0.
        """        
        seed(seed_value)
        np.random.seed(seed_value)
  
    def populate(self):     
        """Fills the network with e and i neurons on evenly spaced [0,1]x[0,1]-grid.
        """        
        if params.neuron_params['sigma_gwn'] == 0:
            integration_method = 'exact'
        else:
            integration_method = 'euler'
 
        # Instantiate the excitatory and inhibitory networks
        self.neuron_groups['e'] = NeuronGroup(params.network_dimensions['n_pop_e'], params.neuron_eqs, threshold='v>Vt',
                                   reset='v=Vr', refractory=params.neuron_params['tau_ref'], method=integration_method,
                                   name='e_neurons')
        
        self.neuron_groups['i'] = NeuronGroup(params.network_dimensions['n_pop_i'], params.neuron_eqs, threshold='v>Vt',
                                   reset='v=Vr', refractory=params.neuron_params['tau_ref'], method=integration_method,
                                   name='i_neurons')

        # Neuron parameters
        for param in params.neuron_params:
            self.neuron_groups['e'].namespace[param] = params.neuron_params[param]
            self.neuron_groups['i'].namespace[param] = params.neuron_params[param]
        self.neuron_groups['e'].namespace['ta_in'] = params.ta_in
        self.neuron_groups['i'].namespace['ta_in'] = params.ta_in
        self.neuron_groups['e'].namespace['ta_pulse'] = params.ta_pulse
        self.neuron_groups['i'].namespace['ta_pulse'] = params.ta_pulse

        # Place neurons on evenly spaced grid [0,1]x[0,1]. i neurons are shifted to lay in between e neurons.
        n_row_e = float(params.network_dimensions['n_row_e'])
        n_row_i = float(params.network_dimensions['n_row_i'])
        n_col_e = float(params.network_dimensions['n_col_e'])
        n_col_i = float(params.network_dimensions['n_col_i'])
        self.neuron_groups['e'].x = '(i // n_col_e) / n_col_e'
        self.neuron_groups['e'].y = '(i % n_row_e) / n_row_e'
        self.neuron_groups['i'].x = '(i // n_col_i) / n_col_i + 1/(2*n_col_e)'
        self.neuron_groups['i'].y = '(i % n_row_i) / n_row_i + 1/(2*n_row_e)'

        # Add to network attribute
        self.network.add(self.neuron_groups['e'])
        self.network.add(self.neuron_groups['i'])

    def connect_random(self):
        """Connects a randomly connected network.
        """        
        # Define synapses
        self.synapses['ee'] = Synapses(self.neuron_groups['e'], on_pre='ke += Je')
        self.synapses['ie'] = Synapses(self.neuron_groups['e'], self.neuron_groups['i'], on_pre='ke += Je')
        self.synapses['ei'] = Synapses(self.neuron_groups['i'], self.neuron_groups['e'], on_pre='ki += Ji')
        self.synapses['ii'] = Synapses(self.neuron_groups['i'], on_pre='ki += Ji')

        # Synapse parameters
        for param in params.synapse_params:
            self.synapses['ee'].namespace[param] = params.synapse_params[param]
            self.synapses['ie'].namespace[param] = params.synapse_params[param]
            self.synapses['ei'].namespace[param] = params.synapse_params[param]
            self.synapses['ii'].namespace[param] = params.synapse_params[param]
    
        # Make synapses
        synapse_names = ['ee', 'ie', 'ei', 'ii']
        for syn_name in synapse_names:
            if syn_name == 'ee' or syn_name == 'ie':
                p_con = params.synapse_params['p_e']
            elif syn_name == 'ei' or syn_name == 'ii':
                p_con = params.synapse_params['p_i']
            self.synapses[syn_name].connect(p=p_con)

        # Synapse delay
        self.synapses['ee'].delay = self.synapses['ie'].delay = \
             self.synapses['ei'].delay = self.synapses['ii'].delay = 'synapse_delay'

        # Add to network attribute
        self.network.add(self.synapses['ee'])
        self.network.add(self.synapses['ie'])
        self.network.add(self.synapses['ei'])
        self.network.add(self.synapses['ii'])
    
    def connect(self, allow_multiple_connections=True, perlin_seed_value=0):
        """Connects neurons with synapses.

        Args:
            allow_multiple_connections (bool, optional): If True, multiple connections can be made where probability
                                                            of connecting is greater than one. Defaults to True.
            perlin_seed (int, optional): Seed passed to generate_perlin(). Defaults to 0.
        """      
        
        # Generate perlin map
        perlin_map = generate_perlin(int(np.sqrt(params.network_dimensions['n_pop_e'])), params.perlin_scale,
            seed_value=perlin_seed_value)

        idx = 0
        for i in range(params.network_dimensions['n_col_e']):
            for j in range(params.network_dimensions['n_row_e']):
                self.neuron_groups['e'].x_shift[idx] = params.grid_offset / params.network_dimensions['n_col_e'] * np.cos(perlin_map[i, j])
                self.neuron_groups['e'].y_shift[idx] = params.grid_offset / params.network_dimensions['n_row_e'] * np.sin(perlin_map[i, j])
                idx += 1
      
        # Define synapses
        self.synapses['ee'] = Synapses(self.neuron_groups['e'], on_pre='ke += Je')
        self.synapses['ie'] = Synapses(self.neuron_groups['e'], self.neuron_groups['i'], on_pre='ke += Je')
        self.synapses['ei'] = Synapses(self.neuron_groups['i'], self.neuron_groups['e'], on_pre='ki += Ji')
        self.synapses['ii'] = Synapses(self.neuron_groups['i'], on_pre='ki += Ji')

        # Synapse parameters
        for param in params.synapse_params:
            self.synapses['ee'].namespace[param] = params.synapse_params[param]
            self.synapses['ie'].namespace[param] = params.synapse_params[param]
            self.synapses['ei'].namespace[param] = params.synapse_params[param]
            self.synapses['ii'].namespace[param] = params.synapse_params[param]
        
        # Make synapses
        synapse_names = ['ee', 'ie', 'ei', 'ii']
        if allow_multiple_connections:
            for syn_name in synapse_names:
                p_con = params.p_con[syn_name]
                n_con = '(int(' + p_con + ')+1)'                    # ceil the likelihood. This is number of connections
                p_con += '/' + n_con                                # divide the likelihood with its ceil
                self.synapses[syn_name].connect(p=p_con, n=n_con)   # n_con connections are made with p=p_con
        else:
            for syn_name in synapse_names:
                self.synapses[syn_name].connect(p=params.p_con[syn_name])
            
        # Synapse delay
        self.synapses['ee'].delay = self.synapses['ie'].delay = \
             self.synapses['ei'].delay = self.synapses['ii'].delay = 'synapse_delay'

        # Add to network attribute
        self.network.add(self.synapses['ee'])
        self.network.add(self.synapses['ie'])
        self.network.add(self.synapses['ei'])
        self.network.add(self.synapses['ii'])

    def print_average_projections(self):
        """Prints the average number of outgoing projections for the e and i population, respectively.
        """        
        e_out = np.add(self.synapses['ee'].N_outgoing_pre, self.synapses['ie'].N_outgoing_pre)
        i_out = np.add(self.synapses['ei'].N_outgoing_pre, self.synapses['ii'].N_outgoing_pre)
        print('Number of outgoing projections for e network is ' +
              str(np.mean(e_out)) + ' +/- ' + str(np.std(e_out)))
        print('Number of outgoing projections for i network is ' +
              str(np.mean(i_out)) + ' +/- ' + str(np.std(i_out)))

    def set_initial_potentials(self, start_at_rest=True, exc_potentials=[], inh_potentials=[]):
        """Sets the initial membrane potentials for all neurons

        Args:
            start_at_rest (bool, optional):   Controls if neurons start at rest or uniformly random between Vr and Vt. 
                                              Defaults to True.
            exc_potentials (array, optional): Use this to explicitly set the initial voltages. Must set start_at_rest=False
                                              and provide inh_potentials as well. Defaults to empty list.
        """        
        if start_at_rest:      
            self.neuron_groups['e'].v = 'Vr'
            self.neuron_groups['i'].v = 'Vr'
        elif len(exc_potentials):
            self.neuron_groups['e'].v = 'Vr + rand() * (Vt - Vr)'
            self.neuron_groups['i'].v = 'Vr + rand() * (Vt - Vr)'
        elif len(exc_potentials) and len(inh_potentials):
            self.neuron_groups['e'].v = exc_potentials * mV
            self.neuron_groups['i'].v = inh_potentials * mV
        else:
            print('Error in setting initial potentials.')

    def prepare_external_input(self, samples, times, sigma_spike=0):
        """Reshapes flattened input data to n_row_e x n_col_e and returns the list of indices and
            their respective spike times

        Args:
            samples ([np.array]): list of flattened input data.
            times ([ms]): the corresponding spike times (in ms) of the samples.
            sigma_spike (int, optional): standard deviation (in ms) of spike timing. Defaults to 0.

        Returns:
            spike_indices [int]: list of neuron indices
            spike_times [ms] : list of corresponding spike times for all neurin indices in spike_indices
        """        
        
        # Takes 1d input vector of length M^2 and outputs length N^2
        # Reshapes vector to MxM image, interpolates, resamples, reshape back to vector
        # Code below written by Andrew Lehr.
        M = int(np.sqrt(np.shape(samples[0])[0]))
        N = params.network_dimensions['n_row_e']
        spike_indices = []
        spike_times = []
        
        for t, v in enumerate(samples):
            time = times[t]
            v = np.rot90(v.reshape(M,M), k=3)
            x = np.arange(M)
            y = np.arange(M)
            xx, yy = np.meshgrid(x, y)
            f = interpolate.interp2d(x, y, v, kind='linear') 
            xnew = np.linspace(1, M, N)
            ynew = np.linspace(1, M, N)
            znew = f(xnew, ynew)
            znew = np.where(znew.reshape(N*N,) > 0)[0]
        # End of Andrew Lehr's code.
            for idx in znew:
                spike_indices.append(idx)
                spike_times.append(max(time/ms+sigma_spike*np.round(np.random.randn(),1),0)*ms)

        return spike_indices, spike_times
        
    def connect_external_input(self, spike_idcs, pulse_time=10*ms, mean_stim=1.1, sigma_stim=0.2):
        """Generate external input (e.g. MNIST) at a specific time
        Args:
            spike_idcs ([int]): the corresponding indices
            pulse_time (int*ms, optional): the time of stimulus
            mean_stim (float, optional): the mean stimulation (1 drives the neuron to Vt). Defaults to 1.1
            sigma_stim (float, optional): the std of stimulation. Defaults to 0.2. With these settings, 70% fire.
        """      

        def alpha_fun(times, spike_time, num_currents, mean_stim, sigma_stim):
            #pot_gap = -((params.neuron_params['Vr'] - params.neuron_params['Vt'] + \
            #    params.neuron_params['mu_gwn'] / params.neuron_params['gL']) / mV)

            pot_gap = 15    # if mu_gwn != 0, the potential gap will be something different.
            Jsyn = pot_gap * 10 * np.e / 0.22
            tau = params.neuron_params['tau_e']
            time_mat = np.array([times, ]*num_currents).T * second
            all_currents = np.maximum(0, (time_mat - spike_time) / tau * np.exp(-(time_mat - spike_time) / tau)*\
                (mean_stim+sigma_stim*np.random.randn(num_currents))*Jsyn)
            return all_currents

        max_time = int(1 * second / defaultclock.dt)    # If simulation is longer than 1 second, max_time should be increased (=sim_time/dt)
        times = np.arange(0, 1000, defaultclock.dt/ms) * ms
        ta_values = np.zeros((max_time, params.network_dimensions['n_pop_e']))
        ta_values[:, spike_idcs] = alpha_fun(times, pulse_time, len(spike_idcs), mean_stim, sigma_stim)
        ta = TimedArray(ta_values * pA, dt=0.1*ms)
        self.neuron_groups['e'].namespace['ta_pulse'] = ta

        
    def connect_excitatory_input(self, times):
        ta = TimedArray(times / ms, dt=0.1*ms)
        self.neuron_groups['e'].namespace['ta_in'] = ta
        
        
    #def connect_inhibitory_subset(self, neurons):
     
    def connect_spike_generator(self, N, r, T):
        
        # define spikes
        n_spikes = int(r * T / 1000)
        n_time_steps = int(T / (defaultclock.dt / ms))
        ts = np.zeros((N,n_spikes))
        for neuron in range(N):
            ts[neuron, :] = np.sort(np.random.choice(n_time_steps, size=n_spikes, replace=False))
        ts = ts.reshape(N*n_spikes)
        
        # Create spike generator group
        self.spike_generator['sg_e'] = SpikeGeneratorGroup(N, ids, ts*ms)
        
        # Create synapses
        self.spike_generator_synapses['sg_e'] = Synapses(self.spike_generator['sg_e'], 
                                                         self.neuron_groups['e'], 
                                                         on_pre='ke += J_e_input')
        
        # Synapse parameters
        for param in params.spike_generator_params:
            self.spike_generator_synapses['sg_e'].namespace[param] = params.spike_generator_params[param]
            
        # connect
        self.spike_generator_synapses['sg_e'].connect(p=params.spike_generator_params['p_sg'])
        
        self.network.add(self.spike_generator['sg_e'])
        self.network.add(self.spike_generator_synapses['sg_e'])
        
    def connect_spike_monitors(self):
        """Connect spike monitors to neuron_groups.
        """        
        self.spike_monitors['e'] = SpikeMonitor(self.neuron_groups['e'], record=True)
        self.spike_monitors['i'] = SpikeMonitor(self.neuron_groups['i'], record=True)
        self.network.add(self.spike_monitors['e'])
        self.network.add(self.spike_monitors['i'])

    def connect_state_monitors(self, e_idcs, i_idcs):
        """Connect state monitors to neuron_groups.

        Args:
            e_idcs ([int]): indices of neurons to be recorded
            i_idcs ([int]): indices of neurons to be recorded
        """              
        self.state_monitors['e'] = StateMonitor(self.neuron_groups['e'], 'v', record=e_idcs)
        self.state_monitors['i'] = StateMonitor(self.neuron_groups['i'], 'v', record=i_idcs)
        self.network.add(self.state_monitors['e'])
        self.network.add(self.state_monitors['i'])

    def store_network(self):
        """Stores network. restore() restores to this point.
        """        
        self.network.store()

    def run_sim(self, simulation_time=1*second, is_report=True):
        """Runs the simulation.

        Args:
            simulation_time (second, optional): duration of the simulation. Defaults to 1*second.
            is_report (bool, optional): determines if simulation progress is reported in terminal or not. Defaults to True
        """   
        # Print warning if the spike_monitors are not connected
        if self.spike_monitors['e'] is None or self.spike_monitors['i'] is None:
            warnings.warn('Simulation running without SpikeMonitors!')

        # Run simulation
        if is_report:
            self.network.run(simulation_time, report='stderr', report_period=2*second)
        else:
            self.network.run(simulation_time, report=None)

    def save_monitors(self, simulation_name, SUPERDIR=''):
        """Saves any existing montitors (spike and state) to folders spike_monitors and state_monitors, respectively.

        Args:
            simulation_name (str): The name of the simulation.
            SUPERDIR (str, optional): The directory in which the "saves"-directory exists. Defaults to ''.
        """        
        # Save spike_monitors
        if self.spike_monitors['e'] is not None: 
            np.save(SUPERDIR+'saves/spike_monitors/'+simulation_name+'_e_i', self.spike_monitors['e'].i[:])
            np.save(SUPERDIR+'saves/spike_monitors/'+simulation_name+'_e_t', self.spike_monitors['e'].t[:])
        if self.spike_monitors['i'] is not None:
            np.save(SUPERDIR+'saves/spike_monitors/'+simulation_name+'_i_i', self.spike_monitors['i'].i[:])
            np.save(SUPERDIR+'saves/spike_monitors/'+simulation_name+'_i_t', self.spike_monitors['i'].t[:])

        # Save state_monitors
        if self.state_monitors['e'] is not None:
            np.save(SUPERDIR+'saves/state_monitors/'+simulation_name+'_e_v', self.state_monitors['e'].v[:])
            np.save(SUPERDIR+'saves/state_monitors/'+simulation_name+'_e_t', self.state_monitors['e'].t[:])
        if self.state_monitors['i'] is not None:
            np.save(SUPERDIR+'saves/state_monitors/'+simulation_name+'_i_v', self.state_monitors['i'].v[:])
            np.save(SUPERDIR+'saves/state_monitors/'+simulation_name+'_i_t', self.state_monitors['i'].t[:])

    def restore_network(self):
        """Restores all attributes of the current class instance to the point where store_network() was called.
        """        
        if hasattr(self.network, 'sorted_objects'):
            if self.spike_generator in self.network.sorted_objects:
                self.network.remove(self.spike_generator)
                self.network.remove(self.spike_generator_synapses)
        self.network.restore()
