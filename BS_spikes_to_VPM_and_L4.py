#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AA population of VPM neurons receives brainstem spikes and it excites the L4.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as constants
from brian2 import *
import random

defaultclock.dt = 0.01*ms

# make input
NeuronID_Dir90 = pd.read_csv("/Users/teatompos/Documents/GitHub/PhD_Project3--full-somatosensory-pathway/Brainstem_spikes/direction_90_id.csv", header=None)
NeuronID_Dir90.columns = ["NeuronID"]
Spikes_Dir90 = pd.read_csv("/Users/teatompos/Documents/GitHub/PhD_Project3--full-somatosensory-pathway/Brainstem_spikes/direction_90_time.csv", header=None)
Spikes_Dir90.columns = ["Spike_times"]

BrainstemNeurons = 70
SimulationDuration = 400   # ms

# make VPM
ENa_vpm = 50*mV 
EK_vpm = -100*mV 

Eh_vpm = -43*mV 
Eleak_vpm = -70*mV 
EKleak_vpm = -100*mV

gNa_vpm = 90*msiemens/cm**2
gK_vpm = 20*msiemens/cm**2
gCa_vpm = 2*msiemens/cm**2
gh_vpm = 0.005*msiemens/cm**2 # lower gh allows more spikes in a burst, but it makes the system very fragile
gleak_vpm = 0.01*msiemens/cm**2
gKleak_vpm = 0.0172*msiemens/cm**2

FaradConst = constants.physical_constants['Faraday constant'][0] * coulomb/mole
GasConst = constants.gas_constant * joule / (kelvin * mole)

# Regular-spiking param-values
RS_gleak     = 1.05e-5      * siemens/cm**2
RS_Eleak     = -70.3        * mV
RS_V_T       = -59          * mV
RS_gNa       = 0.086        * siemens/cm**2
RS_gK        = 0.006        * siemens/cm**2
RS_gM        = 15e-5        * siemens/cm**2 
RS_tau_max_M = 808          * ms

# Fast-spiking param-values
FS_diam      = 56.9        * um
FS_surface   = pi*FS_diam**2

FS_gleak     = 3.8e-5       * siemens/cm**2
FS_Eleak     = -70.4        * mV
FS_gNa       = 0.078        * siemens/cm**2
FS_V_T       = -58          * mV
FS_gK        = 0.0039       * siemens/cm**2
FS_gM        = 5e-5         * siemens/cm**2
FS_tau_max_M = 602          * ms
    
# Shared values
EK_ctx = -90    * mV
ENa_ctx = 50    * mV
EM_ctx = -100   * mV

VPM = '''
         dv/dt = (Iapp - INa - IK - IT - Ih - ILeak - IKleak)/Cm : volt
         
         INa = gNa_vpm * m**3 * h * (v - ENa_vpm) : amp/meter**2
         IK = gK_vpm * n**4 * (v - EK_vpm) : amp/meter**2
         ILeak = gleak_vpm * (v - Eleak_vpm) : amp/meter**2
         IKleak = gKleak_vpm * (v - EKleak_vpm) : amp/meter**2
         IT = gCa_vpm * mT_inf**2 * hT * (v - ECa) : amp/meter**2
         Ih = gh_vpm * (o1 + 2*(1 - c1 - o1)) * (v - Eh_vpm) : amp/meter**2
         
         dm/dt = alpha_m * (1 - m) - beta_m * m : 1
         dh/dt = alpha_h * (1 - h) - beta_h * h : 1
         dn/dt = alpha_n * (1 - n) - beta_n * n : 1
         dhT/dt = (hT_inf - hT) / tau_hT : 1
         
         ECa = ((GasConst * 309.15*kelvin) / (2 * FaradConst)) * log(2 / CaBuffer) : volt
         CaBufferTemp = ((-10 * IT/(amp/meter**2)) / (2 * FaradConst / (coulomb/mole))) : 1
         dCaBuffer/dt = (int(CaBufferTemp > 0) * CaBufferTemp + (0.00024-CaBuffer)/5 ) / ms : 1 
         
         do1/dt = (0.001 * (1 - c1 - o1) - 0.001 * ( (1-p0) / 0.01)) * o1 / ms : 1
         dp0/dt = (0.0004 *  (1 - p0) - 0.0004 * ((CaBuffer / 0.002)**4) * p0) / ms : 1
         dc1/dt = beta * o1 - alpha * c1 : 1
         
         alpha_m = ((0.32 * (13 - (v/mV + 35))) / (exp((13 - (v/mV + 35))/4) - 1)) / ms : Hz
         beta_m = ((0.28 * ((v/mV + 35) - 40)) / (exp(((v/mV + 35) - 40)/5) - 1)) / ms : Hz
         alpha_h = (0.128 * exp((17 - v/mV - 35)/18)) / ms : Hz
         beta_h = (4 / (1 + exp((40 - v/mV - 35)/5))) / ms : Hz
         
         alpha_n = ((0.032 * (15 - (v/mV + 25))) / (exp((15 - (v/mV + 25))/5) - 1)) / ms : Hz
         beta_n = (0.5 * exp((10 - (v/mV + 25)) / 40)) / ms : Hz
         
         alpha = h_inf / tau_s : Hz
         beta = (1 - h_inf) / tau_s : Hz
         
         mT_inf = 1 / (1 + exp((-((v/mV + 2) + 57)) / 6.2)) : 1
         hT_inf = 1 / (1 + exp(((v/mV + 2) + 81)/4)) : 1
         tau_hT = (( 30.8 + (211.4 + exp(((v/mV+2) + 113.2)/5))/(1+exp(((v/mV + 2)+84)/3.2))) / 3.73) * ms : second
         
         h_inf =  1 / (1 + exp( (v/mV + 75)/5.5 )) : 1
         tau_s = (20 + 1000 / (exp( (v/mV + 71.5)/14.2 ) + exp( (-(v/mV + 89))/11.6))) * ms: second
           
         Cm = 1*uF/cm**2 : farad/meter**2
         Iapp : amp/meter**2
         
         taud : second
         taur : second
         slope : volt
         
         ds1/dt = -s1/taud + 1/2 * (1 + (tanh(v/slope)))* (1-s1)/taur : 1 
        '''
CTX = '''
        dv/dt = 1/Cm * (IK + INa + IM + Ileak + Igaba + Iampa + Inoise)  : volt
        
        gaba2 : 1
        ampa2 : 1
        taud_g : second
        taur_g : second
        slope_g : volt
        gsyn_g : siemens/meter**2
        Esyn_g : volt
        taud_a : second
        taur_a : second
        slope_a : volt
        gsyn_a : siemens/meter**2
        Esyn_a : volt
        
        Cm = 1*uF/cm**2 : farad/meter**2
        gleak : siemens/meter**2
        gNa : siemens/meter**2
        gK : siemens/meter**2
        gM  : siemens/meter**2
        Eleak : volt
        V_T : volt
        tau_max_M : second
        
        dn/dt = alpha_n * (1-n) - beta_n * n: 1
        dm/dt = alpha_m * (1-m) - beta_m * m : 1
        dh/dt = alpha_h * (1-h) - beta_h * h: 1
        dp/dt = (p_inf - p)/tau_p : 1
        
        alpha_n = -0.0320/mV*(v-V_T-15.0*mV) / (exp( -(v-V_T - 15.0*mV)/(5.0*mV)) - 1) * 1/msecond : Hz
        beta_n = 0.50 * exp( -(v-V_T - 10.0*mV)/(40.0*mV))*1/msecond : Hz
        alpha_m =  -0.32/mV*(v-V_T-13.*mV) / (exp(-(v-V_T-13.*mV)/(4.*mV))-1) * 1/msecond : Hz
        beta_m = ( 0.28/mV * ((v-V_T-40.*mV))) / (exp((v-V_T-40.*mV)/(5.*mV))-1 ) * 1/msecond : Hz
        alpha_h = 0.128 * exp(-(v-V_T-17.*mV)/(18.*mV)) * 1/msecond : Hz
        beta_h = 4./(1.+exp(-(v-V_T-40.*mV)/(5.*mV))) * 1/msecond : Hz
        p_inf = 1 / (1.+exp(-(v+35.*mV)/(10.*mV))) : 1
        tau_p = tau_max_M /( (3.3*exp((v+35.*mV)/(20.*mV)) + exp(-(v+35.*mV)/(20.*mV)))) : second
        
        IK    = -gK * n**4 *(v-EK_ctx) : ampere/meter**2
        INa   = -gNa*m**3*h*(v-ENa_ctx)   : ampere/meter**2
        IM    = -gM *p *(v-EM_ctx)  : ampere/meter**2
        Ileak = -gleak*(v-Eleak): ampere/meter**2
        
        dgaba1/dt = -gaba1/taud_g + 1/2 * (1 + (tanh(v/slope_g)))* (1-gaba1)/taur_g : 1 
        dampa1/dt = -ampa1/taud_a + 1/2 * (1 + (tanh(v/slope_a)))* (1-ampa1)/taur_a : 1 

        Igaba = -gsyn_g * gaba2 * (v-Esyn_g): amp/meter**2
        Iampa = -gsyn_a * ampa2 * (v-Esyn_a): amp/meter**2
        
        noise_std = 5 * mA/meter**2 : amp/meter**2
        noise_mean = 0 * nA/meter**2 : amp/meter**2
        random_var : 1
        Inoise = noise_mean + noise_std * random_var: amp/meter**2
'''
        
spike_detect = 0*mV
VPM_neuron_num = 70
L4E_neuron_num = 188;
L4I_neuron_num = 33;

TC_cells = NeuronGroup(VPM_neuron_num, VPM, threshold = 'v > spike_detect', refractory = 'v > spike_detect', method='rk4')
TC_cells.v = -65*mV + (10 * np.random.rand(1, VPM_neuron_num) * mV) # Resting potential

TC_cells.m = np.finfo(float).eps * np.random.rand(1, VPM_neuron_num)
TC_cells.h = np.finfo(float).eps * np.random.rand(1, VPM_neuron_num)
TC_cells.n = np.finfo(float).eps * np.random.rand(1, VPM_neuron_num)
TC_cells.hT = np.finfo(float).eps * np.random.rand(1, VPM_neuron_num)
TC_cells.o1 = np.finfo(float).eps * np.random.rand(1, VPM_neuron_num)
TC_cells.c1 = np.finfo(float).eps * np.random.rand(1, VPM_neuron_num)
TC_cells.p0 = np.finfo(float).eps * np.random.rand(1, VPM_neuron_num)
TC_cells.CaBuffer = np.finfo(float).eps * np.random.rand(1, VPM_neuron_num)

L4E_cells = NeuronGroup(L4E_neuron_num, CTX, threshold = 'v > spike_detect', refractory = 'v > spike_detect', method = 'rk4')

L4E_cells.gleak    =   [RS_gleak]
L4E_cells.Eleak    =   [RS_Eleak]
L4E_cells.V_T      =   [RS_V_T]
L4E_cells.gNa      =   [RS_gNa]
L4E_cells.gK       =   [RS_gK]
L4E_cells.gM       =   [RS_gM]
L4E_cells.tau_max_M =  [RS_tau_max_M]
L4E_cells.v        =   -65*mV + (5 * np.random.rand(1, L4E_neuron_num) * mV) # Resting potential

L4I_cells = NeuronGroup(L4I_neuron_num, CTX, threshold = 'v > spike_detect', refractory = 'v > spike_detect', method = 'rk4')

L4I_cells.gleak    =   [FS_gleak]
L4I_cells.Eleak    =   [FS_Eleak]
L4I_cells.V_T      =   [FS_V_T]
L4I_cells.gNa      =   [FS_gNa]
L4I_cells.gK       =   [FS_gK]
L4I_cells.gM       =   [FS_gM]
L4I_cells.tau_max_M =  [FS_tau_max_M]
L4I_cells.v        =   -65*mV + (5 * np.random.rand(1, L4I_neuron_num) * mV) # Resting potential

# AMPA params
Esyn_AMPA = 0       * mV  
gsyn_AMPA = 0.1 * 0.17    * msiemens/cm**2
slope_AMPA = 10     * mvolt
taud_AMPA = 2       * ms
taur_AMPA = 0.1     * ms

# GABA params
Esyn_GABA = -90         * mV 
gsyn_GABA = 0.25 * 0.5   * msiemens/cm**2 
taud_GABA = 10      * ms
taur_GABA = 0.2     * ms
slope_GABA = 10     * mvolt

# Define dynamics
VPM_to_L4E = Synapses(TC_cells, L4E_cells, on_pre = 'ampa2_post = 7 * s1_pre')
VPM_to_L4I = Synapses(TC_cells, L4I_cells, on_pre = 'ampa2_post = 9 * s1_pre')

L4E_to_L4E = Synapses(L4E_cells, L4E_cells, on_pre = 'ampa2_post = 3.3 * ampa1_pre')
L4E_to_L4I = Synapses(L4E_cells, L4I_cells, on_pre = 'ampa2_post = 7.9 * ampa1_pre')
L4I_to_L4E = Synapses(L4I_cells, L4E_cells, on_pre = 'gaba2_post = 16 * gaba1_pre')
L4I_to_L4I = Synapses(L4I_cells, L4I_cells, on_pre = 'gaba2_post = 14 * gaba1_pre')

# Create connections
VPM_to_L4E.connect(p=0.043)
VPM_to_L4I.connect(p=0.05)

L4E_to_L4E.connect(p=0.076)
L4E_to_L4I.connect(p=0.042)
L4I_to_L4E.connect(p=0.063)
L4I_to_L4I.connect(p=0.062)

# Initial parameters
TC_cells.taud     = [taud_AMPA]
TC_cells.taur     = [taur_AMPA]
TC_cells.slope    = [slope_AMPA]

L4E_cells.taud_g     = [taud_GABA]
L4E_cells.taur_g     = [taur_GABA]
L4E_cells.slope_g    = [slope_GABA]
L4E_cells.gsyn_g     = [gsyn_GABA]
L4E_cells.Esyn_g     = [Esyn_GABA]
L4E_cells.gaba2       = [0]
L4E_cells.taud_a     = [taud_AMPA]
L4E_cells.taur_a     = [taur_AMPA]
L4E_cells.slope_a    = [slope_AMPA]
L4E_cells.gsyn_a     = [gsyn_AMPA]
L4E_cells.Esyn_a     = [Esyn_AMPA]
L4E_cells.ampa2       = [0]

L4I_cells.taud_g     = [taud_GABA]
L4I_cells.taur_g     = [taur_GABA]
L4I_cells.slope_g    = [slope_GABA]
L4I_cells.gsyn_g     = [gsyn_GABA]
L4I_cells.Esyn_g     = [Esyn_GABA]
L4I_cells.gaba2       = [0]
L4I_cells.taud_a     = [taud_AMPA]
L4I_cells.taur_a     = [taur_AMPA]
L4I_cells.slope_a    = [slope_AMPA]
L4I_cells.gsyn_a     = [gsyn_AMPA]
L4I_cells.Esyn_a     = [Esyn_AMPA]
L4I_cells.ampa2       = [0]

##### Input #####
@network_operation(dt=1*ms)
def update_I(): # noise to single L4 neurons
    L4E_cells.random_var[:L4E_neuron_num] = np.random.normal(0, 1, L4E_neuron_num)
    L4I_cells.random_var[:L4I_neuron_num] = np.random.normal(0, 1, L4I_neuron_num)

GeneratedSpikes = SpikeGeneratorGroup(BrainstemNeurons,  # number of spiking sources
      indices=np.array(NeuronID_Dir90["NeuronID"]), # which neurons will make each spike
      times=np.array(Spikes_Dir90["Spike_times"]*1000)*ms) # spike times for each neuron

VPMinput = Synapses(GeneratedSpikes, TC_cells, on_pre='v += 1.5*mV')
VPMinput.connect(p=0.35)

# simulate
VPM_activity = StateMonitor(TC_cells, variables=['v'], record=True)
VPM_spikes = SpikeMonitor(TC_cells)

L4E_activity = StateMonitor(L4E_cells, variables=['v'], record=True)
L4E_spikes = SpikeMonitor(L4E_cells)

L4I_activity = StateMonitor(L4I_cells, variables=['v'], record=True)
L4I_spikes = SpikeMonitor(L4I_cells)

run(SimulationDuration*ms)

# plot
figure(figsize=(10, 8))
for i in range(30):  
    subplot(611)
    plot(VPM_activity.t/ms, VPM_activity.v[i]/mV)
    title('VPM')
    ylabel('Voltage (mV)')
    xlim([0, SimulationDuration])
    
    subplot(613)
    plot(L4E_activity.t/ms, L4E_activity.v[i]/mV)
    title('L4E')
    ylabel('Voltage (mV)')
    xlim([0, SimulationDuration])
    
    subplot(615)
    plot(L4I_activity.t/ms, L4I_activity.v[i]/mV)
    title('L4I')
    ylabel('Voltage (mV)')
    xlim([0, SimulationDuration])
    
subplot(612)
plt.plot(VPM_spikes.t/ms, VPM_spikes.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron')
plt.xlim([0, SimulationDuration])

subplot(614)
plt.plot(L4E_spikes.t/ms, L4E_spikes.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron')
plt.xlim([0, SimulationDuration])

subplot(616)
plt.plot(L4I_spikes.t/ms, L4I_spikes.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron')
plt.xlim([0, SimulationDuration])
plt.show()

