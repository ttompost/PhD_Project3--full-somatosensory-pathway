#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AA population of VPM neurons receives brainstem spikes.
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
        '''
        
spike_detect = 0*mV
VPM_neuron_num = 70

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

GeneratedSpikes = SpikeGeneratorGroup(BrainstemNeurons,  # number of spiking sources
      indices=np.array(NeuronID_Dir90["NeuronID"]), # which neurons will make each spike
      times=np.array(Spikes_Dir90["Spike_times"]*1000)*ms) # spike times for each neuron

VPMinput = Synapses(GeneratedSpikes, TC_cells, on_pre='v += 1.5*mV')
VPMinput.connect(p=0.35)

# simulate
VPM_activity = StateMonitor(TC_cells, variables=['v'], record=True)
VPM_spikes = SpikeMonitor(TC_cells)

run(SimulationDuration*ms)

# plot
figure(figsize=(10, 6))
subplot(211)
for i in range(VPM_neuron_num):    
    plot(VPM_activity.t/ms, VPM_activity.v[i]/mV)
    
title('Spiking neuron')
ylabel('Voltage (mV)')
xlim([0, SimulationDuration])
    
subplot(212)
plt.plot(VPM_spikes.t/ms, VPM_spikes.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.title('Raster Plot')
plt.xlim([0, SimulationDuration])
plt.show()

