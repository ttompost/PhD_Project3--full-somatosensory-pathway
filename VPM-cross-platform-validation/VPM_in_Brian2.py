#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:48:21 2024

@author: teatompos
"""

from brian2 import *
import numpy as np
from scipy import constants as constants
import matplotlib.pyplot as plt

# Constants and equations for the TC cells
ENa = 50*mV 
EK = -100*mV 

Eh = -43*mV 
Eleak = -70*mV 
EKleak = -100*mV

gNa = 90*msiemens/cm**2
gK = 20*msiemens/cm**2
gCa = 2*msiemens/cm**2
gh = 0.01*msiemens/cm**2
gleak = 0.01*msiemens/cm**2
gKleak = 0.0172*msiemens/cm**2

FaradConst = constants.physical_constants['Faraday constant'][0] * coulomb/mole
GasConst = constants.gas_constant * joule / (kelvin * mole)

# System parameters
defaultclock.dt = 0.01*ms

eqs = '''
        dv/dt = (Iapp - INa - IK - IT - Ih - ILeak - IKleak)/Cm : volt
        
        INa = gNa * m**3 * h * (v - ENa) : amp/meter**2
        IK = gK * n**4 * (v - EK) : amp/meter**2
        ILeak = gleak * (v - Eleak) : amp/meter**2
        IKleak = gKleak * (v - EKleak) : amp/meter**2
        IT = gCa * mT_inf**2 * hT * (v - ECa) : amp/meter**2
        Ih = gh * (o1 + 2*(1 - c1 - o1)) * (v - Eh) : amp/meter**2
        
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
        
# Create TC cells
VPM_neuron_num = 1;
spike_detect = 0*mV

# Stimulation parameters
start_time = 200
end_time = 500
stim_ms = 50

I_amps = [0, 0.66667, 1.33333, 2, 3.33333, 5.33333, 8, 13.33333]

# Plot results
fig, axes = plt.subplots(4, 2, figsize=(10, 6))

for i, (I_amp, ax) in enumerate(zip(I_amps, axes.ravel())):  
    
    TC_cells = NeuronGroup(VPM_neuron_num, eqs, threshold = 'v > spike_detect', refractory = 'v > spike_detect', method='rk4')
    TC_cells.v = -65*mV  # Resting potential

    TC_cells.m = 0.05 + 0.1 * np.random.rand(1, VPM_neuron_num)
    TC_cells.h = 0.54 + 0.1 * np.random.rand(1, VPM_neuron_num)
    TC_cells.n = 0.1 * np.random.rand(1, VPM_neuron_num)
    TC_cells.hT = 0.34 + 0.1 * np.random.rand(1, VPM_neuron_num)
    TC_cells.CaBuffer = 0.0001 * np.random.rand(1, VPM_neuron_num)
    TC_cells.o1 = 0 * np.random.rand(1, VPM_neuron_num)
    TC_cells.c1 = 0.5 * np.random.rand(1, VPM_neuron_num)
    TC_cells.p0 = 0.5 * np.random.rand(1, VPM_neuron_num)

    @network_operation(dt=1*ms)
    def update_I():
        TC_cells.Iapp[:VPM_neuron_num] = Istim(defaultclock.t)
   
    # Create a time array for the current input
    times = np.concatenate([np.zeros(start_time), np.ones(stim_ms), np.zeros(end_time - start_time - stim_ms)])
    
    # Create the TimedArray
    Istim = TimedArray(times * (I_amp * uA/cm**2), dt=1*ms)

    # Set up monitors and run the simulation
    M = StateMonitor(TC_cells, variables=['v'], record=True)
    run(end_time*ms)
    
    ax.plot(M.t/ms, M.v[0]/mV)
    ax.set_title(f"Iapp = {I_amp} uA/cm2")
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    
plt.tight_layout()
plt.show()