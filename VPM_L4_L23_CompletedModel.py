#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 23:17:18 2025

@author: teatompos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A population of VPM neurons

with

a population of L4 neurons
"""

from brian2 import *
import numpy as np
from scipy import constants as constants
import matplotlib.pyplot as plt

# System parameters
defaultclock.dt = 0.01*ms

########### Constants for VPM cells ###########
ENa_vpm = 50*mV 
EK_vpm = -100*mV 

Eh_vpm = -43*mV 
Eleak_vpm = -70*mV 
EKleak_vpm = -100*mV

gNa_vpm = 90*msiemens/cm**2
gK_vpm = 20*msiemens/cm**2
gCa_vpm = 2*msiemens/cm**2
gh_vpm = 0.01*msiemens/cm**2
gleak_vpm = 0.01*msiemens/cm**2
gKleak_vpm = 0.0172*msiemens/cm**2

FaradConst = constants.physical_constants['Faraday constant'][0] * coulomb/mole
GasConst = constants.gas_constant * joule / (kelvin * mole)

########### Constants for CTX/L4 cells ###########

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

########### Equations ###########
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
        
########### Populations ###########
spike_detect = 0*mV
VPM_neuron_num = 40;
L4E_neuron_num = 188;
L4I_neuron_num = 33;
L3E_neuron_num = 158;
L3I_neuron_num = 28;
L2E_neuron_num = 86;
L2I_neuron_num = 15;

# Create VPM population
TC_cells = NeuronGroup(VPM_neuron_num, VPM, threshold = 'v > spike_detect', refractory = 'v > spike_detect', method='rk4')
TC_cells.v = -65*mV + (10 * np.random.rand(1, VPM_neuron_num) * mV) # Resting potential

# TC_cells.m = 0.05 + 0.1 * np.random.rand(1, VPM_neuron_num)
# TC_cells.h = 0.54 + 0.1 * np.random.rand(1, VPM_neuron_num)
# TC_cells.n = 0.1 * np.random.rand(1, VPM_neuron_num)
# TC_cells.hT = 0.34 + 0.1 * np.random.rand(1, VPM_neuron_num)
TC_cells.CaBuffer = np.finfo(float).eps
# TC_cells.CaBuffer = 0.0001 * np.random.rand(1, VPM_neuron_num)
# TC_cells.o1 = 0 * np.random.rand(1, VPM_neuron_num)
# TC_cells.c1 = 0.5 * np.random.rand(1, VPM_neuron_num)
# TC_cells.p0 = 0.5 * np.random.rand(1, VPM_neuron_num)

# Create L4E population
L4E_cells = NeuronGroup(L4E_neuron_num, CTX, threshold = 'v > spike_detect', refractory = 'v > spike_detect', method = 'rk4')

L4E_cells.gleak    =   [RS_gleak]
L4E_cells.Eleak    =   [RS_Eleak]
L4E_cells.V_T      =   [RS_V_T]
L4E_cells.gNa      =   [RS_gNa]
L4E_cells.gK       =   [RS_gK]
L4E_cells.gM       =   [RS_gM]
L4E_cells.tau_max_M =  [RS_tau_max_M]
L4E_cells.v        =   [-65] * mV

# Create L4I population
L4I_cells = NeuronGroup(L4I_neuron_num, CTX, threshold = 'v > spike_detect', refractory = 'v > spike_detect', method = 'rk4')

L4I_cells.gleak    =   [FS_gleak]
L4I_cells.Eleak    =   [FS_Eleak]
L4I_cells.V_T      =   [FS_V_T]
L4I_cells.gNa      =   [FS_gNa]
L4I_cells.gK       =   [FS_gK]
L4I_cells.gM       =   [FS_gM]
L4I_cells.tau_max_M =  [FS_tau_max_M]
L4I_cells.v        =   [-65] * mV

# Create L3E population
L3E_cells = NeuronGroup(L3E_neuron_num, CTX, threshold = 'v > spike_detect', refractory = 'v > spike_detect', method = 'rk4')

L3E_cells.gleak    =   [RS_gleak]
L3E_cells.Eleak    =   [RS_Eleak]
L3E_cells.V_T      =   [RS_V_T]
L3E_cells.gNa      =   [RS_gNa]
L3E_cells.gK       =   [RS_gK]
L3E_cells.gM       =   [RS_gM]
L3E_cells.tau_max_M =  [RS_tau_max_M]
L3E_cells.v        =   [-65] * mV

# Create L3I population
L3I_cells = NeuronGroup(L3I_neuron_num, CTX, threshold = 'v > spike_detect', refractory = 'v > spike_detect', method = 'rk4')

L3I_cells.gleak    =   [FS_gleak]
L3I_cells.Eleak    =   [FS_Eleak]
L3I_cells.V_T      =   [FS_V_T]
L3I_cells.gNa      =   [FS_gNa]
L3I_cells.gK       =   [FS_gK]
L3I_cells.gM       =   [FS_gM]
L3I_cells.tau_max_M =  [FS_tau_max_M]
L3I_cells.v        =   [-65] * mV


# Create L2E population
L2E_cells = NeuronGroup(L2E_neuron_num, CTX, threshold = 'v > spike_detect', refractory = 'v > spike_detect', method = 'rk4')

L2E_cells.gleak    =   [RS_gleak]
L2E_cells.Eleak    =   [RS_Eleak]
L2E_cells.V_T      =   [RS_V_T]
L2E_cells.gNa      =   [RS_gNa]
L2E_cells.gK       =   [RS_gK]
L2E_cells.gM       =   [RS_gM]
L2E_cells.tau_max_M =  [RS_tau_max_M]
L2E_cells.v        =   [-65] * mV

# Create L2I population
L2I_cells = NeuronGroup(L2I_neuron_num, CTX, threshold = 'v > spike_detect', refractory = 'v > spike_detect', method = 'rk4')

L2I_cells.gleak    =   [FS_gleak]
L2I_cells.Eleak    =   [FS_Eleak]
L2I_cells.V_T      =   [FS_V_T]
L2I_cells.gNa      =   [FS_gNa]
L2I_cells.gK       =   [FS_gK]
L2I_cells.gM       =   [FS_gM]
L2I_cells.tau_max_M =  [FS_tau_max_M]
L2I_cells.v        =   [-65] * mV


########### Connections ###########
# AMPA params
Esyn_AMPA = 0       * mV  
gsyn_AMPA = 0.1 * 0.17    * msiemens/cm**2
slope_AMPA = 10     * mvolt
taud_AMPA = 2       * ms
taur_AMPA = 0.1     * ms

# GABA params
Esyn_GABA = -80         * mV 
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

L4E_to_L3E = Synapses(L4E_cells, L3E_cells, on_pre = 'ampa2_post = 2.4 * ampa1_pre')
L4E_to_L3I = Synapses(L4E_cells, L3I_cells, on_pre = 'ampa2_post = 5.7 * ampa1_pre')
L4I_to_L3E = Synapses(L4I_cells, L3E_cells, on_pre = 'gaba2_post = 14 * gaba1_pre')

L3E_to_L3E = Synapses(L3E_cells, L3E_cells, on_pre = 'ampa2_post = 2.8 * ampa1_pre')
L3E_to_L3I = Synapses(L3E_cells, L3I_cells, on_pre = 'ampa2_post = 8.1 * ampa1_pre')
L3I_to_L3E = Synapses(L3I_cells, L3E_cells, on_pre = 'gaba2_post = 17 * gaba1_pre')
L3I_to_L3I = Synapses(L3I_cells, L3I_cells, on_pre = 'gaba2_post = 15 * gaba1_pre')

L3E_to_L2E = Synapses(L3E_cells, L2E_cells, on_pre = 'ampa2_post = 2.8 * ampa1_pre')
L3E_to_L2I = Synapses(L3E_cells, L2I_cells, on_pre = 'ampa2_post = 8.1 * ampa1_pre')
L3I_to_L2E = Synapses(L3I_cells, L2E_cells, on_pre = 'gaba2_post = 17 * gaba1_pre')
L3I_to_L2I = Synapses(L3I_cells, L2I_cells, on_pre = 'gaba2_post = 15 * gaba1_pre')

L2E_to_L2E = Synapses(L2E_cells, L2E_cells, on_pre = 'ampa2_post = 2.8 * ampa1_pre')
L2E_to_L2I = Synapses(L2E_cells, L2I_cells, on_pre = 'ampa2_post = 8.1 * ampa1_pre')
L2I_to_L2E = Synapses(L2I_cells, L2E_cells, on_pre = 'gaba2_post = 17 * gaba1_pre')
L2I_to_L2I = Synapses(L2I_cells, L2I_cells, on_pre = 'gaba2_post = 15 * gaba1_pre')

# Create connections
VPM_to_L4E.connect(p=0.043)
VPM_to_L4I.connect(p=0.05)

L4E_to_L4E.connect(p=0.076)
L4E_to_L4I.connect(p=0.042)
L4I_to_L4E.connect(p=0.063)
L4I_to_L4I.connect(p=0.062)

L4E_to_L3E.connect(p=0.058)
L4E_to_L3I.connect(p=0.0033)
L4I_to_L3E.connect(p=0.017)

L3E_to_L3E.connect(p=0.056)
L3E_to_L3I.connect(p=0.051)
L3I_to_L3E.connect(p=0.059)
L3I_to_L3I.connect(p=0.051)

L3E_to_L2E.connect(p=0.056)
L3E_to_L2I.connect(p=0.051)
L3I_to_L2E.connect(p=0.059)
L3I_to_L2I.connect(p=0.051)

L2E_to_L2E.connect(p=0.056)
L2E_to_L2I.connect(p=0.051)
L2I_to_L2E.connect(p=0.059)
L2I_to_L2I.connect(p=0.051)

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

L3E_cells.taud_g     = [taud_GABA]
L3E_cells.taur_g     = [taur_GABA]
L3E_cells.slope_g    = [slope_GABA]
L3E_cells.gsyn_g     = [gsyn_GABA]
L3E_cells.Esyn_g     = [Esyn_GABA]
L3E_cells.gaba2       = [0]
L3E_cells.taud_a     = [taud_AMPA]
L3E_cells.taur_a     = [taur_AMPA]
L3E_cells.slope_a    = [slope_AMPA]
L3E_cells.gsyn_a     = [gsyn_AMPA]
L3E_cells.Esyn_a     = [Esyn_AMPA]
L3E_cells.ampa2       = [0]

L3I_cells.taud_g     = [taud_GABA]
L3I_cells.taur_g     = [taur_GABA]
L3I_cells.slope_g    = [slope_GABA]
L3I_cells.gsyn_g     = [gsyn_GABA]
L3I_cells.Esyn_g     = [Esyn_GABA]
L3I_cells.gaba2       = [0]
L3I_cells.taud_a     = [taud_AMPA]
L3I_cells.taur_a     = [taur_AMPA]
L3I_cells.slope_a    = [slope_AMPA]
L3I_cells.gsyn_a     = [gsyn_AMPA]
L3I_cells.Esyn_a     = [Esyn_AMPA]
L3I_cells.ampa2       = [0]

L2E_cells.taud_g     = [taud_GABA]
L2E_cells.taur_g     = [taur_GABA]
L2E_cells.slope_g    = [slope_GABA]
L2E_cells.gsyn_g     = [gsyn_GABA]
L2E_cells.Esyn_g     = [Esyn_GABA]
L2E_cells.gaba2       = [0]
L2E_cells.taud_a     = [taud_AMPA]
L2E_cells.taur_a     = [taur_AMPA]
L2E_cells.slope_a    = [slope_AMPA]
L2E_cells.gsyn_a     = [gsyn_AMPA]
L2E_cells.Esyn_a     = [Esyn_AMPA]
L2E_cells.ampa2       = [0]

L2I_cells.taud_g     = [taud_GABA]
L2I_cells.taur_g     = [taur_GABA]
L2I_cells.slope_g    = [slope_GABA]
L2I_cells.gsyn_g     = [gsyn_GABA]
L2I_cells.Esyn_g     = [Esyn_GABA]
L2I_cells.gaba2       = [0]
L2I_cells.taud_a     = [taud_AMPA]
L2I_cells.taur_a     = [taur_AMPA]
L2I_cells.slope_a    = [slope_AMPA]
L2I_cells.gsyn_a     = [gsyn_AMPA]
L2I_cells.Esyn_a     = [Esyn_AMPA]
L2I_cells.ampa2       = [0]

########### Input ###########
I_amp = 4

# Stimulation parameters
start_time = 200
end_time = 500
stim_ms = 50
 
@network_operation(dt=1*ms)
def update_I():
    TC_cells.Iapp[:VPM_neuron_num] = Istim(defaultclock.t)
    L4E_cells.random_var[:L4E_neuron_num] = np.random.normal(0, 1, L4E_neuron_num)
    L4I_cells.random_var[:L4I_neuron_num] = np.random.normal(0, 1, L4I_neuron_num)
    L3E_cells.random_var[:L3E_neuron_num] = np.random.normal(0, 1, L3E_neuron_num)
    L3I_cells.random_var[:L3I_neuron_num] = np.random.normal(0, 1, L3I_neuron_num)
    L2E_cells.random_var[:L2E_neuron_num] = np.random.normal(0, 1, L2E_neuron_num)
    L2I_cells.random_var[:L2I_neuron_num] = np.random.normal(0, 1, L2I_neuron_num)
   
# Create a time array for the current input
times = np.concatenate([np.zeros(start_time), np.ones(stim_ms), np.zeros(end_time - start_time - stim_ms)])

# Create the TimedArray
Istim = TimedArray(times * (I_amp * uA/cm**2), dt=1*ms)

########### Simulation ###########

# Set up monitors and run the simulation
VPM_activity = StateMonitor(TC_cells, variables=['v'], record=True)
L4E_activity = StateMonitor(L4E_cells, variables=['v'], record=True)
L4I_activity = StateMonitor(L4I_cells, variables=['v', 'Inoise'], record=True)
L3E_activity = StateMonitor(L3E_cells, variables=['v'], record=True)
L3I_activity = StateMonitor(L4I_cells, variables=['v', 'Inoise'], record=True)
L2E_activity = StateMonitor(L2E_cells, variables=['v'], record=True)
L2I_activity = StateMonitor(L2I_cells, variables=['v', 'Inoise'], record=True)

run(end_time*ms)

########### Figures ###########
figure(figsize=(10, 6))
for neuron in range(15):
    subplot(421)
    plot(VPM_activity.t/ms, VPM_activity.v[neuron]/mV)
    
    subplot(423)
    plot(L4E_activity.t/ms, L4E_activity.v[neuron]/mV)
    
    subplot(424)
    plot(L4I_activity.t/ms, L4I_activity.v[neuron]/mV)
    
    subplot(425)
    plot(L3E_activity.t/ms, L3E_activity.v[neuron]/mV)
    
    subplot(426)
    plot(L3I_activity.t/ms, L3I_activity.v[neuron]/mV)
    
    subplot(427)
    plot(L2E_activity.t/ms, L2E_activity.v[neuron]/mV)
    
    subplot(428)
    plot(L2I_activity.t/ms, L2I_activity.v[neuron]/mV)
    
figure(figsize=(10, 6))
plot(L4I_activity.t/ms, L4I_activity.Inoise[1]/mV)
