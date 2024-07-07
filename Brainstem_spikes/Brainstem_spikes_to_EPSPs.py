#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 11:59:12 2024

@author: teatompos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NeuronID_Amp25 = pd.read_csv("/Users/teatompos/Documents/GitHub/PhD_Project3--full-somatosensory-pathway/Brainstem_spikes/amplitude_25_id.csv", header=None)
NeuronID_Amp25.columns = ["NeuronID"]
Spikes_Amp25 = pd.read_csv("/Users/teatompos/Documents/GitHub/PhD_Project3--full-somatosensory-pathway/Brainstem_spikes/amplitude_25_time.csv", header=None)
Spikes_Amp25.columns = ["Spike_times"]

BrainstemNeurons = range(0, 70)

NeuronID_idx = [None] * len(BrainstemNeurons)
for neuronID in BrainstemNeurons: 
    # find spike-time indices for each neuron
    indices = NeuronID_Amp25[NeuronID_Amp25["NeuronID"] == neuronID].index.tolist()
    NeuronID_idx[neuronID] = np.array(indices)
        
NeuronSpikes = [None] * len(BrainstemNeurons)
for neuronID in BrainstemNeurons: 
    # extract spike-times for each neuron
    spikeIdices = NeuronID_idx[neuronID]
    spikeTimes = []
    for spkIdx in spikeIdices:
        spikeTimes.append(Spikes_Amp25["Spike_times"][spkIdx])
    NeuronSpikes[neuronID] = np.array(spikeTimes)


TimeVector = np.arange(0,0.4,0.001)

NeuronSignals = [None] * len(BrainstemNeurons)
for neuronID in BrainstemNeurons: 
    spikeTimes = NeuronSpikes[neuronID]
    signal = [0] * len(TimeVector)
    for spkTime in spikeTimes:
        # idx = np.where(TimeVector == spkTime)[0]
        # signal[int(idx)] = 1
        signal[int(spkTime*1000)] = 1
        
    NeuronSignals[neuronID] = np.array(signal)
    
tau_i = 10
tau_1 = 1
tau_d = 2
tau_r = 0.5
eps = np.finfo(float).eps
t = np.asarray(TimeVector * 1000) # in ms

psp = tau_i * (np.exp(-np.maximum(t - tau_1, 0) / tau_d) -
                 np.exp(-np.maximum(t - tau_1, 0) / tau_r)) / (tau_d - tau_r)
psp = psp[psp > eps]
psp = np.concatenate([np.zeros(len(psp)), psp])

EPSPsignals = [None] * len(BrainstemNeurons)
for neuronID in BrainstemNeurons: 
    EPSPsignals[neuronID] = np.array(np.convolve(NeuronSignals[neuronID], psp, mode='same'))

plt.figure()
plt.subplot(211)
plt.plot(TimeVector, NeuronSignals[0])
plt.subplot(212)
plt.plot(TimeVector, EPSPsignals[0])
plt.show()


#### digested code

# tau_i, tau_1, tau_d, tau_r  = 10, 1, 2, 0.5
# eps = np.finfo(float).eps
# t = np.asarray(TimeVector) # in ms
# I_amp =  180 * nA/cm**2

# psp = tau_i * (np.exp(-np.maximum(t - tau_1, 0) / tau_d) - np.exp(-np.maximum(t - tau_1, 0) / tau_r)) / (tau_d - tau_r)
# psp = psp[psp > eps]
# psp = np.concatenate([np.zeros(len(psp)), psp])
# psp = psp / np.max(psp) # normalize psp for easier input control

# EPSPsignals = [None] * len(BrainstemNeurons)
# NeuronSignals = [None] * len(BrainstemNeurons)
# SpikeTimes = [None] * len(BrainstemNeurons)
# for neuronID in BrainstemNeurons: 
#     indices = np.array(NeuronID_Dir90[NeuronID_Dir90["NeuronID"] == neuronID].index.tolist())

#     signal = [0] * len(TimeVector)
#     spkTimes = []
#     for spkIdx in indices:
#         spikeTime = Spikes_Dir90["Spike_times"][spkIdx]
#         spkTimes.append(spikeTime*1000)
        
#         signal[int(spikeTime*1000)] = 1
        
#     SpikeTimes[neuronID] = np.array(spkTimes) # ms
#     NeuronSignals[neuronID] = np.array(signal)
#     EPSPsignals[neuronID] = np.array(np.convolve(NeuronSignals[neuronID], psp, mode='same'))
