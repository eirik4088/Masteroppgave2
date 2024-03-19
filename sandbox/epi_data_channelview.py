import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import mne
import pathlib

import seaborn as sns
import statistics
from meegkit import dss

from pycrostates.preprocessing import extract_gfp_peaks
from neurokit2.microstates.microstates_clean import microstates_clean
from mne_icalabel import label_components

def zapline_clean(raw, fline):
    data = raw.get_data().T # Convert mne data to numpy darray
    sfreq = raw.info['sfreq'] # Extract the sampling freq
   
    #Apply MEEGkit toolbox function
    out, _ = dss.dss_line(data, fline, sfreq, nkeep=1) # fline (Line noise freq) = 50 Hz for Europe
    print(out.shape)

    cleaned_raw = mne.io.RawArray(out.T, raw.info) # Convert output to mne RawArray again

    return cleaned_raw

data_folder = pathlib.Path(r"C:\Users\workbench\eirik_master\Data\epi_data")
file = data_folder / "TLE_0012.bdf"

raw = mne.io.read_raw_bdf(file, verbose=False)
raw.crop((0.4*60)+64, (0.4*60)+64+60).load_data()
raw.drop_channels(['SO2', 'IO2', 'LO1', 'LO2', 'EXG5', 'EXG6', 'EXG7', 'EXG8'])
raw.set_montage('biosemi128')
raw.set_eeg_reference()

raw_highpass = raw.copy().filter(l_freq=1, h_freq=None, verbose=False)
raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=100, verbose=False)
line_noise = zapline_clean(raw_lowpass, 50)
raw_down_sampled = line_noise.copy().resample(sfreq=200, verbose=False)

raw_down_sampled.plot(block=True)
