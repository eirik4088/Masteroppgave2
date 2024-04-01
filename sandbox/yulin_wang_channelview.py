import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import mne
import pathlib
import sklearn
from meegkit import dss

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


data_folder = pathlib.Path(r"C:\Users\workbench\eirik_master\Data\yulin_wang")
file = data_folder / "sub-03_ses-session1_task-eyesclosed_eeg.vhdr"

raw= mne.io.read_raw_brainvision(file, preload=True)
raw.set_montage('biosemi64', on_missing='warn')

raw_highpass = raw.copy().filter(l_freq=1, h_freq=None, verbose=False)
raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=100, verbose=False)
line_noise = zapline_clean(raw_lowpass, 50)
raw_down_sampled = line_noise.copy().resample(sfreq=200, verbose=False)

raw_down_sampled.set_eeg_reference()
raw_down_sampled.plot(block=True)
