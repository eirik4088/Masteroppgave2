import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import mne
import pathlib
import sklearn
import hdbscan
import seaborn as sns
import statistics
from meegkit import dss

from pycrostates.preprocessing import extract_gfp_peaks
from proof_of_consept import unit_normalize
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


data_folder = pathlib.Path(r"C:\Users\Gulbr\MasterOppgave\Data\SRM_rs_EEG_OpenNeuro")
file = data_folder / "sub-001_ses-t1_task-resteyesc_eeg.edf"

raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
mne.set_eeg_reference(raw, copy=False, verbose=False)
#Be carefull that artifact electrodes are not included in the average calc.
raw.set_montage('biosemi64')


raw_highpass = raw.copy().filter(l_freq=1, h_freq=None, verbose=False)
raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=100, verbose=False)
line_noise = zapline_clean(raw_lowpass, 50)
raw_down_sampled = line_noise.copy().resample(sfreq=200, verbose=False)
#I should maybe downsample, and maybe do it first for computational efficiency. The only thing is that then almost all the data will be in simpochs, so maybe I need stricter limits for gfp peakes.
#Bad segments can maybe be identified with Christoffers algorithm, anything else? I want to do this at the end of the pipeline I think.

"""epocs_ica = mne.make_fixed_length_epochs(raw_down_sampled, verbose=False, preload=True, overlap=0.5)
epocs_ica.drop([201, 202, 203, 219, 220, 221])
ica_a = mne.preprocessing.ICA(max_iter="auto",
    method="infomax",
    random_state=97,
    fit_params=dict(extended=True),
    )
ica_a.fit(epocs_ica)
#Do I feed enough data to ICA?
ic_a_labels = label_components(raw_down_sampled, ica_a, method='iclabel')
ica_a.plot_components()
components = []
for ic in range(63):
    if ic_a_labels['y_pred_proba'][ic] > 0.8:
        if ic_a_labels['labels'][ic] == 'eye blink' or ic_a_labels['labels'][ic] == 'muscle artifact':
            components.append(ic)
            print(ic_a_labels['labels'][ic], ic_a_labels['y_pred_proba'][ic])
ica_a.exclude = components
reconstruction = ica_a.apply(epocs_ica)
reconstruction.plot(block=True"""

raw_down_sampled.plot(block=False)

#data, indices, gfp, info_mne  = microstates_clean(raw_down_sampled, standardize_eeg=False, normalize=False, gfp_method='l2')
#data = (data/gfp)/100000

#raw_down_sampled = mne.io.RawArray(data, raw_down_sampled.info)

#raw_down_sampled.plot(block=False)


data_folder = pathlib.Path(r"C:\Users\Gulbr\MasterOppgave\Data\SRM_rs_EEG_OpenNeuro")
file = data_folder / "sub-002_ses-t1_task-resteyesc_eeg.edf"


raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
mne.set_eeg_reference(raw, copy=False, verbose=False)
#Be carefull that artifact electrodes are not included in the average calc.
raw.set_montage('biosemi64')


raw_highpass = raw.copy().filter(l_freq=1, h_freq=None, verbose=False)
raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=100, verbose=False)
line_noise = zapline_clean(raw_lowpass, 50)
raw_down_sampled = line_noise.copy().resample(sfreq=200, verbose=False)
#I should maybe downsample, and maybe do it first for computational efficiency. The only thing is that then almost all the data will be in simpochs, so maybe I need stricter limits for gfp peakes.
#Bad segments can maybe be identified with Christoffers algorithm, anything else? I want to do this at the end of the pipeline I think.

raw_down_sampled.plot(block=True)

exit()

data_folder = pathlib.Path(r"C:\Users\Gulbr\MasterOppgave\SRM_rs_EEG_OpenNeuro")
file = data_folder / "sub-002_ses-t1_task-resteyesc_eeg.edf"

raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
mne.set_eeg_reference(raw, copy=False, verbose=False)
#Be carefull that artifact electrodes are not included in the average calc.
raw.set_montage('biosemi64')
raw.drop_channels(['P9', 'T7', 'TP7', 'F7', 'F5', 'F7', 'AF3', 'F2', 'F3', 'F8'])#raw.info["bads"].append("O2")
mne.set_eeg_reference(raw, copy=False, verbose=False)



raw_highpass = raw.copy().filter(l_freq=1, h_freq=None, verbose=False)
raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=100, verbose=False)
line_noise = zapline_clean(raw_lowpass, 50)
raw_down_sampled = line_noise.copy().resample(sfreq=200, verbose=False)
#I should maybe downsample, and maybe do it first for computational efficiency. The only thing is that then almost all the data will be in simpochs, so maybe I need stricter limits for gfp peakes.
#Bad segments can maybe be identified with Christoffers algorithm, anything else? I want to do this at the end of the pipeline I think.

raw_down_sampled.plot(block=False)


data_folder = pathlib.Path(r"C:\Users\Gulbr\MasterOppgave\SRM_rs_EEG_OpenNeuro")
file = data_folder / "sub-002_ses-t2_task-resteyesc_eeg.edf"


raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
mne.set_eeg_reference(raw, copy=False, verbose=False)
#Be carefull that artifact electrodes are not included in the average calc.
raw.set_montage('biosemi64')


raw_highpass = raw.copy().filter(l_freq=1, h_freq=None, verbose=False)
raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=100, verbose=False)
line_noise = zapline_clean(raw_lowpass, 50)
raw_down_sampled = line_noise.copy().resample(sfreq=200, verbose=False)
#I should maybe downsample, and maybe do it first for computational efficiency. The only thing is that then almost all the data will be in simpochs, so maybe I need stricter limits for gfp peakes.
#Bad segments can maybe be identified with Christoffers algorithm, anything else? I want to do this at the end of the pipeline I think.

raw_down_sampled.plot(block=False)

data_folder = pathlib.Path(r"C:\Users\Gulbr\MasterOppgave\SRM_rs_EEG_OpenNeuro")
file = data_folder / "sub-002_ses-t2_task-resteyesc_eeg.edf"


raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
mne.set_eeg_reference(raw, copy=False, verbose=False)
#Be carefull that artifact electrodes are not included in the average calc.
raw.set_montage('biosemi64')
raw.drop_channels(['T8', 'T7', 'TP7', 'FT7', 'F7', 'AF7', 'C6', 'F4', 'Fp2', 'POz'])#raw.info["bads"].append("O2")


raw_highpass = raw.copy().filter(l_freq=1, h_freq=None, verbose=False)
raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=100, verbose=False)
line_noise = zapline_clean(raw_lowpass, 50)
raw_down_sampled = line_noise.copy().resample(sfreq=200, verbose=False)
#I should maybe downsample, and maybe do it first for computational efficiency. The only thing is that then almost all the data will be in simpochs, so maybe I need stricter limits for gfp peakes.
#Bad segments can maybe be identified with Christoffers algorithm, anything else? I want to do this at the end of the pipeline I think.
raw_down_sampled.set_eeg_reference(ref_channels='average')
raw_down_sampled.plot(block=True)

