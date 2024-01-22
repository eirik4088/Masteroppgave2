import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import mne
import pathlib
import sklearn
import hdbscan
import seaborn as sns

from pycrostates.preprocessing import extract_gfp_peaks
from package1.tbd import ModKMeansSkstab, scale_data

data_folder = pathlib.Path(r"C:\Users\Gulbr\MasterOppgave\SRM_rs_EEG_OpenNeuro")
file = data_folder / "sub-002_ses-t1_task-resteyesc_eeg.edf"
#processed_folder = pathlib.Path(r"C:\Users\Gulbr\MasterOppgave\SRM_rs_EEG_OpenNeuro_cleaned")
#processed_file = processed_folder / "sub-001_ses-t1_task-resteyesc_desc-epochs_eeg.set"

raw = mne.io.read_raw_edf(file, preload=True, verbose=False)

raw_highpass = raw.copy().filter(l_freq=0.5, h_freq=None, verbose=False)
raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=40, verbose=False)
raw_down_sampled = raw_lowpass.copy().resample(sfreq=256)

gfp_peakes = extract_gfp_peaks(raw_down_sampled, min_peak_distance=1)
#print(gfp_peakes.get_data().shape)
top_indices = np.argpartition(np.std(gfp_peakes.get_data(), axis=0), -gfp_peakes.get_data().shape[1])[-gfp_peakes.get_data().shape[1]:]
#print(top_indices.shape)
gfp_data = gfp_peakes.get_data()[:, top_indices]
#print(gfp_data.shape)

norms = scale_data(gfp_data.T, z_score=False)
activation = norms.dot(norms.T)
absolute_cosine_matrix = np.abs(activation)
#np.fill_diagonal(absolute_cosine_matrix, 0)
print(absolute_cosine_matrix.shape)
distance_matrix = np.abs(absolute_cosine_matrix - 1)

cluster = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=5, min_samples=2)
cluster.fit(distance_matrix)

base = len(cluster.labels_[np.where(cluster.labels_==-1)])/len(cluster.labels_)

outlier_percentage = np.ndarray(len(raw_down_sampled.info['ch_names']))
i=0

for c in raw_down_sampled.info['ch_names']:
    raw = mne.io.read_raw_edf(file, preload=True)

    raw_highpass = raw.copy().filter(l_freq=0.5, h_freq=None)
    raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=40)
    raw_down_sampled = raw_lowpass.copy().resample(sfreq=256)
    raw_down_sampled.drop_channels(c)

    mne.set_eeg_reference(raw_down_sampled, verbose=False)

    gfp_peakes = extract_gfp_peaks(raw_down_sampled, min_peak_distance=1)
    #print(gfp_peakes.get_data().shape)
    top_indices = np.argpartition(np.std(gfp_peakes.get_data(), axis=0), -gfp_peakes.get_data().shape[1])[-gfp_peakes.get_data().shape[1]:]
    #print(top_indices.shape)
    gfp_data = gfp_peakes.get_data()[:, top_indices]
    #print(gfp_data.shape)

    norms = scale_data(gfp_data.T, z_score=False)
    activation = norms.dot(norms.T)
    absolute_cosine_matrix = np.abs(activation)
    #np.fill_diagonal(absolute_cosine_matrix, 0)
    print(absolute_cosine_matrix.shape)
    distance_matrix = np.abs(absolute_cosine_matrix - 1)

    cluster = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=5, min_samples=2)
    cluster.fit(distance_matrix)

    outlier_percentage[i] = len(cluster.labels_[np.where(cluster.labels_==-1)])/len(cluster.labels_)
    i+=1
print(base)
for i in range(len(raw_down_sampled.info['ch_names'])):
    print(f"{raw_down_sampled.info['ch_names'][i]}: {outlier_percentage[i]}")

plt.boxplot(outlier_percentage)
plt.show()