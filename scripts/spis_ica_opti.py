import pathlib
import sklearn
import numpy as np
from eeg_clean import clean
from data_quality import ica_score

# Parameters
data_set = pathlib.Path(r"C:\Users\Gulbr\MasterOppgave\Data\SPIS")

subjects = []
for pth in data_set.iterdir():
    subjects.append(pth)
print(subjects)

channel_quasi = [True, False]
channel_peaks = [True, False]
channel_pca = [None, sklearn.preprocessing.MinMaxScaler()]

epoch_quasi = [None, 0.4]#, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8]
epoch_peaks = [None, 0.7]#, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9]


grid_results = np.empty(
    (
        len(subjects),
        len(channel_quasi),
        len(channel_peaks),
        len(channel_pca),
        len(epoch_quasi),
        len(epoch_peaks), 
        1,
        1,
        1,
        1,
        1
    )
)

#Run eksperiments
for d1 in grid_results.shape[0]:
    i = 1