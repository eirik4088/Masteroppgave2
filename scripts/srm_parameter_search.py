import pathlib
import sklearn
import mne
from multiprocessing import Process
import numpy as np
from meegkit import dss
from eeg_clean import clean_new, epoch_stats
from data_quality import ica_score

data_set = pathlib.Path(r"C:\Users\workbench\eirik_master\Data\srm_data")

subjects = []
for pth in data_set.iterdir():
    subjects.append(pth)

random_start = [148, 49, 49, 87, 68, 87, 38, 72, 148, 34, 25, 72, 150, 116, 52, 44]

quasi = [False, True]
peak = [False, True]
n_peaks = [False, True]
central_meassure = ["mean", "median"]
spread_correction = [None, "IQR"]
stds = [4, 5]


def zapline_clean(raw, fline):
    data = raw.get_data(verbose=False).T  # Convert mne data to numpy darray
    sfreq = raw.info["sfreq"]  # Extract the sampling freq
    # Apply MEEGkit toolbox function
    out, _ = dss.dss_line(
        data, fline, sfreq, nkeep=1, show=False
    )  # fline (Line noise freq) = 50 Hz for Europe
    cleaned_raw = mne.io.RawArray(
        out.T, raw.info, verbose=False
    )  # Convert output to mne RawArray again

    return cleaned_raw


def process(my_index):
    base_line = np.zeros(5)
    process_results = np.empty(
        (
            len(quasi),
            len(peak),
            len(n_peaks),
            len(central_meassure),
            len(spread_correction),
            len(stds),
            len(stds),
            5,
        )
    )

    raw = mne.io.read_raw_edf(subjects[my_index], verbose=False)
    raw.set_montage("biosemi64", verbose=False)

    eeg = (
        raw.copy()
        .crop(random_start[my_index], random_start[my_index] + 60)
        .load_data(verbose=False)
    )
    eeg.filter(l_freq=1, h_freq=None, verbose=False)
    eeg.filter(l_freq=None, h_freq=100, verbose=False)
    eeg = zapline_clean(eeg, 50)
    eeg.resample(sfreq=201, verbose=False)
    epochs = mne.make_fixed_length_epochs(eeg, 0.5, verbose=False, preload=True)

    for q, qb in enumerate(quasi):
        for p, pb in enumerate(peak):
            for np, npb in enumerate(n_peaks):

                if q == 0 and p == 0 and np == 0:
                    continue

                for c, cm in enumerate(central_meassure):
                    for s, sc in enumerate(spread_correction):
                        for sd, std in enumerate(stds):
                            for sd2, std2 in enumerate(stds):

                                if qb and pb and npb:
                                    clean_new.CleanNew(
                                        epochs,
                                        thresholds=[std, std2],
                                        distspecifics={
                                            "quasi": {
                                                "central": cm,
                                                "spred_corrected": sc,
                                            },
                                            "peak": {
                                                "central": "median",
                                                "spred_corrected": "IQR",
                                            },
                                            "n_peaks": {
                                                "central": "mean",
                                                "spred_corrected": None,
                                            },
                                        }
                                    )
