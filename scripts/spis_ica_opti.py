#!/usr/bin/python
import pathlib
import sklearn
import mne
import random
import pymatreader
from multiprocessing import Process
import numpy as np
from meegkit import dss
from eeg_clean import clean
from data_quality import ica_score

# Parameters
data_set = pathlib.Path(r"C:\Users\workbench\eirik_master\Data\SPIS-Resting-State-Dataset\Pre-SART EEG")

subjects = []
for pth in data_set.iterdir():
    subjects.append(pth)

channel_quasi = [True, False]
channel_peaks = [True, False]
channel_corr = [True, False]
channel_pca = [None, sklearn.preprocessing.MinMaxScaler()]

epoch_quasi = [None, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8]
epoch_peaks = [None, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9]



def zapline_clean(raw, fline):
    data = raw.get_data().T  # Convert mne data to numpy darray
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
    process_results = np.empty(
        (
            len(channel_quasi),
            len(channel_peaks),
            len(channel_corr),
            len(channel_pca),
            len(epoch_quasi),
            len(epoch_peaks),
            5,
        )
    )
    print(subjects[my_index])
    dict = pymatreader.read_mat(subjects[my_index])

    info = mne.create_info(sfreq=256, ch_types="eeg", ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'])

    raw= mne.io.RawArray(dict['dataRest'][:64, :], info)

    raw.set_montage("biosemi64", verbose=False)
    raw.filter(l_freq=1, h_freq=None, verbose=False)
    raw.filter(l_freq=None, h_freq=100, verbose=False)
    raw = zapline_clean(raw, 50)
    raw.resample(sfreq=201, verbose=False)
    max_ep = raw.get_data().shape[1]/201

    start = 15
    stop = 145
    random_start = round(random.uniform(start, stop), 2)
    eeg = raw.copy().crop(random_start, random_start + 60).load_data()
    epochs = mne.make_fixed_length_epochs(eeg, 0.5, verbose=False, preload=True)

    for cq, chqas in enumerate(channel_quasi):
        for cp, chpea in enumerate(channel_peaks):
            for cr, corr in enumerate(channel_corr):
                for cd, chpca in enumerate(channel_pca):
                    processor = clean.Clean(
                        epochs,
                        sklearn_scaler=sklearn.preprocessing.MinMaxScaler(),
                        quasi=chqas,
                        peaks=chpea,
                        corr=corr,
                        av_ref=chpca,
                    )
                    for eq, epqas in enumerate(epoch_quasi):
                        for ep, eppea in enumerate(epoch_peaks):
                            if epqas is None:
                                if eppea is None:
                                    processor.find_bad_epochs()
                                else:
                                    processor.find_bad_epochs(
                                        peaks_args={
                                            "method": "function_threshold",
                                            "function": np.poly1d([-eppea, 1]),
                                            "exclude": "bigger",
                                        }
                                    )
                            else:
                                if eppea is None:
                                    processor.find_bad_epochs(
                                        quasi_args={
                                            "method": "function_threshold",
                                            "function": np.poly1d([epqas, 1.2]),
                                            "exclude": "bigger",
                                        }
                                    )
                                else:
                                    processor.find_bad_epochs(
                                        peaks_args={
                                            "method": "function_threshold",
                                            "function": np.poly1d([-eppea, 1]),
                                            "exclude": "bigger",
                                        },
                                        quasi_args={
                                            "method": "function_threshold",
                                            "function": np.poly1d([epqas, 1.2]),
                                            "exclude": "bigger",
                                        },
                                    )

                            if (
                                processor.bad_channel_index is not None
                                and processor.bad_epoch_index is not None
                            ):
                                process_results[cq, cp, cr, cd, eq, ep, 0] = (
                                    processor.bad_channel_index.size / 128
                                    + processor.bad_epoch_index.size / 120
                                )
                            elif processor.bad_channel_index is not None:
                                process_results[cq, cp, cr, cd, eq, ep, 0] = (
                                    processor.bad_channel_index.size / 128
                                )
                            elif processor.bad_epoch_index is not None:
                                process_results[cq, cp, cr, cd, eq, ep, 0] = (
                                    processor.bad_epoch_index.size / 120
                                )
                            else:
                                process_results[cq, cp, cr, cd, eq, ep, 0] = 0

                            if process_results[cq, cp, cr, cd, eq, ep, 0] < 1:
                                for_ica = epochs.copy()
                                for_ica.set_eeg_reference()

                                if processor.bad_epoch_index is not None:
                                    for_ica.drop(processor.bad_epoch_index)

                                if processor.bad_channel_index is not None:
                                    for_ica.drop_channels(
                                        processor.ch_names[
                                            processor.bad_channel_index
                                        ]
                                    )

                                evaluate = ica_score.IcaScore(for_ica)
                                process_results[cq, cp, cr, cd, eq, ep, 1] = (
                                    evaluate.get_n_components()[0]
                                )
                                process_results[cq, cp, cr, cd, eq, ep, 2] = (
                                    evaluate.get_n_components()[0]
                                    + evaluate.get_n_components()[1]
                                )
                                process_results[cq, cp, cr, cd, eq, ep, 3] = (
                                    evaluate.get_explained_var()["eeg"]
                                )
                                process_results[cq, cp, cr, cd, eq, ep, 4] = (
                                    evaluate.get_explained_var(bio_components=True)[
                                        "eeg"
                                    ]
                                )
                            else:
                                process_results[cq, cp, cr, cd, eq, ep, 1:] = (
                                    -np.inf
                                )

    save_path = data_set / f"epi_grid_{my_index}"
    np.save(save_path, process_results)


# Run experiments
if __name__ == "__main__":
    p0 = Process(target=process, args=(0,))
    p1 = Process(target=process, args=(1,))
    p2 = Process(target=process, args=(2,))
    p3 = Process(target=process, args=(3,))
    p4 = Process(target=process, args=(4,))
    p5 = Process(target=process, args=(5,))
    p6 = Process(target=process, args=(6,))
    p7 = Process(target=process, args=(7,))
    p8 = Process(target=process, args=(8,))
    p9 = Process(target=process, args=(9,))
    p10 = Process(target=process, args=(10,))
    p11 = Process(target=process, args=(11,))
    p12 = Process(target=process, args=(12,))
    p13 = Process(target=process, args=(13,))
    p14 = Process(target=process, args=(14,))
    p15 = Process(target=process, args=(15,))

    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()
    p11.start()
    p12.start()
    p13.start()
    p14.start()
    p15.start()

    p0.join()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()
    p11.join()
    p12.join()
    p13.join()
    p14.join()
    p15.join()
