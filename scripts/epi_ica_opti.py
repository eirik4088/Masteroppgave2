#!/usr/bin/python
import pathlib
import sklearn
import mne
import random
from multiprocessing import Process, Queue
import numpy as np
from meegkit import dss
from eeg_clean import clean
from data_quality import ica_score

# Parameters
data_set = pathlib.Path(r"C:\Users\workbench\eirik_master\Data\epi_data")

subjects = []
time_starts = [(0.76, 5.58), (0.4, 4.92), (0.49, 4.84), (0.62, 4.98), (0.46, 4.7), (0.78, 5.15), (0.72, 5.03), (0.71, 5), (0.57, 4.91), (0.57, 4.82), (1.35, 5.8), (0.58, 4.97), (0.55, 4.91), (0.54, 4.83), (0.68, 5.02), (0.84, 5.2), (0.49, 4.77), (0.58, 4.88), (0.59, 5.43)]
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
            2,
            len(channel_quasi),
            len(channel_peaks),
            len(channel_corr),
            len(channel_pca),
            len(epoch_quasi),
            len(epoch_peaks),
            5,
        )
    )
    raw = mne.io.read_raw_bdf(subjects[my_index], verbose=False)
    print(subjects[my_index])
    raw.drop_channels(
        ["SO2", "IO2", "LO1", "LO2", "EXG5", "EXG6", "EXG7", "EXG8", "Status"]
    )
    raw.set_montage("biosemi128", verbose=False)

    for c in range(2):
        new = raw.copy().crop(time_starts[my_index][c], time_starts[my_index][c]+180).load_data()
        new.filter(l_freq=1, h_freq=None, verbose=False)
        new.filter(l_freq=None, h_freq=100, verbose=False)
        new = zapline_clean(new, 50)
        new.resample(sfreq=201, verbose=False)

        start = 18
        stop = 162
        ec_random_start = round(random.uniform(start, stop), 2)
        cond = new.copy().crop(ec_random_start, ec_random_start + 60).load_data()
        epochs = mne.make_fixed_length_epochs(cond, 0.5, verbose=False, preload=True)

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
                                    process_results[c, cq, cp, cr, cd, eq, ep, 0] = (
                                        processor.bad_channel_index.size / 128
                                        + processor.bad_epoch_index.size / 120
                                    )
                                elif processor.bad_channel_index is not None:
                                    process_results[c, cq, cp, cr, cd, eq, ep, 0] = (
                                        processor.bad_channel_index.size / 128
                                    )
                                elif processor.bad_epoch_index is not None:
                                    process_results[c, cq, cp, cr, cd, eq, ep, 0] = (
                                        processor.bad_epoch_index.size / 120
                                    )
                                else:
                                    process_results[c, cq, cp, cr, cd, eq, ep, 0] = 0

                                if process_results[c, cq, cp, cr, cd, eq, ep, 0] < 1:
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
                                    process_results[c, cq, cp, cr, cd, eq, ep, 1] = (
                                        evaluate.get_n_components()[0]
                                    )
                                    process_results[c, cq, cp, cr, cd, eq, ep, 2] = (
                                        evaluate.get_n_components()[0]
                                        + evaluate.get_n_components()[1]
                                    )
                                    process_results[c, cq, cp, cr, cd, eq, ep, 3] = (
                                        evaluate.get_explained_var()["eeg"]
                                    )
                                    process_results[c, cq, cp, cr, cd, eq, ep, 4] = (
                                        evaluate.get_explained_var(bio_components=True)[
                                            "eeg"
                                        ]
                                    )
                                else:
                                    process_results[c, cq, cp, cr, cd, eq, ep, 1:] = (
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

