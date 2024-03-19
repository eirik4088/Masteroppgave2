import pathlib
import sklearn
import mne
from multiprocessing import Process
import numpy as np
from meegkit import dss
from eeg_clean import clean, epoch_stats
from data_quality import ica_score

# Parameters
data_set = pathlib.Path(r"C:\Users\workbench\eirik_master\Data\epi_data")

quasi_dis_path = pathlib.Path(
    r"C:\Users\workbench\eirik_master\Results\epi_data\accumulate\quasi\dis"
)
quasi_abs_dis_path = pathlib.Path(
    r"C:\Users\workbench\eirik_master\Results\epi_data\accumulate\quasi\abs_dis"
)
peaks_dis_path = pathlib.Path(
    r"C:\Users\workbench\eirik_master\Results\epi_data\accumulate\peaks\dis"
)
peaks_abs_dis_path = pathlib.Path(
    r"C:\Users\workbench\eirik_master\Results\epi_data\accumulate\peaks\abs_dis"
)

subjects = []
time_starts = [
    (0.76 * 60, 5.58 * 60),
    (0.4 * 60, 4.92 * 60),
    (0.49 * 60, 4.84 * 60),
    (0.62 * 60, 4.98 * 60),
    (0.46 * 60, 4.7 * 60),
    (0.78 * 60, 5.15 * 60),
    (0.72 * 60, 5.03 * 60),
    (0.71 * 60, 5 * 60),
    (0.57 * 60, 4.91 * 60),
    (0.57 * 60, 4.82 * 60),
    (1.35 * 60, 5.8 * 60),
    (0.58 * 60, 4.97 * 60),
    (0.55 * 60, 4.91 * 60),
    (0.54 * 60, 4.83 * 60),
    (0.68 * 60, 5.02 * 60),
    (0.84 * 60, 5.2 * 60),
    (0.49 * 60, 4.77 * 60),
    (0.58 * 60, 4.88 * 60),
    (0.59 * 60, 5.43 * 60),
]
for pth in data_set.iterdir():
    subjects.append(pth)

random_start = [
    [56, 64, 60, 71, 74, 112, 42, 68, 131, 52, 144, 152, 147, 138, 99, 48],
    [59, 110, 44, 55, 37, 76, 64, 67, 74, 74, 146, 103, 25, 76, 58, 127],
]

# zapline = [False, True] ?
channel_quasi = [False, True]
channel_peaks = [False, True]
channel_corr = [False, True]
average = [False, True]

epoch_quasi = [None, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8]
epoch_peaks = [None, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9]


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
    process_results = np.empty(
        (
            2,
            len(channel_quasi),
            len(channel_peaks),
            len(channel_corr),
            len(average),
            len(epoch_quasi),
            len(epoch_peaks),
            5,
        )
    )
    raw = mne.io.read_raw_bdf(subjects[my_index], verbose=False)

    raw.drop_channels(
        ["SO2", "IO2", "LO1", "LO2", "EXG5", "EXG6", "EXG7", "EXG8", "Status"]
    )
    raw.set_montage("biosemi128", verbose=False)

    for c in range(2):
        new = raw.copy().crop(time_starts[my_index][c], time_starts[my_index][c] + 240)

        cond = new.copy().crop(random_start[c][my_index], random_start[c][my_index] + 60).load_data(verbose=False)
        cond.filter(l_freq=1, h_freq=None, verbose=False)
        cond.filter(l_freq=None, h_freq=100, verbose=False)
        cond = zapline_clean(cond, 50)
        cond.resample(sfreq=201, verbose=False)
        epochs = mne.make_fixed_length_epochs(cond, 0.5, verbose=False, preload=True)

        h = epoch_stats.EpochStats(epochs)
        h.calc_stability()
        np.save(quasi_dis_path / f"{my_index}_{c}", h.quasi_stability.get_mean_stab())
        np.save(
            quasi_abs_dis_path / f"{my_index}_{c}",
            h.quasi_stability.get_mean_abs_stab(),
        )
        np.save(peaks_dis_path / f"{my_index}_{c}", h.peak_stability.get_mean_stab())
        np.save(
            peaks_abs_dis_path / f"{my_index}_{c}", h.peak_stability.get_mean_abs_stab()
        )

        for cq, chqas in enumerate(channel_quasi):
            for cp, chpea in enumerate(channel_peaks):
                for cr, corr in enumerate(channel_corr):
                    for a, avr in enumerate(average):
                        processor = clean.Clean(
                            epochs.copy(),
                            sklearn_scaler=sklearn.preprocessing.MinMaxScaler(),
                            quasi=chqas,
                            peaks=chpea,
                            corr=corr,
                            av_ref=avr,
                        )

                        if processor.bad_channel_index is not None:
                            replicate = epochs.copy().drop_channels(
                                processor.ch_names[processor.bad_channel_index]
                            )
                        else:
                            replicate = epochs.copy()

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
                                    process_results[c, cq, cp, cr, a, eq, ep, 0] = (
                                        processor.bad_channel_index.size / 128
                                        + processor.bad_epoch_index.size / 120
                                    )
                                elif processor.bad_channel_index is not None:
                                    process_results[c, cq, cp, cr, a, eq, ep, 0] = (
                                        processor.bad_channel_index.size / 128
                                    ) * 2
                                elif processor.bad_epoch_index is not None:
                                    process_results[c, cq, cp, cr, a, eq, ep, 0] = (
                                        processor.bad_epoch_index.size / 120
                                    ) * 2
                                else:
                                    process_results[c, cq, cp, cr, a, eq, ep, 0] = 0

                                if process_results[c, cq, cp, cr, a, eq, ep, 0] < 2:
                                    for_ica = replicate.copy()
                                    for_ica.set_eeg_reference(verbose=False)

                                    if processor.bad_epoch_index is not None:
                                        for_ica.drop(
                                            processor.bad_epoch_index, verbose=False
                                        )

                                    evaluate = ica_score.IcaScore(for_ica)
                                    process_results[c, cq, cp, cr, a, eq, ep, 1] = (
                                        evaluate.get_n_components()[0]
                                    )
                                    process_results[c, cq, cp, cr, a, eq, ep, 2] = (
                                        evaluate.get_n_components()[0]
                                        + evaluate.get_n_components()[1]
                                    )
                                    process_results[c, cq, cp, cr, a, eq, ep, 3] = (
                                        evaluate.get_explained_var()["eeg"]
                                    )
                                    process_results[c, cq, cp, cr, a, eq, ep, 4] = (
                                        evaluate.get_explained_var(bio_components=True)[
                                            "eeg"
                                        ]
                                    )
                                else:
                                    process_results[c, cq, cp, cr, a, eq, ep, 1:] = 0

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
