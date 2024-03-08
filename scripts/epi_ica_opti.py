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
data_set = pathlib.Path(r"C:\Users\Gulbr\MasterOppgave\Data\epilepsy_data")

subjects = []
time_starts = [(0.4, 4.92), (0.49, 4.84)]
for pth in data_set.iterdir():
    subjects.append(pth)

channel_quasi = [True, False]
channel_peaks = [True, False]
channel_corr = [True, False]
channel_pca = [None, sklearn.preprocessing.MinMaxScaler()]

epoch_quasi = [None, 0.4]  # , 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8]
epoch_peaks = [None, 0.7]  # , 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9]


grid_results = np.empty(
    (
        len(subjects),
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

start_times = np.empty((len(subjects), 2))

Q = Queue()


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
    raw.drop_channels(
        ["SO2", "IO2", "LO1", "LO2", "EXG5", "EXG6", "EXG7", "EXG8", "Status"]
    )
    raw.set_montage("biosemi128", verbose=False)
    raw.crop(0, 360).load_data()
    raw.filter(l_freq=1, h_freq=None, verbose=False)
    raw.filter(l_freq=None, h_freq=100, verbose=False)
    raw = zapline_clean(raw, 50)
    raw.resample(sfreq=201, verbose=False)

    ec_start = time_starts[my_index][0] * 1.1
    ec_stop = ec_start + (4 * 0.9) - 1
    ec_random_start = round(random.uniform(ec_start, ec_stop), 2)
    start_times[my_index, 0] = ec_random_start
    ec = raw.copy().crop(ec_start, ec_start + 1).load_data()

    eo_start = time_starts[my_index][1] * 1.1
    eo_stop = eo_start + (3 * 0.9) - 1
    eo_random_start = round(random.uniform(eo_start, eo_stop), 2)
    start_times[my_index, 1] = eo_random_start
    eo = raw.copy().crop(ec_start, ec_start + 1).load_data()

    conditions = [ec, eo]

    for c, cond in enumerate(conditions):
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

    Q.put((process_results, my_index))


# Run experiments
if __name__ == "__main__":
    for d1 in range(grid_results.shape[0]):
        p0 = Process(target=process, args=(0,))
        p1 = Process(target=process, args=(1,))

        p0.start()
        p1.start()

        res = Q.get()
        grid_results[res[1], :, :, :, :, :, :, :, :] = res[0]
        res = Q.get()
        grid_results[res[1], :, :, :, :, :, :, :, :] = res[0]

        p0.join()
        p1.join()

        save_path = data_set / "epi_grid"
        save_times = data_set / "times"
        np.save(save_path, grid_results)
        np.save(save_times, start_times)
