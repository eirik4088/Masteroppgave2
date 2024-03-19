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

epochs_path = pathlib.Path(
    r"C:\Users\workbench\eirik_master\Results\epi_data\accumulated_clean"
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
    process_results = np.zeros(
        (
            2,
            8,
            8,
            5,
        )
    )
    raw = mne.io.read_raw_bdf(subjects[my_index], verbose=False)

    raw.drop_channels(
        ["SO2", "IO2", "LO1", "LO2", "EXG5", "EXG6", "EXG7", "EXG8", "Status"]
    )
    raw.set_montage("biosemi128", verbose=False)

    for c in range(0, 2):
        new = raw.copy().crop(time_starts[my_index][c], time_starts[my_index][c] + 240)

        cond = (
            new.copy()
            .crop(random_start[c][my_index], random_start[c][my_index] + 60)
            .load_data(verbose=False)
        )
        cond.filter(l_freq=1, h_freq=None, verbose=False)
        cond.filter(l_freq=None, h_freq=100, verbose=False)
        cond = zapline_clean(cond, 50)
        cond.resample(sfreq=201, verbose=False)
        epochs = mne.make_fixed_length_epochs(cond, 0.5, verbose=False, preload=True)

        for cq in range(8):
            for cp in range(8):
                processor = clean.Clean(
                    epochs.copy(),
                    sklearn_scaler=sklearn.preprocessing.MinMaxScaler(),
                    find_random=False,
                    top_n=(7 - cq, 7 - cp),
                )

                if processor.bad_channel_index is not None:
                    replicate = epochs.copy().drop_channels(
                        processor.ch_names[processor.bad_channel_index]
                    )
                else:
                    replicate = epochs.copy()

                if processor.bad_channel_index is not None:
                    process_results[c, cq, cp, 0] = (
                        processor.bad_channel_index.size / 128
                    ) * 2
                else:
                    process_results[c, cq, cp, 0] = 0

                if process_results[c, cq, cp, 0] < 2:
                    for_ica = replicate.copy()
                    for_ica.set_eeg_reference(verbose=False)

                    evaluate = ica_score.IcaScore(for_ica)
                    process_results[c, cq, cp, 1] = evaluate.get_n_components()[0]
                    process_results[c, cq, cp, 2] = (
                        evaluate.get_n_components()[0] + evaluate.get_n_components()[1]
                    )
                    process_results[c, cq, cp, 3] = evaluate.get_explained_var()["eeg"]
                    process_results[c, cq, cp, 4] = evaluate.get_explained_var(
                        bio_components=True
                    )["eeg"]
                else:
                    process_results[c, cq, cp, 1:] = 0

                if (
                    process_results[c, cq, cp, 1] >= np.max(process_results[c, :, :, 1])
                    or process_results[c, cq, cp, 2]
                    >= np.max(process_results[c, :, :, 2])
                    or process_results[c, cq, cp, 3]
                    >= np.max(process_results[c, :, :, 3])
                    or process_results[c, cq, cp, 4]
                    >= np.max(process_results[c, :, :, 4])
                ):
                    h = epoch_stats.EpochStats(for_ica)
                    h.calc_stability()
                    if process_results[c, cq, cp, 1] >= np.max(
                        process_results[c, :, :, 1]
                    ):
                        np.save(
                            epochs_path / str(1) / "quasi" / "dis" / f"{my_index}_{c}",
                            h.quasi_stability.get_mean_stab(),
                        )
                        np.save(
                            epochs_path
                            / str(1)
                            / "quasi"
                            / "abs_dis"
                            / f"{my_index}_{c}",
                            h.quasi_stability.get_mean_abs_stab(),
                        )
                        np.save(
                            epochs_path / str(1) / "peaks" / "dis" / f"{my_index}_{c}",
                            h.peak_stability.get_mean_stab(),
                        )
                        np.save(
                            epochs_path
                            / str(1)
                            / "peaks"
                            / "abs_dis"
                            / f"{my_index}_{c}",
                            h.peak_stability.get_mean_abs_stab(),
                        )

                    if process_results[c, cq, cp, 2] >= np.max(
                        process_results[c, :, :, 2]
                    ):
                        np.save(
                            epochs_path / str(2) / "quasi" / "dis" / f"{my_index}_{c}",
                            h.quasi_stability.get_mean_stab(),
                        )
                        np.save(
                            epochs_path
                            / str(2)
                            / "quasi"
                            / "abs_dis"
                            / f"{my_index}_{c}",
                            h.quasi_stability.get_mean_abs_stab(),
                        )
                        np.save(
                            epochs_path / str(2) / "peaks" / "dis" / f"{my_index}_{c}",
                            h.peak_stability.get_mean_stab(),
                        )
                        np.save(
                            epochs_path
                            / str(2)
                            / "peaks"
                            / "abs_dis"
                            / f"{my_index}_{c}",
                            h.peak_stability.get_mean_abs_stab(),
                        )

                    if process_results[c, cq, cp, 3] >= np.max(
                        process_results[c, :, :, 3]
                    ):
                        np.save(
                            epochs_path / str(3) / "quasi" / "dis" / f"{my_index}_{c}",
                            h.quasi_stability.get_mean_stab(),
                        )
                        np.save(
                            epochs_path
                            / str(3)
                            / "quasi"
                            / "abs_dis"
                            / f"{my_index}_{c}",
                            h.quasi_stability.get_mean_abs_stab(),
                        )
                        np.save(
                            epochs_path / str(3) / "peaks" / "dis" / f"{my_index}_{c}",
                            h.peak_stability.get_mean_stab(),
                        )
                        np.save(
                            epochs_path
                            / str(3)
                            / "peaks"
                            / "abs_dis"
                            / f"{my_index}_{c}",
                            h.peak_stability.get_mean_abs_stab(),
                        )
                    if process_results[c, cq, cp, 4] >= np.max(
                        process_results[c, cq, cp, 4]
                    ):
                        np.save(
                            epochs_path / str(4) / "quasi" / "dis" / f"{my_index}_{c}",
                            h.quasi_stability.get_mean_stab(),
                        )
                        np.save(
                            epochs_path
                            / str(4)
                            / "quasi"
                            / "abs_dis"
                            / f"{my_index}_{c}",
                            h.quasi_stability.get_mean_abs_stab(),
                        )
                        np.save(
                            epochs_path / str(4) / "peaks" / "dis" / f"{my_index}_{c}",
                            h.peak_stability.get_mean_stab(),
                        )
                        np.save(
                            epochs_path
                            / str(4)
                            / "peaks"
                            / "abs_dis"
                            / f"{my_index}_{c}",
                            h.peak_stability.get_mean_abs_stab(),
                        )
    np.save(epochs_path / "all" / f"{my_index}_{c}", process_results)


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
