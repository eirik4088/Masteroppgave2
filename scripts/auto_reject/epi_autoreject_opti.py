import pathlib
import mne
from multiprocessing import Process
import numpy as np
from meegkit import dss
from eeg_clean import clean_new
from data_quality import ica_score
from autoreject import AutoReject

data_set = pathlib.Path(r"C:\Users\workbench\eirik_master\Data\epi_data")

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

thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
av_ref = [False, True]

#taken from https://mne.discourse.group/t/clean-line-noise-zapline-method-function-for-mne-using-meegkit-toolbox/7407
def zapline_clean(raw, fline):
    data = raw.get_data(verbose=False).T
    sfreq = raw.info["sfreq"]
    out, _ = dss.dss_line(
        data, fline, sfreq, nkeep=1, show=False
    )
    cleaned_raw = mne.io.RawArray(
        out.T, raw.info, verbose=False
    )

    return cleaned_raw


def evaluate(epochs, bad_channels, to_fill: np.ndarray, baseline=None):

    if len(bad_channels) > 0:
        to_fill[0] = len(bad_channels) / 128
    else:

        if baseline is not None:
            return

        to_fill[0] = 0

    if to_fill[0] < 1:
        epochs.set_eeg_reference(verbose=False)

        evaluater = ica_score.IcaScore(epochs)
        to_fill[1] = evaluater.get_n_components()[0]
        to_fill[2] = evaluater.get_n_components()[0] + evaluater.get_n_components()[1]
        to_fill[3] = evaluater.get_explained_var()["eeg"]
        to_fill[4] = evaluater.get_explained_var(bio_components=True)["eeg"]
        if baseline is not None:
            to_fill[1] = to_fill[1] - baseline[1]
            to_fill[2] = to_fill[2] - baseline[2]
            to_fill[3] = to_fill[3] - baseline[3]
            to_fill[4] = to_fill[4] - baseline[4]
    else:
        to_fill[1:] = float("nan")


def process(my_index):
    base_line = np.zeros(5)
    results = np.zeros(
        (
            len(thresholds),
            len(av_ref),
            5,
        )
    )

    raw = mne.io.read_raw_bdf(subjects[my_index], verbose=False)

    raw.drop_channels(
        ["SO2", "IO2", "LO1", "LO2", "EXG5", "EXG6", "EXG7", "EXG8", "Status"]
    )
    raw.set_montage("biosemi128", verbose=False)

    for eye in range(2):

        new = raw.copy().crop(
            time_starts[my_index][eye], time_starts[my_index][eye] + 240
        )
        cond = (
            new.copy()
            .crop(random_start[eye][my_index], random_start[eye][my_index] + 60)
            .load_data(verbose=False)
        )
        cond.filter(l_freq=1, h_freq=None, verbose=False)
        cond.filter(l_freq=None, h_freq=100, verbose=False)
        cond = zapline_clean(cond, 50)
        cond.resample(sfreq=201, verbose=False)

        epochs = mne.make_fixed_length_epochs(cond, 0.5, verbose=False, preload=True)
        evaluate(epochs.copy(), [], base_line)

        for t, thrs in enumerate(thresholds):
            for r, af in enumerate(av_ref):
                epochs_copy = epochs.copy()
                if af:
                    epochs_copy.set_eeg_reference(verbose=False)

                reject = AutoReject(
                    consensus=[1.0], n_interpolate=[0], random_state=97, verbose=False
                )
                reject.fit(epochs_copy)
                log = reject.get_reject_log(epochs_copy)
                n_epochs = len(epochs)
                n_bads = log.labels.sum(axis=0)
                bads_index = np.where(n_bads > n_epochs * thrs)[0]

                if bads_index.size > 0:
                    bads_name = [epochs_copy.ch_names[idx] for idx in bads_index]
                    epochs_copy.drop_channels(bads_name)
                else:
                    bads_name = []

                evaluate(epochs_copy, bads_name, results[t, r, :], base_line)

        save_folder = pathlib.Path(
            r"C:\Users\workbench\eirik_master\Results\epi_data\auto_reject"
        )
        np.save(save_folder / str(eye) / f"{my_index}", results)


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
