import pathlib
import mne
from multiprocessing import Process
import numpy as np
from meegkit import dss
from eeg_clean import clean_new
from data_quality import ica_score
from autoreject import AutoReject

data_set = pathlib.Path(r"C:\Users\workbench\eirik_master\Data\texas_data")

subjects = []
for pth in data_set.iterdir():
    subjects.append(pth)

times = [
    [
        [749, 15980],
        [98547, 113778],
        [81686, 96917],
        [49639, 64870],
        [711, 15942],
        [98263, 113494],
        [81711, 96942],
        [114402, 129505],
        [33278, 48509],
        [81700, 96931],
        [98727, 113503],
        [728, 15959],
        [97827, 113408],
        [65419, 80650],
        [16982, 32213],
        [49434, 64665],
    ],
    [
        [114938, 130169],
        [82092, 97323],
        [621, 15852],
        [33192, 48423],
        [16927, 32158],
        [81910, 97141],
        [774, 16005],
        [661, 15892],
        [16865, 32096],
        [33068, 48299],
        [114527, 129758],
        [16982, 32213],
        [81679, 96910],
        [16937, 32168],
        [33138, 48369],
        [97818, 113049],
    ],
]

thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
av_ref = [False, True]


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
        ['M1', 'M2', 'NAS', 'LVEOG', 'RVEOG', 'LHEOG', 'RHEOG', 'NFpz', 'Status']
    )
    raw.set_montage("biosemi64", verbose=False)

    for eye in range(2):

        cond = raw.copy().crop(times[eye][my_index][0]/256, times[eye][my_index][1]/256).load_data(verbose=False)

        cond.filter(l_freq=1, h_freq=None, verbose=False)
        cond.filter(l_freq=None, h_freq=100, verbose=False)
        cond = zapline_clean(cond, 60)
        cond.resample(sfreq=201, verbose=False)
        epochs = mne.make_fixed_length_epochs(cond, 0.5, verbose=False, preload=True)

        evaluate(epochs.copy(), [], base_line)

        for t, thrs in enumerate(thresholds):
            for r, af in enumerate(av_ref):
                epochs_copy = epochs.copy()
                if af:
                    epochs_copy.set_eeg_reference()

                # Create autoreject object and fit it with the data
                reject = AutoReject(consensus=[1.0], n_interpolate=[0])
                reject.fit(epochs_copy)
                # find where channels are considered bad, and extract the ones that are bad longer then threshold percentage
                log = reject.get_reject_log(epochs_copy())
                n_epochs = len(epochs)
                n_bads = log.labels.sum(axis=0)
                # Index of bad channels, drop them and evaluate...
                bads_index = np.where(n_bads > n_epochs * thrs)[0]

                if bads_index.size > 0:
                    bads_name = [epochs_copy.ch_names[idx] for idx in bads_index]
                    epochs_copy.drop_channels(bads_name)
                else:
                    bads_name=[]

                evaluate(epochs_copy, bads_name, results[t, r, :], base_line)

        save_folder = pathlib.Path(
            r"C:\Users\workbench\eirik_master\Results\texas_data\auto_reject"
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