import pathlib
import mne
from multiprocessing import Process
import numpy as np
from meegkit import dss
from eeg_clean import clean_new
from data_quality import ica_score
from pyprep import PrepPipeline

data_set = pathlib.Path(r"C:\Users\workbench\eirik_master\Data\yulin_wang")

subjects = []
to_include = np.arange(1, 97, 3)
for i, pth in enumerate(data_set.iterdir()):
    if i in to_include:
        subjects.append(pth)

random_start = [
    [117, 186, 199, 89, 206, 41, 124, 165, 106, 119, 173, 138, 187, 90, 170, 178],
    [66, 112, 52, 193, 31, 97, 146, 91, 52, 138, 188, 83, 129, 177, 103, 95],
]

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
        to_fill[0] = len(bad_channels) / 64
    else:

        if baseline is not None:
            return

        to_fill[0] = 0

    if to_fill[0] < 0.74:
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
    results = np.zeros(5)

    for eye in range(2):
        raw = mne.io.read_raw_brainvision(subjects[my_index + eye], verbose=False)

        if "Cpz" in raw.ch_names:
            raw.drop_channels("Cpz")
        raw.set_montage("standard_1020", verbose=False)

        cond = (
            raw.copy()
            .crop(
                random_start[eye][int(my_index / 2)],
                random_start[eye][int(my_index / 2)] + 60,
            )
            .load_data(verbose=False)
        )
        cond.filter(l_freq=1, h_freq=None, verbose=False)
        cond.filter(l_freq=None, h_freq=100, verbose=False)
        cond = zapline_clean(cond, 50)
        cond.resample(sfreq=201, verbose=False)
        epochs = mne.make_fixed_length_epochs(cond, 0.5, verbose=False, preload=True)

        evaluate(epochs.copy(), [], base_line)

        montage_kind = "standard_1020"
        montage = mne.channels.make_standard_montage(montage_kind)
        raw_copy = (
            raw.copy()
            .crop(
                random_start[eye][int(my_index / 2)],
                random_start[eye][int(my_index / 2)] + 60,
            )
            .load_data(verbose=False)
            .resample(200, verbose=False)
        )
        sample_rate = raw_copy.info["sfreq"]
        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": np.arange(50, sample_rate / 2, 50),
        }

        prep = PrepPipeline(
            raw_copy, prep_params, montage, matlab_strict=True, random_state=435656
        )
        prep.fit()
        bads_name_prep_mat = prep.interpolated_channels + prep.still_noisy_channels
        evaluate(
            epochs.copy().drop_channels(bads_name_prep_mat),
            bads_name_prep_mat,
            results,
            base_line,
        )

        save_folder = pathlib.Path(
            r"C:\Users\workbench\eirik_master\Results\yulin_wang\prep"
        )
        np.save(save_folder / str(eye) / str(my_index), results)


# Run experiments
if __name__ == "__main__":
    p0 = Process(target=process, args=(0,))
    p1 = Process(target=process, args=(2,))
    p2 = Process(target=process, args=(4,))
    p3 = Process(target=process, args=(6,))
    p4 = Process(target=process, args=(8,))
    p5 = Process(target=process, args=(10,))
    p6 = Process(target=process, args=(12,))
    p7 = Process(target=process, args=(14,))
    p8 = Process(target=process, args=(16,))
    p9 = Process(target=process, args=(18,))
    p10 = Process(target=process, args=(20,))
    p11 = Process(target=process, args=(22,))
    p12 = Process(target=process, args=(24,))
    p13 = Process(target=process, args=(26,))
    p14 = Process(target=process, args=(28,))
    p15 = Process(target=process, args=(30,))

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
