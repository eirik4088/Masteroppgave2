import pathlib
import mne
from multiprocessing import Process
import numpy as np
from meegkit import dss
from eeg_clean import clean_new
from data_quality import ica_score
from pyprep import PrepPipeline

data_set = pathlib.Path(r"C:\Users\workbench\eirik_master\Data\srm_data")

subjects = []
for pth in data_set.iterdir():
    subjects.append(pth)

random_start = [148, 49, 49, 87, 68, 87, 38, 72, 148, 34, 25, 72, 150, 116, 52, 44]

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

    evaluate(epochs.copy(), [], base_line)

    montage_kind = "biosemi64"
    montage = mne.channels.make_standard_montage(montage_kind)
    raw_copy = (
        raw.copy()
        .crop(random_start[my_index], random_start[my_index] + 60)
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

    save_folder =  pathlib.Path(r"C:\Users\workbench\eirik_master\Results\srm_data\prep")
    np.save(save_folder / str(my_index), results)

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
