import pathlib
import mne
from multiprocessing import Process
import numpy as np
from meegkit import dss
from eeg_clean import clean_new
from data_quality import ica_score
import pymatreader

data_set = pathlib.Path(
    r"C:\Users\workbench\eirik_master\Data\SPIS-Resting-State-Dataset\Pre-SART EEG"
)

subjects = []
for pth in data_set.iterdir():
    subjects.append(pth)

random_start = [22, 35, 65, 46, 69, 61, 57, 22, 15, 72, 69, 48, 63, 56, 37, 60]

quasi = [False, True]
peak = [False, True]
central_meassure = ["mean", "median"]
q_stds = [2.5, 3.5, 4.5]
p_stds = [3, 3.5, 4.5, 5.5]


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


def evaluate(processor, to_fill: np.ndarray, baseline=None):

    if processor.bad_channels is not None:
        to_fill[0] = len(processor.bad_channels) / 64
    else:

        if baseline is not None:
            return

        to_fill[0] = 0

    if to_fill[0] < 0.74:
        for_ica = processor.epochs_obj.copy()
        for_ica.set_eeg_reference(verbose=False)

        evaluater = ica_score.IcaScore(for_ica)
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
    quasi_results = np.zeros(
        (
            len(central_meassure),
            len(q_stds),
            5,
        )
    )
    peak_results = np.zeros(
        (
            len(central_meassure),
            len(p_stds),
            5,
        )
    )
    combined_results = np.zeros(
        (
            len(central_meassure),
            len(q_stds),
            len(central_meassure),
            len(p_stds),
            5,
        )
    )

    dict = pymatreader.read_mat(subjects[my_index])

    info = mne.create_info(
        sfreq=256,
        ch_types="eeg",
        ch_names=[
            "Fp1",
            "AF7",
            "AF3",
            "F1",
            "F3",
            "F5",
            "F7",
            "FT7",
            "FC5",
            "FC3",
            "FC1",
            "C1",
            "C3",
            "C5",
            "T7",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "P1",
            "P3",
            "P5",
            "P7",
            "P9",
            "PO7",
            "PO3",
            "O1",
            "Iz",
            "Oz",
            "POz",
            "Pz",
            "CPz",
            "Fpz",
            "Fp2",
            "AF8",
            "AF4",
            "AFz",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "FT8",
            "FC6",
            "FC4",
            "FC2",
            "FCz",
            "Cz",
            "C2",
            "C4",
            "C6",
            "T8",
            "TP8",
            "CP6",
            "CP4",
            "CP2",
            "P2",
            "P4",
            "P6",
            "P8",
            "P10",
            "PO8",
            "PO4",
            "O2",
        ],
    )

    raw = mne.io.RawArray(dict["dataRest"][:64, :], info)

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

            if not qb and not pb:
                processor = clean_new.CleanNew(
                    epochs.copy(), dist_specifics={"dummy_key": None}, thresholds=[None]
                )
                evaluate(processor, base_line)
                continue

            for c, cm in enumerate(central_meassure):

                if qb and not pb:

                    for q_sd, q_std in enumerate(q_stds):
                        processor = clean_new.CleanNew(
                            epochs.copy(),
                            thresholds=[q_std],
                            dist_specifics={
                                "quasi": {
                                    "central": cm,
                                    "spred_corrected": "IQR",
                                }
                            },
                        )
                        evaluate(processor, quasi_results[c, q_sd, :], base_line)

                if not qb and pb:

                    for p_sd, p_std in enumerate(p_stds):

                        processor = clean_new.CleanNew(
                            epochs.copy(),
                            thresholds=[p_std],
                            dist_specifics={
                                "peak": {
                                    "central": cm,
                                    "spred_corrected": "IQR",
                                }
                            },
                        )
                        evaluate(processor, peak_results[c, p_sd, :], base_line)

                else:
                    for q_sd, q_std in enumerate(q_stds):
                        for c2, cm2 in enumerate(central_meassure):
                            for p_sd, p_std in enumerate(p_stds):
                                                            
                                processor = clean_new.CleanNew(
                                    epochs.copy(),
                                    thresholds=[q_std, p_std],
                                    dist_specifics={
                                        "quasi": {
                                            "central": cm,
                                            "spred_corrected": "IQR",
                                        },
                                        "peak": {
                                            "central": cm2,
                                            "spred_corrected": "IQR",
                                        },
                                    },
                                )
                                evaluate(
                                    processor,
                                    combined_results[c, q_sd, c2, p_sd, :],
                                    base_line,
                                )

    save_folder = pathlib.Path(
        r"C:\Users\workbench\eirik_master\Results\SPIS-Resting-State-Dataset\results_final"
    )
    np.save(save_folder / "base_line" / f"{my_index}", base_line)
    np.save(save_folder / "quasi" / f"{my_index}", quasi_results)
    np.save(save_folder / "peak" / f"{my_index}", peak_results)
    np.save(save_folder / "combined" / f"{my_index}", combined_results)


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
