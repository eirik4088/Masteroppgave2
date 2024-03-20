import pathlib
import sklearn
import mne
from multiprocessing import Process
import numpy as np
import pymatreader
from meegkit import dss
from eeg_clean import clean, epoch_stats
from data_quality import ica_score

# Parameters
data_set = pathlib.Path(
    r"C:\Users\workbench\eirik_master\Data\SPIS-Resting-State-Dataset\Pre-SART EEG"
)

epochs_path = pathlib.Path(
    r"C:\Users\workbench\eirik_master\Results\SPIS-Resting-State-Dataset\accumulated_clean"
)

subjects = []
for pth in data_set.iterdir():
    subjects.append(pth)

random_start = random_start = [22, 35, 65, 46, 69, 61, 57, 22, 15, 72, 69, 48, 63, 56, 37, 60]


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
        (   8,
            8,
            5
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

    eeg = raw.copy().crop(random_start[my_index], random_start[my_index] + 60).load_data(verbose=False)
    eeg.filter(l_freq=1, h_freq=None, verbose=False)
    eeg.filter(l_freq=None, h_freq=100, verbose=False)
    eeg = zapline_clean(eeg, 50)
    eeg.resample(sfreq=201, verbose=False)
    epochs = mne.make_fixed_length_epochs(eeg, 0.5, verbose=False, preload=True)

    for cq in range(8):
        for cp in range(8):
            processor = clean.Clean(
                epochs.copy(),
                sklearn_scaler=sklearn.preprocessing.MinMaxScaler(),
                find_random=False,
                top_n = (7-cq, 7-cp)
            )

            if processor.bad_channel_index is not None:
                replicate = epochs.copy().drop_channels(
                    processor.ch_names[processor.bad_channel_index]
                )
            else:
                replicate = epochs.copy()

            if processor.bad_channel_index is not None:
                process_results[cq, cp, 0] = (
                    processor.bad_channel_index.size / 128
                ) * 2
            else:
                process_results[cq, cp, 0] = 0

            if process_results[cq, cp,0] < 2:
                for_ica = replicate.copy()
                for_ica.set_eeg_reference(verbose=False)

                evaluate = ica_score.IcaScore(for_ica)
                process_results[cq, cp, 1] = evaluate.get_n_components()[0]
                process_results[cq, cp, 2] = (
                    evaluate.get_n_components()[0] + evaluate.get_n_components()[1]
                )
                process_results[cq, cp, 3] = evaluate.get_explained_var()["eeg"]
                process_results[cq, cp, 4] = evaluate.get_explained_var(
                    bio_components=True
                )["eeg"]
            else:
                process_results[cq, cp, 1:] = 0

            if (
                process_results[cq, cp, 1] == np.max(process_results[:, :, 1])
                or process_results[cq, cp, 2] == np.max(process_results[:, :, 2])
                or process_results[cq, cp, 3] == np.max(process_results[:, :, 3])
                or process_results[cq, cp, 4] == np.max(process_results[:, :, 4])
            ):
                h = epoch_stats.EpochStats(for_ica)
                h.calc_stability()
                if process_results[cq, cp, 1] == np.max(process_results[:, :, 1]):
                    np.save(
                        epochs_path / str(1) / "quasi" / "dis" / f"{my_index}",
                        h.quasi_stability.get_mean_stab(),
                    )
                    np.save(
                        epochs_path
                        / str(1)
                        / "quasi"
                        / "abs_dis"
                        / f"{my_index}",
                        h.quasi_stability.get_mean_abs_stab(),
                    )
                    np.save(
                        epochs_path / str(1) / "peaks" / "dis" / f"{my_index}",
                        h.peak_stability.get_mean_stab(),
                    )
                    np.save(
                        epochs_path
                        / str(1)
                        / "peaks"
                        / "abs_dis"
                        / f"{my_index}",
                        h.peak_stability.get_mean_abs_stab(),
                    )

                if process_results[cq, cp, 2] == np.max(process_results[:, :, 2]):
                    np.save(
                        epochs_path / str(2) / "quasi" / "dis" / f"{my_index}",
                        h.quasi_stability.get_mean_stab(),
                    )
                    np.save(
                        epochs_path
                        / str(2)
                        / "quasi"
                        / "abs_dis"
                        / f"{my_index}",
                        h.quasi_stability.get_mean_abs_stab(),
                    )
                    np.save(
                        epochs_path / str(2) / "peaks" / "dis" / f"{my_index}",
                        h.peak_stability.get_mean_stab(),
                    )
                    np.save(
                        epochs_path
                        / str(2)
                        / "peaks"
                        / "abs_dis"
                        / f"{my_index}",
                        h.peak_stability.get_mean_abs_stab(),
                    )

                if process_results[cq, cp, 3] == np.max(process_results[:, :, 3]):
                    np.save(
                        epochs_path / str(3) / "quasi" / "dis" / f"{my_index}",
                        h.quasi_stability.get_mean_stab(),
                    )
                    np.save(
                        epochs_path
                        / str(3)
                        / "quasi"
                        / "abs_dis"
                        / f"{my_index}",
                        h.quasi_stability.get_mean_abs_stab(),
                    )
                    np.save(
                        epochs_path / str(3) / "peaks" / "dis" / f"{my_index}",
                        h.peak_stability.get_mean_stab(),
                    )
                    np.save(
                        epochs_path
                        / str(3)
                        / "peaks"
                        / "abs_dis"
                        / f"{my_index}",
                        h.peak_stability.get_mean_abs_stab(),
                    )
                if process_results[cq, cp, 4] == np.max(process_results[:, :, 4]):
                    np.save(
                        epochs_path / str(4) / "quasi" / "dis" / f"{my_index}",
                        h.quasi_stability.get_mean_stab(),
                    )
                    np.save(
                        epochs_path
                        / str(4)
                        / "quasi"
                        / "abs_dis"
                        / f"{my_index}",
                        h.quasi_stability.get_mean_abs_stab(),
                    )
                    np.save(
                        epochs_path / str(4) / "peaks" / "dis" / f"{my_index}",
                        h.peak_stability.get_mean_stab(),
                    )
                    np.save(
                        epochs_path
                        / str(4)
                        / "peaks"
                        / "abs_dis"
                        / f"{my_index}",
                        h.peak_stability.get_mean_abs_stab(),
                    )
    np.save(epochs_path / "all" / f"{my_index}", process_results)

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
