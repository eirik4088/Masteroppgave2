import pathlib
import mne
from multiprocessing import Process
import numpy as np
from meegkit import dss
from eeg_clean import clean_new
from data_quality import ica_score

data_set = pathlib.Path(r"C:\Users\workbench\eirik_master\Data\srm_data")

subjects = []
for pth in data_set.iterdir():
    subjects.append(pth)

random_start = [148, 49, 49, 87, 68, 87, 38, 72, 148, 34, 25, 72, 150, 116, 52, 44]

quasi = [False, True]
peak = [False, True]
central_meassure = ["mean", "median"]
stds = [3, 4, 5]


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

def evaluate(processor, to_fill:np.ndarray, baseline=None):

    if processor.bad_channel_index is not None:
        to_fill[0] = processor.bad_channel_index.size / 64
    else:

        if baseline is not None:
            return
        
        to_fill[0] = 0

    if to_fill[0] < 1:
        for_ica = processor.epochs_obj.copy()
        for_ica.set_eeg_reference(verbose=False)

        evaluater = ica_score.IcaScore(for_ica)
        to_fill[1] = (
            evaluater.get_n_components()[0]
        )
        to_fill[2] = (
            evaluater.get_n_components()[0]
            + evaluater.get_n_components()[1]
        )
        to_fill[3] = (
            evaluater.get_explained_var()["eeg"]
        )
        to_fill[4] = (
            evaluater.get_explained_var(bio_components=True)[
                "eeg"
            ]
        )
        if baseline is not None:
            to_fill[1] = to_fill[1] - baseline[1]
            to_fill[2] = to_fill[2] - baseline[2]
            to_fill[3] = to_fill[3] - baseline[3]
            to_fill[4] = to_fill[4] - baseline[4]
    else:
        to_fill[1:] = float('nan')

def process(my_index):
    base_line = np.zeros(5)
    quasi_results = np.zeros(
        (
            len(central_meassure),
            len(stds),
            5,
        )
    )
    peak_results = np.zeros(
        (
            len(central_meassure),
            len(stds),
            5,
        )
    )
    combined_results = np.zeros(
        (
            len(central_meassure),
            len(stds),
            len(central_meassure),
            len(stds),
            5,
        )
    )

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



    for q, qb in enumerate(quasi):
        for p, pb in enumerate(peak):

            if not qb and not pb:
                processor = clean_new.CleanNew(epochs.copy(), dist_specifics={"dummy_key": None}, thresholds=[None])
                evaluate(processor, base_line)
                continue

            for c, cm in enumerate(central_meassure):
                for sd, std in enumerate(stds):

                    if qb and not pb:
                        processor = clean_new.CleanNew(
                            epochs.copy(),
                            thresholds=[std],
                            dist_specifics={
                                "quasi": {
                                    "central": cm,
                                    "spred_corrected": None,
                                }
                            },
                        )
                        evaluate(processor, quasi_results[c, sd, :], base_line)
                    
                    elif not qb and pb:
                        processor = clean_new.CleanNew(
                            epochs.copy(),
                            thresholds=[std+1],
                            dist_specifics={
                                "peak": {
                                    "central": cm,
                                    "spred_corrected": None,
                                }
                            },
                        )
                        evaluate(processor, peak_results[c, sd, :], base_line)

                    else:
                        for c2, cm2 in enumerate(central_meassure):
                            for sd2, std2 in enumerate(stds):

                                processor = clean_new.CleanNew(
                                    epochs.copy(),
                                    thresholds=[std, std2+1],
                                    dist_specifics={
                                        "quasi": {
                                            "central": cm,
                                            "spred_corrected": None,
                                        },
                                        "peak": {
                                            "central": cm2,
                                            "spred_corrected": None,
                                        },
                                    },
                                )
                                evaluate(processor, combined_results[c, sd, c2, sd2, :], base_line)

    save_folder =  pathlib.Path(r"C:\Users\workbench\eirik_master\Results\srm_data\results_run_1")
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
