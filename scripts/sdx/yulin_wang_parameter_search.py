import pathlib
import mne
from multiprocessing import Process
import numpy as np
from meegkit import dss
from eeg_clean import clean_new
from data_quality import ica_score

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

quasi = [False, True]
peak = [False, True]
central_meassure = ["mean", "median"]
q_stds = [2, 2.5, 3.5, 4.5]
p_stds = [3, 3.5, 4.5, 5.5]

#Taken from: https://mne.discourse.group/t/clean-line-noise-zapline-method-function-for-mne-using-meegkit-toolbox/7407
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


def evaluate(processor, to_fill: np.ndarray, baseline=None):

    if processor.bad_channels is not None:
        to_fill[0] = len(processor.bad_channels) / 61
    else:

        if baseline is not None:
            return

        to_fill[0] = 0

    if to_fill[0] < 0.72:
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
        print("NaN!!!!")


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
            r"C:\Users\workbench\eirik_master\Results\yulin_wang\results_final"
        )
        np.save(save_folder / str(eye) / "base_line" / f"{my_index}", base_line)
        np.save(save_folder / str(eye) / "quasi" / f"{my_index}", quasi_results)
        np.save(save_folder / str(eye) / "peak" / f"{my_index}", peak_results)
        np.save(save_folder / str(eye) / "combined" / f"{my_index}", combined_results)


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
