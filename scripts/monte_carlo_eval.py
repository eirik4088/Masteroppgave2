import pathlib
import mne
from data_quality.monte_carlo_search import MonteCarloSearch
from eeg_clean.epoch_stats import EpochStats
import numpy as np
from meegkit import dss
from eeg_clean import clean_new
from data_quality import ica_score
from autoreject import AutoReject
from pyprep import PrepPipeline

data_set = pathlib.Path(r"C:\Users\workbench\eirik_master\Data\texas_data")
results_folder = pathlib.Path(r"C:\Users\workbench\eirik_master\Results\texas_data")

subjects = []
for pth in data_set.iterdir():
    subjects.append(pth)

base_line_epochs_list = []
my_heavy_epochs_list = []
my_heavy_removed = []
my_robust_epochs_list = []
my_robust_removed = []
my_light_epochs_list = []
my_light_removed = []
heavy_auto_epochs_list = []
heavy_auto_removed = []
light_auto_epochs_list = []
light_auto_removed = []
prep_epochs_list = []
prep_removed = []
prep_mat_epochs_list = []
prep_mat_removed = []


for i, sub in enumerate(subjects):

    raw = mne.io.read_raw_bdf(sub, verbose=False, preload=True)
    raw.drop_channels(["M1", "M2", "NAS", "LVEOG", "RVEOG", "LHEOG", "RHEOG", "NFpz"])
    raw.set_montage("biosemi64", verbose=False)

    raw_highpass = raw.copy().filter(l_freq=1, h_freq=None, verbose=False)
    raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=30, verbose=False)
    events = mne.find_events(raw, stim_channel="Status", verbose=False)
    raw_down_sampled = raw_lowpass.copy().resample(
        sfreq=64, verbose=False, events=events
    )
    events = mne.find_events(raw_down_sampled[0], stim_channel="Status", verbose=False)
    raw_down_sampled[0].drop_channels("Status")
    baseline = (0, 0)  # means from the first instant to t = 0
    epochs = mne.Epochs(
        raw=raw_down_sampled[0],
        events=events,
        event_id=[101, 201],
        tmin=0,
        tmax=0.5,
        baseline=None,
        preload=True,
        verbose=False,
    )
    eval_epochs = mne.Epochs(
        raw=raw_down_sampled[0],
        events=events,
        event_id=[101, 201],
        tmin=0,
        tmax=2,
        baseline=None,
        preload=True,
        verbose=False,
    )

    base_line_epochs_list.append(eval_epochs.copy())
    base_stats = EpochStats(eval_epochs.copy())
    base_stats.calc_stability()
    np.save(results_folder / "base_line"  / "accumulate" / "peaks" / "abs_dis" / str(i), base_stats.get_peak_stability().get_mean_abs_stab())
    np.save(results_folder / "base_line"   / "accumulate" / "peaks" / "dis" / str(i), base_stats.get_peak_stability().get_mean_stab())
    np.save(results_folder / "base_line"   / "accumulate" / "quasi" / "abs_dis" / str(i), base_stats.get_quasi_stability().get_mean_abs_stab())
    np.save(results_folder / "base_line"   / "accumulate" / "quasi" / "dis" / str(i), base_stats.get_quasi_stability().get_mean_stab())

    my_processor = clean_new.CleanNew(
        epochs["201"].copy(),
        thresholds=[2.5, 3],
        dist_specifics={
            "quasi": {
                "central": "mean",
                "spred_corrected": "IQR",
            },
            "peak": {
                "central": "mean",
                "spred_corrected": "IQR",
            },
        },
    )
    my_processor2 = clean_new.CleanNew(
        epochs["101"].copy(),
        thresholds=[2.5, 3],
        dist_specifics={
            "quasi": {
                "central": "mean",
                "spred_corrected": "IQR",
            },
            "peak": {
                "central": "mean",
                "spred_corrected": "IQR",
            },
        },
    )

    if (
        my_processor.bad_channels is not None
        and my_processor2.bad_channels is not None
    ):
        bad_ch = np.unique(
            np.concatenate(
                [my_processor.bad_channels, my_processor2.bad_channels]
            )
        )
    elif my_processor.bad_channels is not None:
        bad_ch = my_processor.bad_channels
    elif my_processor2.bad_channels is not None:
        bad_ch = my_processor2.bad_channels
    else:
        bad_ch = []

    if len(bad_ch) > 0:
        my_heavy_removed.append(len(bad_ch))
        my_heavy_epochs_list.append(
            eval_epochs.copy().drop_channels(bad_ch)
        )
        stats = EpochStats(eval_epochs.copy().drop_channels(bad_ch))
        stats.calc_stability()
        np.save(results_folder / "my_heavy" / "accumulate" / "peaks" / "abs_dis" / str(i), stats.get_peak_stability().get_mean_abs_stab())
        np.save(results_folder / "my_heavy" / "accumulate" / "peaks" / "dis" / str(i), stats.get_peak_stability().get_mean_stab())
        np.save(results_folder / "my_heavy" / "accumulate" / "quasi" / "abs_dis" / str(i), stats.get_quasi_stability().get_mean_abs_stab())
        np.save(results_folder / "my_heavy" / "accumulate" / "quasi" / "dis" / str(i), stats.get_quasi_stability().get_mean_stab())
        np.save(results_folder / "my_heavy" / "bad_channels" / str(i), np.array(bad_ch))
    else:
        my_heavy_epochs_list.append(eval_epochs.copy())
        my_heavy_removed.append(0)
        np.save(results_folder  / "my_heavy" / "accumulate" / "peaks" / "abs_dis" / str(i), base_stats.get_peak_stability().get_mean_abs_stab())
        np.save(results_folder  / "my_heavy" / "accumulate" / "peaks" / "dis" / str(i), base_stats.get_peak_stability().get_mean_stab())
        np.save(results_folder  / "my_heavy" / "accumulate" / "quasi" / "abs_dis" / str(i), base_stats.get_quasi_stability().get_mean_abs_stab())
        np.save(results_folder  / "my_heavy" / "accumulate" / "quasi" / "dis" / str(i), base_stats.get_quasi_stability().get_mean_stab())


    my_processor = clean_new.CleanNew(
        epochs["201"].copy(),
        thresholds=[2.5, 5.5],
        dist_specifics={
            "quasi": {
                "central": "median",
                "spred_corrected": "IQR",
            },
            "peak": {
                "central": "median",
                "spred_corrected": "IQR",
            },
        },
    )

    my_processor2 = clean_new.CleanNew(
        epochs["101"].copy(),
        thresholds=[2.5, 5.5],
        dist_specifics={
            "quasi": {
                "central": "median",
                "spred_corrected": "IQR",
            },
            "peak": {
                "central": "median",
                "spred_corrected": "IQR",
            },
        },
    )

    if (
        my_processor.bad_channels is not None
        and my_processor2.bad_channels is not None
    ):
        bad_ch2 = np.unique(
            np.concatenate(
                [my_processor.bad_channels, my_processor2.bad_channels]
            )
        )
    elif my_processor.bad_channels is not None:
        bad_ch2 = my_processor.bad_channels
    elif my_processor2.bad_channels is not None:
        bad_ch2 = my_processor2.bad_channels
    else:
        bad_ch2 = []

    if len(bad_ch2) > 0:
        my_robust_removed.append(len(bad_ch2))
        my_robust_epochs_list.append(
            eval_epochs.copy().drop_channels(bad_ch2)
        )
        stats = EpochStats(eval_epochs.copy().drop_channels(bad_ch2))
        stats.calc_stability()
        np.save(results_folder / "my_robust" / "accumulate" / "peaks" / "abs_dis" / str(i), stats.get_peak_stability().get_mean_abs_stab())
        np.save(results_folder / "my_robust" / "accumulate" / "peaks" / "dis" / str(i), stats.get_peak_stability().get_mean_stab())
        np.save(results_folder / "my_robust" / "accumulate" / "quasi" / "abs_dis" / str(i), stats.get_quasi_stability().get_mean_abs_stab())
        np.save(results_folder / "my_robust" / "accumulate" / "quasi" / "dis" / str(i), stats.get_quasi_stability().get_mean_stab())
        np.save(results_folder / "my_robust" / "bad_channels" / str(i), np.array(bad_ch2))
    else:
        my_robust_epochs_list.append(eval_epochs.copy())
        my_robust_removed.append(0)
        np.save(results_folder  / "my_robust" / "accumulate" / "peaks" / "abs_dis" / str(i), base_stats.get_peak_stability().get_mean_abs_stab())
        np.save(results_folder  / "my_robust" / "accumulate" / "peaks" / "dis" / str(i), base_stats.get_peak_stability().get_mean_stab())
        np.save(results_folder  / "my_robust" / "accumulate" / "quasi" / "abs_dis" / str(i), base_stats.get_quasi_stability().get_mean_abs_stab())
        np.save(results_folder  / "my_robust" / "accumulate" / "quasi" / "dis" / str(i), base_stats.get_quasi_stability().get_mean_stab())


    my_processor = clean_new.CleanNew(
        epochs["201"].copy(),
        thresholds=[3.5, 5.5],
        dist_specifics={
            "quasi": {
                "central": "median",
                "spred_corrected": "IQR",
            },
            "peak": {
                "central": "median",
                "spred_corrected": "IQR",
            },
        },
    )
    my_processor2 = clean_new.CleanNew(
        epochs["101"].copy(),
        thresholds=[3.5, 5.5],
        dist_specifics={
            "quasi": {
                "central": "median",
                "spred_corrected": "IQR",
            },
            "peak": {
                "central": "median",
                "spred_corrected": "IQR",
            },
        },
    )

    if (
        my_processor.bad_channels is not None
        and my_processor2.bad_channels is not None
    ):
        bad_ch3 = np.unique(
            np.concatenate(
                [my_processor.bad_channels, my_processor2.bad_channels]
            )
        )
    elif my_processor.bad_channels is not None:
        bad_ch3 = my_processor.bad_channels
    elif my_processor2.bad_channels is not None:
        bad_ch3 = my_processor2.bad_channels
    else:
        bad_ch3 = []

    if len(bad_ch3) > 0:
        my_light_removed.append(len(bad_ch3))
        my_light_epochs_list.append(
            eval_epochs.copy().drop_channels(bad_ch3)
        )
        stats = EpochStats(eval_epochs.copy().drop_channels(bad_ch3))
        stats.calc_stability()
        np.save(results_folder / "my_light" / "accumulate" / "peaks" / "abs_dis" / str(i), stats.get_peak_stability().get_mean_abs_stab())
        np.save(results_folder / "my_light" / "accumulate" / "peaks" / "dis" / str(i), stats.get_peak_stability().get_mean_stab())
        np.save(results_folder / "my_light" / "accumulate" / "quasi" / "abs_dis" / str(i), stats.get_quasi_stability().get_mean_abs_stab())
        np.save(results_folder / "my_light" / "accumulate" / "quasi" / "dis" / str(i), stats.get_quasi_stability().get_mean_stab())
        np.save(results_folder / "my_light" / "bad_channels" / str(i), np.array(bad_ch3))
    else:
        my_light_epochs_list.append(eval_epochs.copy())
        my_light_removed.append(0)
        np.save(results_folder  / "my_light" / "accumulate" / "peaks" / "abs_dis" / str(i), base_stats.get_peak_stability().get_mean_abs_stab())
        np.save(results_folder  / "my_light" / "accumulate" / "peaks" / "dis" / str(i), base_stats.get_peak_stability().get_mean_stab())
        np.save(results_folder  / "my_light" / "accumulate" / "quasi" / "abs_dis" / str(i), base_stats.get_quasi_stability().get_mean_abs_stab())
        np.save(results_folder  / "my_light" / "accumulate" / "quasi" / "dis" / str(i), base_stats.get_quasi_stability().get_mean_stab())

    epochs_copy = epochs.copy()
    reject = AutoReject(
        consensus=[1.0], n_interpolate=[0], random_state=97, verbose=False
    )
    reject.fit(epochs_copy)
    # find where channels are considered bad, and extract the ones that are bad longer then threshold percentage
    log = reject.get_reject_log(epochs_copy)
    n_epochs = len(epochs_copy)
    n_bads = log.labels.sum(axis=0)
    # Index of bad channels, drop them and evaluate...
    bads_index2 = np.where(n_bads > n_epochs * 0.5)[0]

    if bads_index2.size > 0:
        bads_name2 = [epochs_copy.ch_names[idx] for idx in bads_index2]
        light_auto_epochs_list.append(eval_epochs.copy().drop_channels(bads_name2))
        light_auto_removed.append(bads_index2.size)
        stats = EpochStats(eval_epochs.copy().drop_channels(bads_name2))
        stats.calc_stability()
        np.save(results_folder / "light_auto" / "accumulate" / "peaks" / "abs_dis" / str(i), stats.get_peak_stability().get_mean_abs_stab())
        np.save(results_folder / "light_auto" / "accumulate" / "peaks" / "dis" / str(i), stats.get_peak_stability().get_mean_stab())
        np.save(results_folder / "light_auto" / "accumulate" / "quasi" / "abs_dis" / str(i), stats.get_quasi_stability().get_mean_abs_stab())
        np.save(results_folder / "light_auto" / "accumulate" / "quasi" / "dis" / str(i), stats.get_quasi_stability().get_mean_stab())
        np.save(results_folder / "light_auto" / "bad_channels" / str(i), bads_name2)
    else:
        light_auto_epochs_list.append(eval_epochs.copy())
        light_auto_removed.append(0)
        np.save(results_folder  / "light_auto" / "accumulate" / "peaks" / "abs_dis" / str(i), base_stats.get_peak_stability().get_mean_abs_stab())
        np.save(results_folder  / "light_auto" / "accumulate" / "peaks" / "dis" / str(i), base_stats.get_peak_stability().get_mean_stab())
        np.save(results_folder  / "light_auto" / "accumulate" / "quasi" / "abs_dis" / str(i), base_stats.get_quasi_stability().get_mean_abs_stab())
        np.save(results_folder  / "light_auto" / "accumulate" / "quasi" / "dis" / str(i), base_stats.get_quasi_stability().get_mean_stab())

    montage_kind = "biosemi64"
    montage = mne.channels.make_standard_montage(montage_kind)
    # Extract some info
    sample_rate = raw.info["sfreq"]
    # Make a copy of the data
    raw_copy = raw.copy()
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

    if len(bads_name_prep_mat) > 0:
        prep_mat_epochs_list.append(
            eval_epochs.copy().drop_channels(bads_name_prep_mat)
        )
        prep_mat_removed.append(len(bads_name_prep_mat))
        stats = EpochStats(eval_epochs.copy().drop_channels(bads_name_prep_mat))
        stats.calc_stability()
        np.save(results_folder / "prep_mat" / "accumulate" / "peaks" / "abs_dis" / str(i), stats.get_peak_stability().get_mean_abs_stab())
        np.save(results_folder / "prep_mat" / "accumulate" / "peaks" / "dis" / str(i), stats.get_peak_stability().get_mean_stab())
        np.save(results_folder / "prep_mat" / "accumulate" / "quasi" / "abs_dis" / str(i), stats.get_quasi_stability().get_mean_abs_stab())
        np.save(results_folder / "prep_mat" / "accumulate" / "quasi" / "dis" / str(i), stats.get_quasi_stability().get_mean_stab())
        np.save(results_folder / "prep_mat" / "bad_channels" / str(i), bads_name_prep_mat)
    else:
        prep_mat_epochs_list.append(eval_epochs.copy())
        prep_mat_removed.append(0)
        np.save(results_folder  / "prep_mat" / "accumulate" / "peaks" / "abs_dis" / str(i), base_stats.get_peak_stability().get_mean_abs_stab())
        np.save(results_folder  / "prep_mat" / "accumulate" / "peaks" / "dis" / str(i), base_stats.get_peak_stability().get_mean_stab())
        np.save(results_folder  / "prep_mat" / "accumulate" / "quasi" / "abs_dis" / str(i), base_stats.get_quasi_stability().get_mean_abs_stab())
        np.save(results_folder  / "prep_mat" / "accumulate" / "quasi" / "dis" / str(i), base_stats.get_quasi_stability().get_mean_stab())


base_line = MonteCarloSearch(
    epochs_list=base_line_epochs_list,
    n_resamples=1000,
    repetition_list=[1, 5, 9, 14, 19, 24, 29],
    significance_level=0.05,
    ec_marker="201",
    eo_marker="101",
)
base_line.search()
np.save(results_folder / "base_line" / "delta", base_line.expected_delta_diff)
np.save(results_folder / "base_line" / "theta", base_line.expected_theta_diff)
np.save(results_folder / "base_line" / "alpha", base_line.expected_alpha_diff)
np.save(results_folder / "base_line" / "beta", base_line.expected_beta_diff)
np.save(results_folder / "base_line" / "combined", base_line.expected_diff_percentage)

my_light = MonteCarloSearch(
    epochs_list=my_light_epochs_list,
    n_resamples=1000,
    repetition_list=[1, 5, 9, 14, 19, 24, 29],
    significance_level=0.05,
    ec_marker="201",
    eo_marker="101",
)
my_light.search()
np.save(results_folder / "my_light" / "delta", my_light.expected_delta_diff)
np.save(results_folder / "my_light" / "theta", my_light.expected_theta_diff)
np.save(results_folder / "my_light" / "alpha", my_light.expected_alpha_diff)
np.save(results_folder / "my_light" / "beta", my_light.expected_beta_diff)
np.save(results_folder / "my_light" / "combined", my_light.expected_diff_percentage)
np.save(results_folder / "my_light" / "removed", np.array(my_light_removed))

light_auto = MonteCarloSearch(
    epochs_list=light_auto_epochs_list,
    n_resamples=1000,
    repetition_list=[1, 5, 9, 14, 19, 24, 29],
    significance_level=0.05,
    ec_marker="201",
    eo_marker="101",
)
light_auto.search()
np.save(results_folder / "light_auto" / "delta", light_auto.expected_delta_diff)
np.save(results_folder / "light_auto" / "theta", light_auto.expected_theta_diff)
np.save(results_folder / "light_auto" / "alpha", light_auto.expected_alpha_diff)
np.save(results_folder / "light_auto" / "beta", light_auto.expected_beta_diff)
np.save(results_folder / "light_auto" / "combined", light_auto.expected_diff_percentage)
np.save(results_folder / "light_auto" / "removed", np.array(light_auto_removed))

prep_mat = MonteCarloSearch(
    epochs_list=prep_mat_epochs_list,
    n_resamples=1000,
    repetition_list=[1, 5, 9, 14, 19, 24, 29],
    significance_level=0.05,
    ec_marker="201",
    eo_marker="101",
)
prep_mat.search()
np.save(results_folder / "prep_mat" / "delta", prep_mat.expected_delta_diff)
np.save(results_folder / "prep_mat" / "theta", prep_mat.expected_theta_diff)
np.save(results_folder / "prep_mat" / "alpha", prep_mat.expected_alpha_diff)
np.save(results_folder / "prep_mat" / "beta", prep_mat.expected_beta_diff)
np.save(results_folder / "prep_mat" / "removed", np.array(prep_mat_removed))
np.save(
    results_folder / "prep_mat" / "combined",
    np.array(prep_mat.expected_diff_percentage),
)
