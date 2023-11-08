import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import mne
import pathlib
import pycrostates
import sklearn
import neurokit2

from skstab import StadionEstimator
from package1.tbd import ModKMeansSkstab, scale_data


from neurokit2.microstates.microstates_clean import microstates_clean
from neurokit2.microstates.microstates_segment import microstates_segment, _microstates_segment_runsegmentation
from neurokit2.microstates.microstates_classify import microstates_classify, _microstates_sort
import numpy as np

class NeurokitCluster():
    def __init__(
        self,
        n_microstates=4,
        train="all",
        method="kmod",
        gfp_method="l1",
        sampling_rate=None,
        standardize_eeg=False,
        n_runs=10,
        max_iterations=1000,
        criterion="gev",
        random_state=None,
        optimize=False,
        **kwargs
    ):
        self._n_microstates = n_microstates
        self._train = train
        self._method = method
        self._gfp_method = gfp_method
        self._sampling_rate = sampling_rate
        self._standardize_eeg = standardize_eeg
        self._n_runs = n_runs
        self._max_iterations = max_iterations
        self._criterion = criterion
        self._random_state = random_state
        self._optimize = optimize
        self._kwargs = kwargs
        
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, data):
        data = data.T
        #print(data.shape)
        #data, peakes, _, _ = microstates_clean(
         #   eeg = data,
          #  train = self._train,
           # sampling_rate = self._sampling_rate, 
            #standardize_eeg = self._standardize_eeg, 
            #gfp_method = self._gfp_method
            #)

        segmented_microstate_dict = microstates_segment(
            eeg = data,
            n_microstates = self._n_microstates,
            train = self._train,
            method = self._method,
            gfp_method = self._gfp_method,
            sampling_rate = self._sampling_rate,
            standardize_eeg = self._standardize_eeg,
            n_runs = self._n_runs,
            max_iterations = self._max_iterations,
            criterion = self._criterion,
            random_state = self._random_state,
            optimize = self._optimize,
            )
        
        self.labels_ = segmented_microstate_dict["Sequence"]
        self.cluster_centers_ = segmented_microstate_dict["Microstates"]
        
        #print(self.labels_.shape)
        return self

    def predict(self, data):
        data = data.T
        pred_labels, _, _, _ = _microstates_segment_runsegmentation(data, self.cluster_centers_, None, self._n_microstates)
        #print(pred_labels.shape)
        return pred_labels
    
