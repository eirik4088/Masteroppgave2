class nSignChan:
    def __init__(self, mne_epochs_obj):
        self.data = mne_epochs_obj
        self.score = None
    
    def calc_score(self):
        score = self._n_signficant_channels()
        self.score = score
        return score

    def _n_signficant_channels(self):
        return 'inconclusive'