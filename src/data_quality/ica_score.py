class icaScore:
    def __init__(self, mne_epochs_obj):
        self.data = mne_epochs_obj
        self.method = None
        self.score = None
    
    def calc_score(self, method):
        if method == 'test':
            self.method = method
            score = self._test()
            self.score = score
        return score
    
    def _test(self):
        return 'inconclusive'