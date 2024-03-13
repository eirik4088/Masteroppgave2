"""_summary_

    _extended_summary_
    """
import numpy as np
import mne

class SignificantChannels:
    def __init__(self, mne_epochs_obj):
        self.data = mne_epochs_obj
        self.score = None
    
    def calc_score(self):
        score = self._n_signficant_channels()
        self.score = score
        return score

    def _n_signficant_channels(self):
        return 'inconclusive'

def test():
    print("riktig?")

if __name__ == "__main__":
    pass