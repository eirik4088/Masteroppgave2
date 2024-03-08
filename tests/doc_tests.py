"""Script to run all doctests from every module in the package.
 """
import unittest
import doctest
from stability import similarity, epoch_stability
from plotting import templates
from eeg_clean import channel_stats, clean, epoch_stats
from data_quality import ica_score, n_sign_chan

suite = doctest.DocTestSuite(similarity)
unittest.TextTestRunner().run(suite)

suite = doctest.DocTestSuite(epoch_stability)
unittest.TextTestRunner().run(suite)

suite = doctest.DocTestSuite(templates)
unittest.TextTestRunner().run(suite)

suite = doctest.DocTestSuite(channel_stats)
unittest.TextTestRunner().run(suite)

suite = doctest.DocTestSuite(epoch_stats)
unittest.TextTestRunner().run(suite)

suite = doctest.DocTestSuite(clean)
unittest.TextTestRunner().run(suite)

suite = doctest.DocTestSuite(ica_score)
unittest.TextTestRunner().run(suite)

suite = doctest.DocTestSuite(n_sign_chan)
unittest.TextTestRunner().run(suite)
