"""
Test the build model functions.
"""

import unittest

import torch

from gerbilizer.architectures.simplenet import GerbilizerSimpleNetwork
from gerbilizer.architectures.ensemble import GerbilizerEnsemble

from constants import (
    SIMPLENET_BASE,
    SIMPLENET_COV,
    ENSEMBLE,
    ENSEMBLE_AVG,
    ENSEMBLE_MISSING_COV,
)


class TestOuptutShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # number of channels of input with four mics and passed-in cross correlations
        NUM_CHANNELS_MICS_AND_XCORR = 10
        cls.fake_input = torch.zeros(
            1, NUM_CHANNELS_MICS_AND_XCORR, SIMPLENET_BASE["SAMPLE_LEN"]
        )

    def test_simplenet(self):
        # test loading the models
        cnn_no_cov = GerbilizerSimpleNetwork(SIMPLENET_BASE)
        cnn_cov = GerbilizerSimpleNetwork(SIMPLENET_COV)

        no_cov_expected_shape = (1, 2)
        cov_expected_shape = (1, 3, 2)

        self.assertEqual(cnn_no_cov(self.fake_input).shape, no_cov_expected_shape)
        self.assertEqual(cnn_cov(self.fake_input).shape, cov_expected_shape)

    def test_ensemble(self):
        ensemble = GerbilizerEnsemble(ENSEMBLE)
        expected_shape = (1, len(ENSEMBLE["MODELS"]), 3, 2)
        self.assertEqual(ensemble(self.fake_input).shape, expected_shape)

        avgd_expected_shape = (1, 3, 2)
        avg_ensemble = GerbilizerEnsemble(ENSEMBLE_AVG)
        self.assertEqual(avg_ensemble(self.fake_input).shape, avgd_expected_shape)

        # test that the class throws an error if one of the models
        # doesn't have the 'OUTPUT_COV' flag
        with self.assertRaises(ValueError):
            GerbilizerEnsemble(ENSEMBLE_MISSING_COV)
