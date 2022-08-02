import numpy as np
import torch

from loss import CCCLoss
from eval import calc_ccc, flatten_stress_for_ccc


class TestLosses:

    def test_cccloss(self):
        testdata = np.load('unittests/testdata/debugging_ccc.npz')
        y_true = torch.from_numpy(testdata['y_true'])
        y_pred = torch.from_numpy(testdata['y_pred'])
        seq_lens = torch.from_numpy(testdata['seq_lens']) # ignore seq lens so functions are equivalent

        ccc = CCCLoss()
        ccc_loss = ccc(y_pred, y_true)

        ccc_metric = calc_ccc(flatten_stress_for_ccc(y_pred.numpy()),
                              flatten_stress_for_ccc(y_true.numpy()))

        assert np.isclose(ccc_loss.numpy(), 1-ccc_metric)

