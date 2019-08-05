
import unittest
import numpy as np

from deliravision.torch.metrics import dice_score


class MetricTest(unittest.TestCase):
    
    def test_dice_score(self):
        # check standard case
        gt = np.array([0, 1, 2])[None]

        pred = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])[None]
        self.assertAlmostEqual(dice_score(pred, gt, cls_idx=0), 0.0)
        self.assertAlmostEqual(dice_score(pred, gt, cls_idx=1), 0.5)
        self.assertAlmostEqual(dice_score(pred, gt, cls_idx=2), 0.0)
        self.assertAlmostEqual(dice_score(pred, gt), 0.25)
        self.assertAlmostEqual(dice_score(pred, gt, bg=True), 0.5 / 3)

        pred1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[None]
        self.assertAlmostEqual(dice_score(pred1, gt, cls_idx=0), 1.0)
        self.assertAlmostEqual(dice_score(pred1, gt, cls_idx=1), 1.0)
        self.assertAlmostEqual(dice_score(pred1, gt, cls_idx=2), 1.0)
        self.assertAlmostEqual(dice_score(pred1, gt), 1.0)
        self.assertAlmostEqual(dice_score(pred1, gt, bg=True), 1.0)

        pred0 = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]])[None]
        self.assertAlmostEqual(dice_score(pred0, gt, cls_idx=0), 0.0)
        self.assertAlmostEqual(dice_score(pred0, gt, cls_idx=1), 0.0)
        self.assertAlmostEqual(dice_score(pred0, gt, cls_idx=2), 0.0)
        self.assertAlmostEqual(dice_score(pred0, gt), 0.0)
        self.assertAlmostEqual(dice_score(pred0, gt, bg=True), 0.0)

        # check special case with no foreground gt
        gt = np.array([0, 0, 0])[None]
        pred = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])[None]
        self.assertAlmostEqual(dice_score(pred, gt, cls_idx=0), 0.0)
        self.assertAlmostEqual(dice_score(
            pred, gt, cls_idx=1, no_fg_score=1.0), 1.0)
        self.assertAlmostEqual(dice_score(
            pred, gt, cls_idx=1, no_fg_score=0.0), 0.0)
        self.assertAlmostEqual(dice_score(
            pred, gt, cls_idx=2, no_fg_score=1.0), 1.0)
        self.assertAlmostEqual(dice_score(
            pred, gt, cls_idx=2, no_fg_score=0.0), 0.0)

        self.assertAlmostEqual(dice_score(pred, gt, no_fg_score=1.0), 1.0)
        self.assertAlmostEqual(dice_score(pred, gt, no_fg_score=0.0), 0.0)

        self.assertAlmostEqual(dice_score(
            pred, gt, bg=True, no_fg_score=1.0), 2 / 3)
        self.assertAlmostEqual(dice_score(
            pred, gt, bg=True, no_fg_score=0.0), 0.0)

        # check special case with no foreground prediction
        gt = np.array([1, 2, 0])[None]
        pred = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])[None]
        with self.assertLogs(level='WARNING') as log:
            ds = dice_score(pred, gt, cls_idx=0)
        self.assertAlmostEqual(ds, 0.5)

        with self.assertLogs(level='WARNING') as log:
            ds = dice_score(pred, gt, cls_idx=1)
        self.assertAlmostEqual(ds, 0.0)

        with self.assertLogs(level='WARNING') as log:
            ds = dice_score(pred, gt, cls_idx=2)
        self.assertAlmostEqual(ds, 0.0)

        with self.assertLogs(level='WARNING') as log:
            ds = dice_score(pred, gt)
        self.assertAlmostEqual(ds, 0.0)

        with self.assertLogs(level='WARNING') as log:
            ds = dice_score(pred, gt, bg=True)
        self.assertAlmostEqual(ds, 0.5 / 3)


if __name__ == "__main__":
    unittest.main()
