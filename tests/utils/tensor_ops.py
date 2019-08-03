from delira import get_backends
import unittest
import numpy as np


class TensorOpTest(unittest.TestCase):
    def setUp(self) -> None:
        self._labels = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])
        self._n_classes = 6
        self._targets = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ], dtype=np.float)

    @unittest.skipIf("TORCH" not in get_backends(),
                     "No Torch Backend Installed")
    def test_make_onehot_torch(self):
        import torch
        from deliravision.utils.tensor_ops import make_onehot_torch

        self.assertListEqual(self._targets.tolist(),
                             make_onehot_torch(
                                 torch.from_numpy(self._labels).reshape(-1, 1),
                                 self._n_classes).numpy().tolist()
                             )


if __name__ == '__main__':
    unittest.main()