import torch.onnx.operators

from src.rl.utils import TensorList
from src.soar.surrogate import SimpleGpSurrogate

import torch

from unittest import TestCase


class TestSimpleGpSurrogate(TestCase):

    def setUp(self) -> None:
        self.parameters = TensorList(iter([
            torch.Tensor([
                [1.0, 2.0],
                [1.0, 1.0]
            ]),
            torch.Tensor([0.5, 1.5])
        ])).flatten()
        self.surrogate = SimpleGpSurrogate(self.parameters,
                                           max_size=4)
        x1 = TensorList(iter([
            torch.Tensor([
                [1, 0],
                [0, 1.0]
            ]),
            torch.Tensor([3.0, -1.0])
        ])).flatten()
        x2 = TensorList(iter([
            torch.Tensor([
                [-1, 1.0],
                [0, 0]
            ]),
            torch.Tensor([2.0, -1.0])
        ])).flatten()
        x3 = TensorList(iter([
            torch.Tensor([
                [0.0, 0],
                [1.0, 1.0]
            ]),
            torch.Tensor([0.0, 1.0])
        ])).flatten()
        self.data_points = torch.stack([x1, x2, x3])
        self.y = torch.Tensor([1, -1, 4])
        self.surrogate.init(self.data_points, self.y)

    def test_init(self):
        surrogate = self.surrogate

        # check to see that the coefficients are close
        r_01 = torch.exp(-torch.square(torch.Tensor([2, -0.5, 0, 1, 2, 0]))).prod()
        r_02 = torch.exp(-torch.square(torch.Tensor([1, 0, -1, 0, 6, -(4 / 3)]))).prod()
        r_12 = torch.exp(-torch.square(torch.Tensor([-1, 0.5, -1, -1, 4, 4 / 3]))).prod()
        corr_matrix = surrogate._corr_matrix

        self.assertTrue(torch.isclose(r_01, corr_matrix[0, 1]))
        self.assertTrue(torch.isclose(r_02, corr_matrix[0, 2]))
        self.assertTrue(torch.isclose(r_12, corr_matrix[1, 2]))

        # check to see that the matrix is symmetric
        self.assertTrue(torch.allclose(corr_matrix, corr_matrix.T))

        # check to see that the data is zero outside the first 3x3 coordinates
        self.assertEqual(0, torch.count_nonzero(corr_matrix[3:, :]))
        self.assertEqual(0, torch.count_nonzero(corr_matrix[:, 3:]))
        self.assertEqual(9, torch.count_nonzero(corr_matrix))

        R = torch.Tensor([
            [1.0, r_01, r_02],
            [r_01, 1.0, r_12],
            [r_02, r_12, 1.0]
        ])
        R_inv = torch.linalg.inv(R)

        mean_estimate = torch.ones(3).T @ R_inv @ self.y / (torch.ones(3).T @ R_inv @ torch.ones(3))
        self.assertTrue(torch.isclose(mean_estimate, surrogate._mean_estimate))

    def test_add_point(self):
        surrogate = self.surrogate

        x4 = TensorList(iter([
            torch.Tensor([
                [-1.0, 2],
                [0.5, 1.0]
            ]),
            torch.Tensor([-1.0, -1.0])
        ])).flatten()
        y4 = torch.Tensor([2.0])

        surrogate.add_point(x4, y4)

        flattened_params = torch.Tensor([1.0, 2.0, 1.0, 1.0, 0.5, 1.5])

        r_03 = torch.exp(-torch.square(torch.Tensor([-2, -0.5, 0, 1, 2, 0]) / flattened_params)).prod()
        r_13 = torch.exp(-torch.square(torch.Tensor([0, -1.0, -0.5, -1, 3, 0]) / flattened_params)).prod()
        r_23 = torch.exp(-torch.square(torch.Tensor([1, -2.0, 0.5, 0, 1, 2]) / flattened_params)).prod()

        corr_matrix = surrogate._corr_matrix
        self.assertTrue(torch.isclose(r_03, corr_matrix[0, 3]))
        self.assertTrue(torch.isclose(r_13, corr_matrix[1, 3]))
        self.assertTrue(torch.isclose(r_23, corr_matrix[2, 3]))

        inv_shape = torch.Size([4, 4])
        self.assertEqual(inv_shape, surrogate._corr_inv.shape)

    def test_incr_size(self):
        surrogate = self.surrogate

        x4 = TensorList(iter([
            torch.Tensor([
                [-1.0, 2],
                [0.5, 1.0]
            ]),
            torch.Tensor([-1.0, -1.0])
        ])).flatten()
        y4 = torch.Tensor([2.0])
        x5 = TensorList(iter([
            torch.Tensor([
                [1.0, 0],
                [0.0, 0.0]
            ]),
            torch.Tensor([0.0, 0.0])
        ])).flatten()
        y5 = torch.Tensor([0.0])

        surrogate.add_point(x4, y4)
        surrogate.add_point(x5, y5)

        new_max_size = 24
        obs_shape = torch.Size([24])
        corr_matr_shape = torch.Size([24, 24])

        self.assertEqual(new_max_size, surrogate._max_size)

        self.assertEqual(new_max_size, surrogate._max_size)
        self.assertEqual(obs_shape, surrogate._observations.shape)
        self.assertEqual(corr_matr_shape, surrogate._corr_matrix.shape)

    def test_predict_val(self):
        surrogate = self.surrogate

        x4 = TensorList(iter([
            torch.Tensor([
                [-1.0, 2],
                [0.5, 1.0]
            ]),
            torch.Tensor([-1.0, -1.0])
        ])).flatten()

        flattened_params = torch.Tensor([1.0, 2.0, 1.0, 1.0, 0.5, 1.5])

        r_03 = torch.exp(-torch.square(torch.Tensor([-2, -0.5, 0, 1, 2, 0]) / flattened_params)).prod()
        r_13 = torch.exp(-torch.square(torch.Tensor([0, -1.0, -0.5, -1, 3, 0]) / flattened_params)).prod()
        r_23 = torch.exp(-torch.square(torch.Tensor([1, -2.0, 0.5, 0, 1, 2]) / flattened_params)).prod()

        pred = (surrogate._mean_estimate + torch.Tensor([r_03, r_13, r_23]).T @ surrogate._corr_inv @
                (surrogate._observations[:3] - surrogate._mean_estimate * torch.ones(3)))

        r = surrogate._compute_corr_vector(x4)
        ones = torch.ones(3)
        temp = (1 - ones.T @ surrogate._corr_inv @ r).square()
        temp /= ones.T @ surrogate._corr_inv @ ones.T
        temp += 1 - r.T @ surrogate._corr_inv @ r
        var_pred = surrogate._var_estimate * temp

        self.assertTrue(torch.isclose(pred, surrogate.predict(x4)))
        self.assertTrue(torch.isclose(var_pred, surrogate.predict_var(x4)))
        print(pred)
        print(surrogate.mean_estimate)
        print(surrogate.predict_var(x4))
        print(surrogate.var_estimate)
        print(r_03, r_13, r_23)
