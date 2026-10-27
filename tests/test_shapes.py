"""
Shape, Architecture and Parameter Count Smoke Tests.
Verifies model parameter counts, evaluation invariants, and forward pass tensor shapes on CPU.
Runs cleanly under both standard `unittest` and `pytest`.
"""
import unittest
import torch
import mlx.core as mx
import mlx.utils
from python.torch_bench import StandardCNN as TorchCNN
from python.mlx_bench import StandardCNN as MlxCNN


class TestModelShapesAndParity(unittest.TestCase):
    def test_torch_cnn_shapes_and_eval(self):
        model = TorchCNN()
        model.eval()
        self.assertFalse(model.training)

        # Check forward output shape on CPU
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([2, 10]))

        # Check exact trainable parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(total_params, 421834)

    def test_mlx_cnn_shapes_and_eval(self):
        model = MlxCNN()
        model.eval()

        # Check forward output shape (MLX uses NHWC format)
        x = mx.random.normal((2, 28, 28, 1))
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

        # Check exact trainable parameter count
        tree = model.trainable_parameters()
        total_params = sum(v.size for _, v in mlx.utils.tree_flatten(tree))
        self.assertEqual(total_params, 421834)


if __name__ == "__main__":
    unittest.main()
