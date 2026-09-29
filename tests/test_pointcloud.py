"""Optional CPU-only representation-learning smoke tests; no checkpoints."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'code/04_pointcloud_ssl'
sys.path.insert(0, str(MODULE))
HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_MESH = all(importlib.util.find_spec(name) is not None for name in ('h5py', 'trimesh'))


class PointCloudTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, 'torch is not installed')
    def test_pointnet_forward(self):
        import torch
        from model.pointnetv2_encoder import PointNetV2
        torch.set_num_threads(2)
        torch.manual_seed(19)
        model = PointNetV2().eval()
        with torch.no_grad():
            result = model(torch.randn(2, 1024, 3))
        self.assertEqual(tuple(result.shape), (2, 1024))
        self.assertTrue(torch.isfinite(result).all())

    @unittest.skipUnless(HAS_TORCH, 'torch is not installed')
    def test_simsiam_backward(self):
        import torch
        from simsiam import SimSiam
        torch.set_num_threads(2)

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 6)

            def forward(self, x):
                return self.linear(x).mean(1)

        torch.manual_seed(41)
        model = SimSiam(TinyEncoder())
        loss = model(torch.randn(4, 64, 3), torch.randn(4, 64, 3))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    @unittest.skipUnless(HAS_MESH, 'h5py/trimesh are not installed')
    def test_mesh_utility(self):
        import h5py
        with tempfile.TemporaryDirectory(prefix='hgg-pointcloud-') as tmp:
            folder = Path(tmp)
            meshroot = folder / 'meshes'
            meshroot.mkdir()
            (meshroot / 'synthetic.obj').write_text(
                'v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n'
                'f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n')
            output = folder / 'clouds.h5'
            run = subprocess.run([sys.executable, str(MODULE / 'mesh_to_pointcloud.py'),
                                  '--mesh-root', str(meshroot), '--output', str(output),
                                  '--points', '1024', '--seed', '19'],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            with h5py.File(output, 'r') as handle:
                points = handle['cloudpoints'][:]
                self.assertEqual(points.shape, (1, 1024, 3))
                self.assertTrue(np.isfinite(points).all())
                np.testing.assert_allclose(points.mean(1), 0, atol=1e-6)
                self.assertAlmostEqual(float(np.linalg.norm(points, axis=2).max()), 1.0, places=6)
                self.assertEqual(handle['source_mesh'].asstr()[0], 'synthetic.obj')
                self.assertFalse(handle.attrs['historical_training_corpus_reproduction'])


if __name__ == '__main__':
    unittest.main()
