"""CPU tests of public numerical interfaces, using only synthetic inputs."""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
CONTACT = ROOT / 'code/03_contact_analysis'
INSTANCE = ROOT / 'code/02_mitochondrial_instances'
sys.path.insert(0, str(CONTACT))
sys.path.insert(0, str(INSTANCE))
import recompute_mask_proximity as proximity
import extend_fullrange_proximity as extension
import instances_from_probabilities as instances
from assemble_primary_table import assemble


def example_pair():
    records = []
    for partner in ('er', 'golgi'):
        value = 30.0 if partner == 'er' else None
        record = dict(cell=1, instance_id=7, morphology='compact', partner=partner,
                      dilation_xy_pixels_4nm=4,
                      distance_status='resolved_le100_nm' if value is not None else 'greater_than_100_nm',
                      min_distance_nm_le100=value if value is not None else '',
                      golgi_missing_outer_strip_in_halo=1 if partner == 'golgi' else 0)
        for threshold in (30, 50, 80, 100):
            record[f'le_{threshold}nm'] = int(value is not None and value <= threshold)
        records.append(record)
    return records


class NumericalTests(unittest.TestCase):
    def test_source_syntax(self):
        files = list((ROOT / 'code').rglob('*.py'))
        self.assertGreater(len(files), 20)
        for path in files:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                ast.parse(path.read_text(encoding='utf-8-sig'))

    def test_threshold_distance_independent_reference(self):
        proximity.self_test()

    def test_fullrange_independent_expanded_grid(self):
        extension.self_test()

    def test_connected_instances(self):
        fg = np.zeros((4, 24, 32), dtype=np.float32)
        fg[:, 5:18, 3:29] = 0.95
        boundary = np.zeros_like(fg)
        boundary[:, :, 15:17] = 0.95
        result = instances.bc_connected(
            np.rint(np.stack([fg, boundary]) * 255).astype(np.uint8),
            thres1=.65, thres2=.5, thres_small=4, dilation_struct=(1, 1, 1))
        self.assertEqual(np.count_nonzero(np.unique(result)), 2)

    def test_seed_dilation_preserves_support(self):
        interior = np.zeros((4, 24, 32), dtype=np.float32)
        interior[:, 5:18, 3:29] = 0.95
        interior[:, :, 15:17] = 0
        boundary = np.zeros_like(interior)
        boundary[:, 5:18, 15:17] = 0.95
        result = instances.boundary_seeded(interior, boundary, radius=1,
                                            minimum_seed=4, minimum_object=4)
        self.assertEqual(np.count_nonzero(np.unique(result)), 2)
        np.testing.assert_array_equal(result > 0, interior + boundary >= .50)

    def test_instance_cli_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='hgg-instances-') as tmp:
            folder = Path(tmp)
            fg = np.zeros((4, 20, 20), dtype=np.float32)
            fg[:, 4:16, 4:16] = .95
            np.save(folder / 'fg.npy', fg)
            tifffile.imwrite(folder / 'boundary.tif', np.zeros_like(fg), photometric='minisblack')
            command = [sys.executable, str(INSTANCE / 'instances_from_probabilities.py'),
                       '--foreground', str(folder / 'fg.npy'), '--boundary', str(folder / 'boundary.tif'),
                       '--output', str(folder / 'instances.tif')]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            labels = tifffile.imread(folder / 'instances.tif')
            self.assertEqual(labels.shape, fg.shape)
            self.assertEqual(labels.dtype, np.uint32)
            self.assertEqual(np.count_nonzero(np.unique(labels)), 1)
            self.assertTrue((folder / 'instances.json').is_file())
            repeat = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(repeat.returncode, 0)

    def test_probability_range_rejected(self):
        with tempfile.TemporaryDirectory(prefix='hgg-probability-') as tmp:
            path = Path(tmp) / 'bad.npy'
            np.save(path, np.full((2, 3, 4), 2.0, dtype=np.float32))
            with self.assertRaises(ValueError):
                instances.load_probability(path)

    def test_primary_adapter_preserves_censoring_and_flags(self):
        result = assemble(example_pair(), 1, expected=1)[0]
        self.assertEqual(result['er_le_30nm'], 1)
        self.assertEqual(result['golgi_distance_nm_le100'], '')
        self.assertEqual(result['golgi_le_100nm'], 0)
        self.assertEqual(result['golgi_missing_outer_strip_in_halo'], 1)

    def test_primary_adapter_rejects_incomplete_or_duplicate_pairs(self):
        pair = example_pair()
        for rows in [pair[:1], pair + [copy.deepcopy(pair[0])]]:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                assemble(rows, 1, expected=1)

    def test_primary_adapter_rejects_inconsistent_calls(self):
        rows = example_pair()
        rows[0]['le_30nm'] = 0
        with self.assertRaises(ValueError):
            assemble(rows, 1, expected=1)
        rows = example_pair()
        rows[1]['morphology'] = 'elongated'
        with self.assertRaises(ValueError):
            assemble(rows, 1, expected=1)


if __name__ == '__main__':
    unittest.main()
