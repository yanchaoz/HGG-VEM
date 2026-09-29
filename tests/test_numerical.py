"""CPU tests of public numerical interfaces, using only synthetic inputs."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import tifffile
from PIL import Image

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
                      measurement_protocol=proximity.PROTOCOL,
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

    def test_direct_label_geometry_is_not_modified(self):
        mito = np.zeros((5, 80, 80), dtype=bool)
        mito[2, 39:41, 39:41] = True
        partner = np.zeros_like(mito)
        partner[2, 39, 48] = True
        before = mito.copy()
        rows, _ = proximity.evaluate(dict(mito=mito, er=partner, golgi=partner,
                                          origin=(0, 0, 0), cell=1, id=7, morphology='compact'))
        np.testing.assert_array_equal(mito, before)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row['min_distance_nm_le100'], 32.0)
            self.assertEqual(row['le_30nm'], 0)
            self.assertEqual(row['le_50nm'], 1)
            self.assertEqual(row['measurement_protocol'], proximity.PROTOCOL)

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

    def test_primary_adapter_rejects_unversioned_results(self):
        rows = example_pair()
        rows[0].pop('measurement_protocol')
        with self.assertRaises(ValueError):
            assemble(rows, 1, expected=1)

    def test_previous_run_protocol_and_completion_gate(self):
        with tempfile.TemporaryDirectory(prefix='hgg-protocol-') as tmp:
            folder = Path(tmp)
            (folder / 'configuration.json').write_text(json.dumps({'measurement_protocol': 'legacy'}))
            (folder / 'COMPLETE.json').write_text(json.dumps({'status': 'PASS'}))
            with self.assertRaises(ValueError):
                proximity.require_completed_run(folder)
            (folder / 'configuration.json').write_text(json.dumps({'measurement_protocol': proximity.PROTOCOL}))
            with self.assertRaises(ValueError):
                proximity.require_completed_run(folder)
            (folder / 'COMPLETE.json').write_text(json.dumps({'status': 'PASS', 'measurement_protocol': proximity.PROTOCOL}))
            self.assertEqual(proximity.require_completed_run(folder)['measurement_protocol'], proximity.PROTOCOL)

    def test_file_based_label_distance_pipeline(self):
        with tempfile.TemporaryDirectory(prefix='hgg-label-pipeline-') as tmp:
            root = Path(tmp)
            paths = proximity.sources(root, 2)
            for folder in paths.values():
                folder.mkdir(parents=True)
            mito = np.zeros((3, 24, 30), dtype=np.uint16)
            mito[:, 4:6, 2:4] = 7
            er = np.zeros_like(mito, dtype=np.uint8)
            er[1, 4, 8] = 1
            golgi = np.zeros((3, 12, 15), dtype=np.uint16)
            golgi[1, 2, 14] = 9
            for kind, volume in [('mito', mito), ('er', er), ('golgi', golgi)]:
                for z, plane in enumerate(volume):
                    section = z if kind == 'golgi' else z + 1
                    Image.fromarray(plane).save(paths[kind] / f'{section}.tif')
            cohort = root / 'cohort.csv'
            bounds = root / 'bounds.csv'
            proximity.write_csv(cohort, [dict(cell='Cell 2', instance_id=7, morphology='compact')])
            proximity.write_csv(bounds, [dict(cell=2, role='native_stitched', instance_id=7,
                                              min_z=0, max_z_excl=3, min_r=4, max_r_excl=6,
                                              min_c=2, max_c_excl=4, voxel_count=12)])
            previous = root / 'thresholds'
            with mock.patch.dict(proximity.EXPECTED, {2: 1}), mock.patch.dict(proximity.SHAPES, {2: mito.shape}):
                proximity.run_cell(SimpleNamespace(root=root, cohort=cohort, bounds=bounds,
                                                    output=previous, cell=2, workers=1, check_only=False))
                long_rows = proximity.read_csv(previous / 'cell2/per_instance_proximity.csv')
                self.assertEqual(len(long_rows), 2)
                primary_rows = assemble(long_rows, 2, expected=1)
                self.assertEqual(float(primary_rows[0]['er_distance_nm_le100']), 36.0)
                self.assertEqual(primary_rows[0]['golgi_distance_nm_le100'], '')
                primary = root / 'primary.csv'
                proximity.write_csv(primary, primary_rows)
                full = root / 'fullrange'
                extension.main(SimpleNamespace(root=root, previous=previous, primary=primary,
                                               bounds=bounds, output=full, cell=2, workers=1))
                final = proximity.read_csv(full / 'cell2/per_instance_full_distances.csv')[0]
                self.assertEqual(float(final['er_min_distance_nm']), 36.0)
                self.assertEqual(float(final['golgi_min_distance_nm']), 196.0)
                self.assertEqual(final['measurement_protocol'], proximity.PROTOCOL)
                self.assertEqual(final['er_le_30nm'], '0')
                self.assertEqual(final['er_le_50nm'], '1')
if __name__ == '__main__':
    unittest.main()
