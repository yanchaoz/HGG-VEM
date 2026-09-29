"""Extend exact <=100 nm calls to full range without changing source masks.

Target foreground is indexed at its native XY grid. Each native voxel denotes
the exact nearest-neighbour 4-nm block (2x2 ER; 4x4 Golgi), not a point at a
coarse voxel centre. KD-tree centre distances provide rigorous search bounds;
candidate distances are evaluated to every block's occupied 4-nm grid range.
All source query points lie on the 4-nm grid, so clamping to the block's XY
range is an exact discrete-voxel-centre distance, not a continuous membrane
distance. Source mitochondrial labels retain their input geometry.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from recompute_mask_proximity import (
    sources, file_map, SHAPES, EXPECTED, PROTOCOL, expand2, require_completed_run,
)

Image.MAX_IMAGE_PIXELS = None


def read_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    with Path(p).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def block_dist2(q, centers, half):
    delta = np.abs(q - centers)
    delta[..., 1:] = np.maximum(0., delta[..., 1:] - half)
    return (delta * delta).sum(axis=-1)


def exact_minimum(points, tree, half, workers=2):
    """Certified global minimum from all query points to target XY blocks."""
    radius = np.sqrt(2.) * half
    # A nearest centre supplies a valid upper bound to the nearest block.
    near, idx = tree.query(points, workers=workers)
    squared = block_dist2(points, tree.data[idx], half)
    best_i = int(np.argmin(squared))
    best2 = float(squared[best_i])
    witness_q = points[best_i].copy()
    witness_c = tree.data[idx[best_i]].copy()
    # A centre farther than best + block half-diagonal cannot improve best.
    keep = near <= np.sqrt(best2) + radius + 1e-8
    pending = np.flatnonzero(keep)
    k = min(4, tree.n)
    rounds = 0
    while len(pending):
        next_pending = []
        for start in range(0, len(pending), max(64, 1000000 // k)):
            ids = pending[start:start + max(64, 1000000 // k)]
            ids = ids[near[ids] <= np.sqrt(best2) + radius + 1e-8]
            if not len(ids):
                continue
            ds, js = tree.query(points[ids], k=k, workers=workers)
            if k == 1:
                ds = ds[:, None]; js = js[:, None]
            ds2 = block_dist2(points[ids, None, :], tree.data[js], half)
            at = np.unravel_index(int(np.argmin(ds2)), ds2.shape)
            value = float(ds2[at])
            if value < best2:
                best2 = value
                witness_q = points[ids[at[0]]].copy()
                witness_c = tree.data[js[at]].copy()
            if k < tree.n:
                unresolved = ds[:, -1] <= np.sqrt(best2) + radius + 1e-8
                next_pending.extend(ids[unresolved].tolist())
        pending = np.asarray(next_pending, dtype=np.int64)
        k = min(tree.n, k * 4)
        rounds += 1
    target = witness_c.copy()
    target[1:] = np.clip(witness_q[1:], witness_c[1:] - half, witness_c[1:] + half)
    assert np.allclose(target[1:] / 4, np.round(target[1:] / 4))
    assert abs(((target - witness_q)**2).sum() - best2) < 1e-7
    return np.sqrt(best2), witness_q, target, rounds


def self_test():
    rng = np.random.RandomState(927)
    for factor in (2, 4):
        half = (factor - 1) * 2.
        for case in range(24):
            target = rng.rand(7, 20, 23) < .02
            coarse = np.argwhere(target).astype(float)
            centers = coarse * [50, factor * 4, factor * 4]
            centers[:, 1:] += half
            expanded = np.repeat(np.repeat(target, factor, 1), factor, 2)
            exact_targets = np.argwhere(expanded) * [50., 4., 4.]
            q = np.column_stack([rng.randint(0, 7, 80) * 50,
                                 rng.randint(-20, factor * 25, 80) * 4,
                                 rng.randint(-20, factor * 25, 80) * 4]).astype(float)
            result, _, _, _ = exact_minimum(q, cKDTree(centers), half, workers=1)
            expected = cKDTree(exact_targets).query(q)[0].min()
            assert abs(result - expected) < 1e-8, (result, expected, factor, case)
    print('SELF_TEST_PASS: 48 expanded-grid independent KDTree comparisons', flush=True)


def main(args):
    start = time.time()
    require_completed_run(args.previous / f'cell{args.cell}')
    rows = [r for r in read_csv(args.primary) if int(r['cell']) == args.cell]
    if any(row.get('measurement_protocol') != PROTOCOL for row in rows):
        raise ValueError('Incompatible primary table; use the current direct-label run.')
    out = args.output / f'cell{args.cell}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        raise RuntimeError('Completed output exists; no overwrite')
    # Exclusive lock prevents accidental repeated job launch.
    fd = os.open(str(out / 'RUNNING.lock'), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    cell = args.cell
    expected = EXPECTED[cell]
    assert len(rows) == expected and len({r['instance_id'] for r in rows}) == expected
    role = 'native_proofread' if cell == 3 else 'native_stitched'
    bounds = {int(r['instance_id']): r for r in read_csv(args.bounds)
              if int(r['cell']) == cell and r['role'] == role}
    manifest = {(r['kind'], int(r['section_1_based'])): r
                for r in read_csv(args.previous / f'cell{cell}' / 'input_manifest.csv')}
    maps = {k: file_map(v, 1 if k == 'golgi' else 0) for k,v in sources(args.root, cell).items()}
    needed = {int(r['instance_id']) for r in rows
              if any(r[f'{partner}_distance_status'] == 'greater_than_100_nm' for partner in ('er','golgi'))}
    cache = out / 'query_cache'; cache.mkdir(exist_ok=True)
    result = {}
    for r in rows:
        ident = int(r['instance_id'])
        for partner in ('er','golgi'):
            if r[f'{partner}_distance_status'] == 'resolved_le100_nm':
                result[(ident, partner)] = dict(distance_nm=float(r[f'{partner}_distance_nm_le100']),
                                               method='reuse_verified_le100',query_points=0)

    def progress(phase, **extra):
        state = dict(cell=cell,phase=phase,elapsed_seconds=round(time.time()-start,1),**extra)
        (out / 'progress.json').write_text(json.dumps(state,indent=2))
        print(json.dumps(state), flush=True)

    def load_verified(kind, z):
        path = maps[kind][z+1]
        old = manifest[(kind,z+1)]
        with Image.open(path) as im:
            a = np.array(im)
        digest = hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
        assert digest == old['decoded_array_sha256'], ('SOURCE_CHANGED',cell,kind,z,path)
        return a

    # Native source crops are streamed once. Exact queried boundary points are
    # cached per ID; no source label or prior result is edited.
    nz, ny, nx = SHAPES[cell]
    begins = {}; active = {}; completed = 0
    for ident in needed:
        b = bounds[ident]
        z0,z1 = int(b['min_z']),int(b['max_z_excl'])
        y0,y1 = max(0,int(b['min_r'])-3),min(ny,int(b['max_r_excl'])+3)
        x0,x1 = max(0,int(b['min_c'])-3),min(nx,int(b['max_c_excl'])+3)
        begins.setdefault(z0,[]).append((ident,z0,z1,y0,y1,x0,x1))
    for z in range(nz):
        a = load_verified('mito', z)
        for ident,z0,z1,y0,y1,x0,x1 in begins.get(z,[]):
            active[ident] = dict(origin=(z0,y0*2,x0*2),end=z1,box=(y0,y1,x0,x1),
                                 mask=np.zeros((z1-z0,(y1-y0)*2,(x1-x0)*2),dtype=bool))
        done = []
        for ident,o in active.items():
            y0,y1,x0,x1 = o['box']
            o['mask'][z-o['origin'][0]] = expand2(a[y0:y1,x0:x1] == ident)
            if z+1 == o['end']:
                done.append(ident)
        for ident in done:
            o = active.pop(ident)
            count = int(o['mask'].sum() // 4)
            assert count == int(bounds[ident]['voxel_count']), (ident,count,bounds[ident]['voxel_count'])
            # Query the boundary of the original support. Only positive (>100 nm)
            # distances reach this stage, so the nearest source voxel is on it.
            # The original mask is not replaced by an eroded or grown object.
            surface = o['mask'] & ~ndi.binary_erosion(o['mask'], border_value=0)
            pts = np.argwhere(surface) + np.asarray(o['origin'])
            np.save(cache / f'{ident}.npy', pts.astype(np.int32))
            completed += 1
        if z % 25 == 0 or z == nz-1:
            progress('extract_source_surfaces',section=z+1,total_sections=nz,completed=completed,total=len(needed))
    assert completed == len(needed) and not active

    for partner,factor in [('er',2),('golgi',4)]:
        half = (factor - 1) * 2.
        pieces = []; total_points = 0
        for z in range(nz):
            target = load_verified(partner,z) != 0
            ij = np.argwhere(target)
            if len(ij):
                xyz = np.empty((len(ij),3),dtype=np.float64)
                xyz[:,0] = z * 50.
                xyz[:,1:] = ij * (factor*4.) + half
                pieces.append(xyz); total_points += len(ij)
            if z % 50 == 0 or z == nz-1:
                progress('index_target',partner=partner,section=z+1,total_sections=nz,native_foreground_voxels=total_points)
        assert total_points > 0
        coordinates = np.concatenate(pieces); del pieces
        tree = cKDTree(coordinates, leafsize=32, balanced_tree=True, compact_nodes=True)
        unresolved = [r for r in rows if (int(r['instance_id']),partner) not in result]
        progress('query_full_distance',partner=partner,completed=0,total=len(unresolved),target_points=total_points)
        audit_samples = []
        for i,r in enumerate(unresolved):
            ident = int(r['instance_id'])
            q = np.load(cache / f'{ident}.npy').astype(float) * [50.,4.,4.]
            d,p,t,rounds = exact_minimum(q,tree,half,args.workers)
            assert d > 100 + 1e-8, ('CONTRADICTS_PRIOR_THRESHOLD',cell,ident,partner,d)
            result[(ident,partner)] = dict(distance_nm=float(d),method='certified_native_block_kdtree',
                query_points=len(q),rounds=rounds,mito_z_nm=float(p[0]),mito_y_nm=float(p[1]),mito_x_nm=float(p[2]),
                target_z_nm=float(t[0]),target_y_nm=float(t[1]),target_x_nm=float(t[2]))
            if len(audit_samples) < 8:
                # Independent local fine-grid expansion of all potentially
                # closer target blocks, with source subpoints bounded by d+R.
                near,_ = tree.query(q,workers=args.workers)
                eligible = q[near <= d + np.sqrt(2)*half + 1e-8]
                ids = set()
                for batch in np.array_split(eligible,max(1,(len(eligible)+999)//1000)):
                    for found in tree.query_ball_point(batch,r=d+np.sqrt(2)*half+1e-8,workers=args.workers):
                        ids.update(found)
                cc = tree.data[np.asarray(sorted(ids),dtype=int)].copy()
                offsets = np.array([(0,y,x) for y in np.arange(factor)*4-half for x in np.arange(factor)*4-half])
                fine = (cc[:,None,:]+offsets).reshape(-1,3)
                check = cKDTree(fine).query(q,workers=args.workers)[0].min()
                assert abs(check-d)<1e-8,(cell,ident,partner,check,d)
                audit_samples.append(dict(instance_id=ident,distance_nm=d,independent_expanded_grid_distance_nm=float(check)))
            if (i+1)%10 == 0 or i+1 == len(unresolved):
                progress('query_full_distance',partner=partner,completed=i+1,total=len(unresolved),last_instance=ident,last_distance_nm=d)
                checkpoint = [dict(instance_id=k[0],partner=k[1],**v) for k,v in sorted(result.items())]
                write_csv(out / 'distance_checkpoint.csv', [{key:r.get(key,'') for key in sorted(set().union(*(v.keys() for v in checkpoint)))} for r in checkpoint])
        (out / f'{partner}_independent_witness_checks.json').write_text(json.dumps(audit_samples,indent=2))
        del tree,coordinates
    final = []
    for row in rows:
        r = dict(row)
        for partner in ('er','golgi'):
            entry = result[(int(row['instance_id']),partner)]
            r[f'{partner}_min_distance_nm'] = entry['distance_nm']
            r[f'{partner}_full_distance_method'] = entry['method']
            for threshold in (30,50,80,100):
                assert int(row[f'{partner}_le_{threshold}nm']) == int(entry['distance_nm'] <= threshold+1e-8)
        final.append(r)
    write_csv(out / 'per_instance_full_distances.csv',final)
    (out / 'COMPLETE.json').write_text(json.dumps(dict(status='PASS',cell=cell,instances=expected,
        exact_distances=len(result),thresholds_unchanged=True,source_hashes_verified=True,
        elapsed_seconds=time.time()-start,primary_sha256=hashlib.sha256(args.primary.read_bytes()).hexdigest(),
        measurement_protocol=PROTOCOL,
        method='all-range exact minimum between supplied labels on the 4-nm nearest-neighbour grid'),indent=2))
    progress('COMPLETE',instances=expected)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--root',type=Path)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--primary',type=Path)
    parser.add_argument('--bounds',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--cell',type=int,choices=(1,2,3))
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.self_test:self_test()
    else:main(args)
