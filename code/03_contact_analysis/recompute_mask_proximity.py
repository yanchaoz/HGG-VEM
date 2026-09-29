"""Exact <=100-nm proximity calls on 4x4x50-nm voxel centres.

Reads sources without editing them. Fixed IDs are never relabelled or dropped.
Native 8-nm mitochondria/ER and 16-nm Golgi are mapped by integer
nearest-neighbour replication in XY, without changing the physical field of
view. Each instance is virtually dilated independently in XY only.
Object-local halos guarantee all <=100-nm calls; larger distances are censored,
not substituted with a maximum distance. This is not a global-distance table.
"""
import argparse
import csv
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None
SPACING = np.array([50., 4., 4.])  # Z,Y,X
RADII = (0,4)  # Primary: 4 raw-grid pixels = 16 nm; 0 is the paired baseline.
THRESHOLDS = (30, 50, 80, 100)
EXPECTED = {1:2676, 2:530, 3:624}
SHAPES = {1:(544,6625,8480), 2:(340,4500,5000), 3:(415,4560,4728)}


def disk(r):
    y,x=np.ogrid[-r:r+1,-r:r+1]
    return (x*x+y*y <= r*r)[None,:,:]


def expand2(a):
    return np.repeat(np.repeat(a,2,axis=-2),2,axis=-1)


def file_map(folder,section_offset=0):
    answer={}
    for p in folder.iterdir():
        if p.suffix.lower() not in ('.png','.tif','.tiff'):continue
        match=re.search(r'(\d+)',p.stem)
        if match:
            n=int(match.group(1))+section_offset
            assert n not in answer, (folder,n)
            answer[n]=p
    return answer


def sources(root,cell):
    b=root/f'result/cell{cell}_masked_test'
    new=root/f'dist_result_2026/data/cell{cell}'
    return {'mito':b/('mito_instance/mito_after_proof' if cell==3 else 'mito_instance/snitched_stacks'),
            'er':new/'er', 'golgi':new/'golgi/golgi_instance'}


def physical_crop(a,y0,y1,x0,x1,factor):
    """Map raw-grid coordinates to a zero-origin coarse grid, without stretch.

    Only an outer strip beyond the actual coarse field is padded. Callers
    separately audit whether any target-object halo touches this strip.
    """
    yy=np.arange(y0,y1)//factor;xx=np.arange(x0,x1)//factor
    vy=yy<a.shape[0];vx=xx<a.shape[1]
    answer=np.zeros((y1-y0,x1-x0),bool)
    answer[np.ix_(vy,vx)]=a[np.ix_(yy[vy],xx[vx])]
    return answer


def nearest_target(partner,point):
    lo=np.maximum(np.array(point)-np.array([2,25,25]),0)
    hi=np.minimum(np.array(point)+np.array([3,26,26]),partner.shape)
    pts=np.argwhere(partner[tuple(slice(a,b) for a,b in zip(lo,hi))])+lo
    ds=np.sum(((pts-point)*SPACING)**2,axis=1)
    return pts[np.argmin(ds)],float(np.sqrt(ds.min()))


def evaluate(obj):
    mito=obj['mito']
    positions={r:np.flatnonzero(mito if r==0 else ndi.binary_dilation(mito,structure=disk(r))) for r in RADII}
    assert all(len(p)>0 for p in positions.values()),obj['id']
    answer=[]
    origin=np.array(obj['origin'])
    for name in ('er','golgi'):
        partner=obj[name]
        dist=ndi.distance_transform_edt(~partner,sampling=SPACING) if partner.any() else None
        previous=np.inf
        for r in RADII:
            row=dict(cell=obj['cell'],instance_id=obj['id'],morphology=obj['morphology'],
                     partner=name,dilation_xy_pixels_4nm=r,dilation_nm=4*r,
                     golgi_missing_outer_strip_in_halo=int(obj.get('golgi_missing_strip_in_halo',False)) if name=='golgi' else 0,
                     distance_status='greater_than_100_nm',min_distance_nm_le100='',
                     mito_z='',mito_y='',mito_x='',partner_z='',partner_y='',partner_x='')
            if dist is not None:
                vals=dist.ravel()[positions[r]]
                i=int(np.argmin(vals)); d=float(vals[i])
                assert d <= previous+1e-9,(obj['id'],r,d,previous)
                previous=d
                if d <= 100+1e-9:
                    p=np.array(np.unravel_index(positions[r][i],mito.shape))
                    q,check=nearest_target(partner,p)
                    assert abs(check-d)<1e-9,(check,d)
                    row.update(distance_status='resolved_le100_nm',min_distance_nm_le100=d)
                    for prefix,point in [('mito',p+origin),('partner',q+origin)]:
                        for axis,value in zip('zyx',point):row[prefix+'_'+axis]=int(value)
            else:d=np.inf
            for t in THRESHOLDS:row[f'le_{t}nm']=int(d <= t+1e-9)
            answer.append(row)
    return answer,int(mito.sum()//4)


def self_test():
    # Compare against an independent point-cloud nearest-neighbour calculation.
    rng=np.random.RandomState(1026)
    for case in range(8):
        m=np.zeros((7,55,65),bool);m[3,26:29,31:34]=True
        p=rng.rand(*m.shape)<0.001
        if case==0:p[:]=False
        obj=dict(mito=m,er=p,golgi=p,origin=(0,0,0),cell=0,id=1,morphology='test')
        rows,_=evaluate(obj)
        tree=cKDTree(np.argwhere(p)*SPACING) if p.any() else None
        for row in rows:
            r=row['dilation_xy_pixels_4nm']
            expanded=m if r==0 else ndi.binary_dilation(m,structure=disk(r))
            d=float(tree.query(np.argwhere(expanded)*SPACING)[0].min()) if tree else np.inf
            for t in THRESHOLDS:assert row[f'le_{t}nm']==int(d<=t+1e-9)
            if d<=100:assert abs(d-row['min_distance_nm_le100'])<1e-9
    # A same-XY adjacent section is exactly 50 nm, not 4 nm.
    m=np.zeros((5,80,80),bool);m[1,40,40]=1
    p=np.zeros_like(m);p[2,40,40]=1
    rows,_=evaluate(dict(mito=m,er=p,golgi=p,origin=(0,0,0),cell=0,id=1,morphology='test'))
    assert all(r['le_30nm']==0 and r['le_50nm']==1 for r in rows)
    # A distant target omitted by the finite halo cannot change <=100 calls.
    m=np.zeros((9,150,150),bool);m[4,75,75]=1
    p=np.zeros_like(m);p[4,75,100]=1;p[0,0,0]=1
    full,_=evaluate(dict(mito=m,er=p,golgi=p,origin=(0,0,0),cell=0,id=1,morphology='test'))
    sl=(slice(1,8),slice(43,108),slice(43,108))
    crop,_=evaluate(dict(mito=m[sl],er=p[sl],golgi=p[sl],origin=(1,43,43),cell=0,id=1,morphology='test'))
    assert [r['min_distance_nm_le100'] for r in full]==[r['min_distance_nm_le100'] for r in crop]
    # Cropped nearest-neighbour sampling agrees exactly with full expansion.
    a=rng.rand(21,23)>.7
    for f in (2,4):
        full=np.repeat(np.repeat(a,f,axis=0),f,axis=1)
        assert np.array_equal(physical_crop(a,3,37,7,45,f),full[3:37,7:45])
    print('SELF_TEST_PASS: KDTree, empty target, Z anisotropy, inclusive threshold, halo.',flush=True)


def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run_cell(args):
    start=time.time();cell=args.cell
    out=args.output/f'cell{cell}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():raise RuntimeError('Completed run exists; use a new output directory.')
    cohort={int(r['instance_id']):r for r in read_csv(args.cohort) if r['cell']==f'Cell {cell}'}
    assert len(cohort)==EXPECTED[cell]
    role='native_proofread' if cell==3 else 'native_stitched'
    bounds={int(r['instance_id']):r for r in read_csv(args.bounds) if int(r['cell'])==cell and r['role']==role and int(r['instance_id']) in cohort}
    assert set(bounds)==set(cohort)
    maps={k:file_map(p,section_offset=1 if k=='golgi' else 0) for k,p in sources(args.root,cell).items()}
    nz,ny,nx=SHAPES[cell]
    for k,m in maps.items():assert set(m)==set(range(1,nz+1)),(k,len(m))
    config=dict(cell=cell,cohort_size=len(cohort),cohort_sha256=hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
                bounds_sha256=hashlib.sha256(args.bounds.read_bytes()).hexdigest(),
                source_folders={k:str(p) for k,p in sources(args.root,cell).items()},
                xyz_spacing_nm=[4,4,50],instance_native_xyz_spacing_nm=[8,8,50],er_native_xyz_spacing_nm=[8,8,50],golgi_native_xyz_spacing_nm=[16,16,50],
                dilation_radii_xy_4nm_pixels=RADII,z_dilation=0,thresholds_nm=THRESHOLDS,
                threshold_rule='minimum foreground-voxel-centre Euclidean distance <= threshold',
                er_foreground='nonzero',golgi_foreground='nonzero instance ID/palette index',additional_body_mask=False,
                halo_xy_4nm_pixels=32,halo_z_sections=3,
                resampling='nearest neighbour, integer replication, XYZ origin unchanged, no whole-field stretch',
                primary_dilation_4nm_pixels=4,
                golgi_field_edge='Cell 1 has two fewer raw-grid rows; missing outer strip is outside observed target support, never stretched. Affected object halos are flagged.',
                note='ER/Golgi from newly copied data only; Cell 3 proofread mitochondrial IDs. Old cohort distances are not imported.')
    (out/'configuration.json').write_text(json.dumps(config,indent=2))
    begins={};objects={}
    for ident,b in bounds.items():
        b={k:int(v) for k,v in b.items() if k not in ['role','touches_outer_face']}
        z0=max(0,b['min_z']-3);z1=min(nz,b['max_z_excl']+3)
        y0=max(0,b['min_r']-16);y1=min(ny,b['max_r_excl']+16)
        x0=max(0,b['min_c']-16);x1=min(nx,b['max_c_excl']+16)
        o=dict(cell=cell,id=ident,morphology=cohort[ident]['morphology'],bounds=b,
               origin=(z0,y0*2,x0*2),native_xy=(y0,y1,x0,x1),end=z1,
               shape=(z1-z0,(y1-y0)*2,(x1-x0)*2))
        # Preserve the actual 13248-row Golgi FOV in Cell 1, not a stretched
        # 13250-row image. Distances are to observed labels only. Retain and
        # flag objects near this small outer strip instead of dropping IDs.
        o['golgi_missing_strip_in_halo']=y1*2 > (ny//2)*4
        begins.setdefault(z0,[]).append(o);objects[ident]=o
    if args.check_only:
        volumes=sorted([(int(np.prod(o['shape'])),i,o['shape']) for i,o in objects.items()],reverse=True)
        print(json.dumps(dict(cell=cell,validated_input_file_counts={k:len(m) for k,m in maps.items()},
                              cohort=len(cohort),largest_crops=volumes[:5],
                              golgi_missing_strip_halo_ids=[i for i,o in objects.items() if o['golgi_missing_strip_in_halo']],
                              total_object_crop_voxels=sum(v[0] for v in volumes))),flush=True)
        return
    counts=np.zeros(max(max(maps['mito']),max(cohort))+65536,dtype=np.int64)
    results=[];cropcounts={};active={};pending=set();manifest=[]
    progress_path=out/'progress.json'
    def consume(done):
        for future in done:
            rows,n=future.result();results.extend(rows);cropcounts[rows[0]['instance_id']]=n
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for z in range(nz):
            arrays={}
            for kind,m in maps.items():
                path=m[z+1]
                with Image.open(path) as im:a=np.array(im)
                expected=(ny//2,nx//2) if kind=='golgi' else (ny,nx)
                assert a.shape==expected,(cell,kind,z,a.shape,expected)
                manifest.append(dict(kind=kind,section_1_based=z+1,path=str(path),shape_y=a.shape[0],shape_x=a.shape[1],
                                     dtype=str(a.dtype),file_size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
                                     decoded_array_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()))
                if kind=='mito':
                    bc=np.bincount(a.ravel());counts[:len(bc)]+=bc
                else:
                    values=np.unique(a)
                    if kind=='er':assert set(values)<={0,1},(path,values)
                    else:assert values.min()>=0,(path,values)
                    a=a!=0
                arrays[kind]=a
            for o in begins.pop(z,[]):
                for key in ('mito','er','golgi'):o[key]=np.zeros(o['shape'],dtype=bool)
                active[o['id']]=o
            finished=[]
            for ident,o in active.items():
                y0,y1,x0,x1=o['native_xy'];zz=z-o['origin'][0]
                o['mito'][zz]=expand2(arrays['mito'][y0:y1,x0:x1]==ident)
                o['golgi'][zz]=physical_crop(arrays['golgi'],y0*2,y1*2,x0*2,x1*2,4)
                o['er'][zz]=expand2(arrays['er'][y0:y1,x0:x1])
                if z+1==o['end']:finished.append(ident)
            for ident in finished:
                o=active.pop(ident)
                # Remove the global reference so completed crops can be freed.
                objects.pop(ident)
                pending.add(executor.submit(evaluate,o))
                if len(pending)>=args.workers*2:
                    done,pending=wait(pending,return_when=FIRST_COMPLETED);consume(done)
            if z%10==0 or z==nz-1:
                done={f for f in pending if f.done()};pending-=done;consume(done)
                state=dict(cell=cell,section=z+1,total_sections=nz,completed_instances=len(cropcounts),
                           active_instances=len(active),queued_objects=len(pending),elapsed_seconds=round(time.time()-start,1))
                progress_path.write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
            del arrays
        consume(pending)
    assert not active and len(cropcounts)==len(cohort)
    audits=[]
    for ident,b in sorted(bounds.items()):
        expected=int(b['voxel_count']);observed=int(counts[ident]);cropped=cropcounts[ident]
        audits.append(dict(cell=cell,instance_id=ident,archived_native_voxels=expected,current_native_voxels=observed,
                           crop_native_voxels=cropped,exact_count_match=expected==observed==cropped))
    write_csv(out/'cohort_count_audit.csv',audits)
    assert all(r['exact_count_match'] for r in audits),'Source or bounding-box drift: results not certified.'
    results.sort(key=lambda r:(r['instance_id'],r['partner'],r['dilation_xy_pixels_4nm']))
    assert len(results)==EXPECTED[cell]*2*len(RADII)
    write_csv(out/'per_instance_proximity.csv',results)
    write_csv(out/'input_manifest.csv',manifest)
    summary=[]
    for name in ('er','golgi'):
        for radius in RADII:
            selected=[r for r in results if r['partner']==name and r['dilation_xy_pixels_4nm']==radius]
            for t in THRESHOLDS:
                positive=sum(r[f'le_{t}nm'] for r in selected)
                summary.append(dict(cell=cell,partner=name,dilation_xy_pixels_4nm=radius,dilation_nm=radius*4,
                                    threshold_nm=t,positive=positive,denominator=EXPECTED[cell],percent=100*positive/EXPECTED[cell]))
    write_csv(out/'threshold_summary.csv',summary)
    (out/'COMPLETE.json').write_text(json.dumps(dict(cell=cell,elapsed_seconds=time.time()-start,
                                  validated_instances=len(cohort),per_instance_rows=len(results),status='PASS'),indent=2))
    print('COMPLETE',cell,'instances',len(cohort),'seconds',round(time.time()-start),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true')
    p.add_argument('--root',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--cohort',type=Path);p.add_argument('--bounds',type=Path)
    p.add_argument('--cell',type=int,choices=(1,2,3));p.add_argument('--workers',type=int,default=3)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test()
    else:run_cell(args)
