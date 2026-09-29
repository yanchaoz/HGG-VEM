"""New release adapter: area-weighted mesh surface samples -> HDF5 cloudpoints."""
import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np
import trimesh


def sample(mesh, count, rng):
    triangles = np.asarray(mesh.triangles)
    areas = np.asarray(mesh.area_faces)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError('Invalid mesh surface area')
    ids = rng.choice(len(triangles), size=count, p=areas/areas.sum())
    uv = rng.random((count,2))
    reflected = uv.sum(axis=1)>1
    uv[reflected] = 1-uv[reflected]
    tri = triangles[ids]
    points = tri[:,0]+uv[:,0,None]*(tri[:,1]-tri[:,0])+uv[:,1,None]*(tri[:,2]-tri[:,0])
    # Put points about the origin before the source augmentation partitions space.
    points -= points.mean(axis=0)
    scale = np.linalg.norm(points,axis=1).max()
    if scale<=0: raise ValueError('Degenerate mesh')
    return (points/scale).astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--points',type=int,default=4096)
    p.add_argument('--seed',type=int,default=20260929)
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.points<1024: p.error('--points must be >=1024')
    files=sorted(a.mesh_root.rglob('*.obj'))
    if a.limit: files=files[:a.limit]
    if not files: raise ValueError('No OBJ inputs')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed)
    with h5py.File(a.output,'x') as h:
        clouds=h.create_dataset('cloudpoints',(len(files),a.points,3),dtype='float32',compression='gzip')
        names=[]
        for i,path in enumerate(files):
            mesh=trimesh.load(path,force='mesh',process=False)
            clouds[i]=sample(mesh,a.points,rng)
            names.append(str(path.relative_to(a.mesh_root)))
        h.create_dataset('source_mesh',data=names,dtype=h5py.string_dtype('utf-8'))
        h.attrs['method']='area-weighted triangle sampling; centroid subtraction; unit maximum radius'
        h.attrs['seed']=a.seed
        h.attrs['historical_training_corpus_reproduction']=False
    print(json.dumps(dict(objects=len(files),points=a.points,output=str(a.output))))


if __name__=='__main__': main()
