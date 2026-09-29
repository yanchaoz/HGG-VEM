"""Mitochondrial instances from foreground/interior and contour probabilities.

Release CLI adapter; original connected-component kernel is retained alongside.
No network inference or distributed seam reconciliation is performed here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
import tifffile

from connectomics.process import bc_connected


def remove_small(instances, minimum):
    counts=np.bincount(instances.ravel())
    rejected=counts<minimum
    rejected[0]=False
    result=instances.copy()
    result[rejected[result]]=0
    return result


def boundary_seeded(interior, boundary, radius=1, minimum_seed=4, minimum_object=16):
    """Archived 0.35 probability barrier affects seeds only, not support mask."""
    barrier=boundary>=0.35
    if radius:
        y,x=np.ogrid[-radius:radius+1,-radius:radius+1]
        barrier=binary_dilation(barrier,structure=(x*x+y*y<=radius*radius)[None])
    support=interior+boundary>=0.50
    seeds=remove_small_objects(label((interior>=0.50)&~barrier),minimum_seed)
    return remove_small(watershed(-interior.astype(np.float32),seeds,mask=support),minimum_object)


def load_probability(path):
    array=np.load(path,allow_pickle=False) if path.suffix=='.npy' else tifffile.imread(path)
    if array.ndim!=3: raise ValueError('Expected a ZYX volume')
    if not np.isfinite(array).all(): raise ValueError('Non-finite probabilities')
    if array.dtype==np.uint8:
        return array.astype(np.float32)/255.0,array
    array=array.astype(np.float32)
    if array.min()<0 or array.max()>1: raise ValueError('Float probabilities must be in [0,1]')
    return array,np.rint(array*255).astype(np.uint8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--foreground',type=Path,required=True)
    p.add_argument('--boundary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=['connected','validation-watershed'],default='connected')
    p.add_argument('--foreground-threshold',type=float,default=0.65)
    p.add_argument('--boundary-threshold',type=float,default=0.50)
    p.add_argument('--dilation-shape-zyx',type=int,nargs=3,default=[1,5,5])
    p.add_argument('--minimum-object-voxels',type=int,default=128)
    p.add_argument('--seed-radius-xy',type=int,default=1)
    p.add_argument('--body-mask',type=Path)
    p.add_argument('--minimum-max-slice-area',type=int,default=0)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    fg,fg8=load_probability(a.foreground);bd,bd8=load_probability(a.boundary)
    if fg.shape!=bd.shape: raise ValueError('Probability grids differ; align before calling')
    if a.method=='connected':
        instances=bc_connected(np.stack([fg8,bd8]),thres1=a.foreground_threshold,
            thres2=a.boundary_threshold,thres_small=a.minimum_object_voxels,
            dilation_struct=tuple(a.dilation_shape_zyx))
    else:
        instances=boundary_seeded(fg,bd,a.seed_radius_xy,minimum_object=a.minimum_object_voxels)
    if a.body_mask:
        body=tifffile.imread(a.body_mask)!=0
        if body.shape!=instances.shape: raise ValueError('Body mask grid differs')
        instances=np.where(body,instances,0)
    if a.minimum_max_slice_area:
        maxima=np.zeros(int(instances.max())+1,dtype=np.int64)
        for plane in instances:
            counts=np.bincount(plane.ravel(),minlength=len(maxima))
            maxima=np.maximum(maxima,counts)
        keep=maxima>=a.minimum_max_slice_area;keep[0]=False
        instances=np.where(keep[instances],instances,0)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tifffile.imwrite(a.output,instances.astype(np.uint32),metadata={'axes':'ZYX'})
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    settings.update(shape_zyx=list(instances.shape),objects=int(np.count_nonzero(np.unique(instances))))
    a.output.with_suffix('.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    print(json.dumps(settings))


if __name__=='__main__': main()
