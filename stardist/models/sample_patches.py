"""provides a faster sampling function"""

import numpy as np
from csbdeep.utils import _raise, choice


def sample_patches(datas, patch_size, n_samples, valid_inds=None, verbose=False):
    """optimized version of csbdeep.data.sample_patches_from_multiple_stacks"""

    len(patch_size)==datas[0].ndim or _raise(ValueError())

    if not all(( a.shape == datas[0].shape for a in datas )):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not all(( 0 < s <= d for s,d in zip(patch_size,datas[0].shape) )):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % (str(patch_size), str(datas[0].shape)))

    if valid_inds is None:
        valid_inds = tuple(_s.ravel() for _s in np.meshgrid(*tuple(np.arange(p//2,s-p//2+1) for s,p in zip(datas[0].shape, patch_size))))

    n_valid = len(valid_inds[0])

    if n_valid == 0:
        raise ValueError("no regions to sample from!")

    idx = choice(range(n_valid), n_samples, replace=(n_valid < n_samples))
    rand_inds = [v[idx] for v in valid_inds]
    res = [np.stack([data[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,patch_size))] for r in zip(*rand_inds)]) for data in datas]

    return res


def get_valid_inds(datas, patch_size, patch_filter = None):
    len(patch_size)==datas[0].ndim or _raise(ValueError())

    if not all(( a.shape == datas[0].shape for a in datas )):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not all(( 0 < s <= d for s,d in zip(patch_size,datas[0].shape) )):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % (str(patch_size), str(datas[0].shape)))

    if patch_filter is None:
        patch_mask = np.ones(datas[0].shape,dtype=np.bool)
    else:
        patch_mask = patch_filter(datas[0], patch_size)

    # get the valid indices
    # to get the pixel indices for cropping(sample patches) so that each patches will not extend outside the image dimension
    # for example, a image of (520, 696) and patch_size = (256,256) will give valid index ranges from (128,393), (128,569)
    # which means among above ranges patches of 256,256 could be safely drawn without getting outside the image dimension
    # ind is the x,y of the patch center pixel. 
    # can be considered as a valid crop region 
    border_slices = tuple([slice(p // 2, s - p + p // 2 + 1) for p, s in zip(patch_size, datas[0].shape)])
    # the first valid_inds is just tuple of two 1D arrays, row index for 0 to (border_slices[0].start - border_slices[0].end )
    # so that a zip of the 2 arrays will give the x,y index to select within the valid crop region
    valid_inds = np.where(patch_mask[border_slices])
    # here the starting pixel of the border_slices is added to the above inds so that it will be starting at 
    # the border_slices.start rather than 0
    valid_inds = tuple(v + s.start for s, v in zip(border_slices, valid_inds))
    # for patch_filter = None, the valid_inds is just all the x,y indices within the valid crop region
    # so that later on when calling sample patches it will just randomly pick one (or n) indeces and -,+ patch_size//2 to 
    # specify the crop region
    return valid_inds
