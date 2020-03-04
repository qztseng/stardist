from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")
get_ipython().run_line_magic("config", " InlineBackend.figure_format = 'retina'")

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.models import Config2D, StarDist2D, StarDistData2D

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.random.seed(42)
lbl_cmap = random_label_cmap()


from csbdeep.data import transform

tr = transform.permute_axes("x")


type(tr)


X = sorted(glob('/home/qzt/data/centuri/train/images/*.tif'))
Y = sorted(glob('/home/qzt/data/centuri/train/masks/*.tif'))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))


X = list(map(imread,X))
Y = list(map(imread,Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]


axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels get_ipython().run_line_magic("s."", " % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))")
    sys.stdout.flush()

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]


# assert len(X) > 1, "not enough training data"
# rng = np.random.RandomState(42)
# ind = rng.permutation(len(X))
# n_val = max(1, int(round(0.15 * len(ind))))
# ind_train, ind_val = ind[:-n_val], ind[-n_val:]

#manual assign train/val index for debugging
ind_train = [1,2,3,4]
ind_val = [0]

X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
print('number of images: get_ipython().run_line_magic("3d'", " % len(X))")
print('- training:       get_ipython().run_line_magic("3d'", " % len(X_trn))")
print('- validation:     get_ipython().run_line_magic("3d'", " % len(X_val))")


i = 0
img, lbl = X[i], Y[i]
img, lbl = X_val[0], Y_val[0]
assert img.ndim in (2,3)
img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
plt.figure(figsize=(16,10))
plt.subplot(121); plt.imshow(img,cmap='gray');   plt.axis('off'); plt.title('Raw image')
plt.subplot(122); plt.imshow(lbl,cmap=lbl_cmap); plt.axis('off'); plt.title('GT labels')
None;


print(Config2D.__doc__)


# 32 is a good default choice (see 1_data.ipynb)
n_rays = 32

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = False and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel,
    train_batch_size = 1,  #no gpu
    train_epochs = 400,
    train_steps_per_epoch = 100
)
print(conf)
vars(conf)


# if use_gpu:
#     from csbdeep.utils.tf import limit_gpu_memory
#     # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
#     limit_gpu_memory(0.8)


get_ipython().run_line_magic("pwd", "")


model = StarDist2D(conf, name='stardist', basedir='models_test2')


#skipped temporarily
# median_size = calculate_extents(list(Y), np.median)
# fov = np.array(model._axes_tile_overlap('YX'))
# if any(median_size > fov):
#     print("WARNING: median object size larger than field of view of the neural network.")


augmenter = None

# def augmenter(x, y):
#     """Augmentation of a single input/label image pair.
#     x is an input image
#     y is the corresponding ground-truth label image
#     """
#     # modify a copy of x and/or y...
#     return x, y


get_ipython().run_line_magic("cd", " /home/qzt/stardist/examples/2D")


quick_demo = False

if quick_demo:
    print (
        "NOTE: This is only for a quick demonstrationget_ipython().getoutput("\n"")
        "      Please set the variable 'quick_demo = False' for proper (long) training.",
        file=sys.stderr, flush=True
    )
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter,
                epochs=2, steps_per_epoch=10)

    print("====> Stopping training and loading previously trained demo model from disk.", file=sys.stderr, flush=True)
    model = StarDist2D(None, name='2D_demo', basedir='../../models/examples')
    model.basedir = None # to prevent files of the demo model to be overwritten (not needed for your model)
else:
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)
None;


model.optimize_thresholds(X_val, Y_val)


get_ipython().run_line_magic("pwd", "")


model.export_TF(fname='test2')


# export_TF(self, fname=None, single_output=True, upsample_grid=True)
# tf.saved_model.save(model, "/home/qzt/data/")



