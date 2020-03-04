from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")
get_ipython().run_line_magic("config", " InlineBackend.figure_format = 'retina'")

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import random_label_cmap, _draw_polygons
from stardist.models import StarDist2D

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.random.seed(6)
lbl_cmap = random_label_cmap()


get_ipython().run_line_magic("cd", " /home/qzt")


X = sorted(glob('data/centuri/test/images/*.tif'))


X


X = list(map(imread,X))


n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels get_ipython().run_line_magic("s."", " % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))")


# show all test images
if True:
    fig, ax = plt.subplots(1,5, figsize=(16,16))
    for i,(a,x) in enumerate(zip(ax.flat, X)):
        a.imshow(x if x.ndim==2 else x[...,0], cmap='gray')
        a.set_title(i)
    [a.axis('off') for a in ax.flat]
    plt.tight_layout()
None;


get_ipython().run_line_magic("pwd", "")


demo_model = False

if demo_model:
    print (
        "NOTE: This is loading a previously trained demo modelget_ipython().getoutput("\n"")
        "      Please set the variable 'demo_model = False' to load your own trained model.",
        file=sys.stderr, flush=True
    )
    model = StarDist2D(None, name='2D_demo', basedir='../../models/examples')
else:
    model = StarDist2D(None, name='stardist', basedir='/home/qzt/stardist/examples/2D/models_test2')
None;


img = normalize(X[0], 1,99.8, axis=axis_norm)
labels, details = model.predict_instances(img)


plt.figure(figsize=(20,20))
plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
plt.axis('off');


def example(model, i, show_dist=True):
    img = normalize(X[i], 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img)

    plt.figure(figsize=(13,10))
    img_show = img if img.ndim==2 else img[...,0]
    coord, points, prob = details['coord'], details['points'], details['prob']
    plt.subplot(121); plt.imshow(img_show, cmap='gray'); plt.axis('off')
    a = plt.axis()
    _draw_polygons(coord, points, prob, grid=model.config.grid, show_dist=show_dist)
    plt.axis(a)
    plt.subplot(122); plt.imshow(img_show, cmap='gray'); plt.axis('off')
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    plt.tight_layout()
    plt.show()


example(model, 1)


example(model, 0)


example(model, 15, False)


model_paper = StarDist2D(None, name='2D_dsb2018', basedir='../../models/paper')


example(model_paper, 29)
