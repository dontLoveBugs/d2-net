#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-09-22 18:55
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : qualitative_matches.py
"""

# %%
import cv2

import matplotlib.pyplot as plt

import numpy as np

import os

from PIL import Image

from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

# %% md
## Don't forget to run feature extraction before running this script
"""python extract_features.py --image_list_file image_list_qualitative.txt"""

# %% md
### Change the pair index here (possible values: 1, 2 or 3)
# %%
pair_idx = 2
assert (pair_idx in [1, 2, 3])
# %% md
### Loading the features
# %%
pair_path = os.path.join('images', 'pair_%d' % pair_idx)
# %%
image1 = np.array(Image.open(os.path.join(pair_path, '1.jpg')))
image2 = np.array(Image.open(os.path.join(pair_path, '2.jpg')))
# %%
feat1 = np.load(os.path.join(pair_path, '1.jpg.d2-net'))
feat2 = np.load(os.path.join(pair_path, '2.jpg.d2-net'))
# %% md
### Mutual nearest neighbors matching
# %%
matches = match_descriptors(feat1['descriptors'], feat2['descriptors'], cross_check=True)
# %%
print('Number of raw matches: %d.' % matches.shape[0])
# %% md
### Homography fitting
# %%
keypoints_left = feat1['keypoints'][matches[:, 0], : 2]
keypoints_right = feat2['keypoints'][matches[:, 1], : 2]
np.random.seed(0)
model, inliers = ransac(
    (keypoints_left, keypoints_right),
    ProjectiveTransform, min_samples=4,
    residual_threshold=4, max_trials=10000
)
n_inliers = np.sum(inliers)
print('Number of inliers: %d.' % n_inliers)
# %% md
### Plotting
# %%
inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left[inliers]]
inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right[inliers]]
placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right, placeholder_matches, None)

plt.figure(figsize=(15, 15))
plt.imshow(image3)
plt.axis('off')
plt.show()
# %%
