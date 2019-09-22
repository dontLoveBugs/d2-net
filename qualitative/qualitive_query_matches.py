#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-09-22 21:45
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : qualitive_query_matches.py
"""


import cv2
import numpy as np

from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform


def get_feat_path(img_path, feat_ext='.d2-net'):
    return img_path + feat_ext


def read_match_list(match_list_path):
    with open(match_list_path, 'r') as f:
        raw_pairs = f.readlines()

    out_pairs = []
    for raw_pair in raw_pairs:
        im1, im2 = raw_pair.strip('\n').split(' ')
        out_pairs.append((im1, im2))

    return out_pairs


def gen_matches(image1, image2, feat1, feat2):
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
    match_img = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right, placeholder_matches,
                                None)

    return match_img


image_root = '/data/vldata/aachen/images/images_upright'

import os.path as osp

match_list_file = osp.join(image_root, 'night_time_match_list_file.txt')
match_pairs = read_match_list(match_list_file)

match_num = 20
count = 0

from tqdm import tqdm


def item_name(im1, im2):
    item_name = im1 + '-maches-' + im2
    return item_name.replace('/', '_')


def open_image(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = np.array(img_rgb)
    return img


for pair in tqdm(match_pairs, total=len(match_pairs)):
    im1, im2 = pair[0], pair[1]
    item_name = item_name(im1, im2)

    im1 = osp.join(image_root, im1)
    im2 = osp.join(image_root, im2)

    feat1, feat2 = get_feat_path(im1), get_feat_path(im2)

    image1 = open_image(im1)
    image2 = open_image(im2)
    # %%
    feat1 = np.load(feat1)
    feat2 = np.load(feat2)

    match = gen_matches(image1, image2, feat1, feat2)
    match_file = osp.join('./match_resuls/', item_name+'.jpg')

    cv2.imwrite(match_file, match)

