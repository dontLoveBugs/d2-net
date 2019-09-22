#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-09-19 15:07
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : dataset.py
"""

import os
import os.path as osp


class DataBase(object):

    def __init__(self, root, slice=None, ext='.jpg'):
        self.root = root
        self.slice = slice
        self.img_ext = ext

        self.db_imgs = self._get_db_imgs()
        self.query_imgs = self._get_query_imgs()

    def _get_db_imgs(self):
        raise NotImplementedError

    def _get_query_imgs(self):
        raise NotImplementedError


class AAcchenDB(DataBase):

    def __init__(self, root, slice=None, img_ext='.jpg'):
        self.images_root = osp.join(root, 'images', 'images_upright')
        super(AAcchenDB, self).__init__(root, slice, img_ext)

        print('aachen slice {} in aachen has {} db images and {} query images.'.format(
            self.slice, len(self.db_imgs), len(self.query_imgs)
        ))

    def _get_db_imgs(self):
        db_root = osp.join(self.images_root, 'db')
        img_paths = []
        for root, _, files in os.walk(db_root):
            files.sort()
            for filename in files:
                if filename.endswith(self.img_ext):
                    imgpath = os.path.join(root, filename)
                    if os.path.isfile(imgpath):
                        img_paths.append(imgpath)
                    else:
                        print('cannot find the image:', imgpath)
        # print('Found {} images in the database root {}'.format(len(img_paths), db_root))
        return img_paths

    def _get_query_imgs(self):
        query_root = osp.join(self.root, 'queries')
        if self.slice == 'day_time':
            query_file = osp.join(query_root, 'day_time_queries_with_intrinsics.txt')
        elif self.slice == 'night_time':
            query_file = osp.join(query_root, 'night_time_queries_with_intrinsics.txt')
        else:
            raise NotImplementedError

        query_list = []
        qf = open(query_file, 'r')

        for line in qf.readlines():  # 依次读取每行
            line = line.strip()  # 去掉每行头尾空白
            if not len(line) or line.startswith('#'):  # 判断是否是空行或注释行
                continue  # 是的话，跳过不处理
            tmp_path = osp.join(self.images_root, line.split(' ')[0])
            query_list.append(tmp_path)  # 保存
        return query_list

    def get_image_name(self, path):
        return path[len(self.images_root)+1:]


if __name__ == '__main__':
    aachen_db = AAcchenDB(root='/data/vldata/aachen', slice='day_time')

    print(aachen_db.db_imgs)
    print(aachen_db.query_imgs)

    print(aachen_db.get_image_name(aachen_db.db_imgs[0]))
    print(aachen_db.get_image_name(aachen_db.query_imgs[0]))
