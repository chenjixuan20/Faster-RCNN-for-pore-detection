#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__', 'pore')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_20000.ckpt',),'res101': ('res101_faster_rcnn_iter_20000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def batch_detect(load_path, save_path, sess, net):
    images, names = load_images_with_names(load_path)  # 需要写一个函数来加载图片！！！！！

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image, name in zip(images, names):
        scores, boxes = im_detect(sess, net, image)
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, FLAGS.NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] > FLAGS.CONF_THRESH)[0]
            dets = dets[inds, :]
            filename = os.path.join(save_path, '{}.txt'.format(name))
            _write_voc_results_file_with_image(filename, dets, cls)


def load_images(folder_path):
  images = []
  for image_path in sorted(os.listdir(folder_path)):
    if image_path.endswith(('.jpg', '.png', '.bmp')):
        images.append(cv2.imread(os.path.join(folder_path, image_path)))
  return images

def load_images_with_names(images_dir):
    images = load_images(images_dir)
    image_names = [
        path.split('.')[0] for path in sorted(os.listdir(images_dir))
        if path.endswith(('.jpg', '.bmp', '.png'))
    ]
    return images, image_names

def _write_voc_results_file_with_image(filename, dets, cls):
    print('Writing {} VOC results file'.format(filename))
    with open(filename, 'wt') as f:
        if dets == []:
            print("No dets.")
            return
        widths = dets[:, 2] - dets[:, 0] + 1.0  # 宽度
        heights = dets[:, 3] - dets[:, 1] + 1.0  # 高度
        ctr_x = dets[:, 0] + 0.5 * widths  # 中心坐标
        ctr_y = dets[:, 1] + 0.5 * heights
        for k in range(dets.shape[0]):
          print(int(ctr_y[k] + 1), int(ctr_x[k] + 1), file=f)
          # print('\n', file=f)

        # with open(filename, 'w') as f:
        #     for coord in dets:
        #         print(coord[0] + 1, coord[1] + 1, file=f)



def parse_args():
        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
        parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                            choices=NETS.keys(), default='vgg16')
        parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                            choices=DATASETS.keys(), default='pascal_voc')
        parser.add_argument(
            '--polyu_dir_path',
            # required=True,
            type=str,
            default='polyu_hrf',
            help='path to PolyU-HRF dataset')
        parser.add_argument(
            '--results_dir_path',
            type=str,
            default='log/pores',
            help='path to folder in which results should be saved')
        parser.add_argument(
            '--CONF_THRESH',
            type=float,
            default=0.3,
            help='probability threshold to filter detections')
        parser.add_argument(
            '--NMS_THRESH',
            type=float,
            default=0.2,
            help='nms intersection threshold')
        args = parser.parse_args()

        return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.BBOX_REG = True  # Test using bounding-box regressors
    FLAGS = parse_args()

    # model path

    demonet = FLAGS.demo_net
    dataset = FLAGS.dataset
    tfmodel = os.path.join('output', 'vgg16_scale_1_2_4_tb', DATASETS[dataset][0], 'default',  # 改！！！！
                           NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError

    print('Building graph...')
    net.create_architecture("TEST", 2,
                            tag='default', anchor_scales=[1, 2, 4])  # 改！！！！！！
    print('Done')

    print('Restoring model in {}...'.format(tfmodel))
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('Loaded network {:s}'.format(tfmodel))

    # # test
    # print('Detecting pores in PolyU-HRF DBI Training images...')
    # load_path = os.path.join(FLAGS.polyu_dir_path)
    # save_path = os.path.join(FLAGS.results_dir_path)
    # batch_detect(load_path, save_path, sess, net)
    # print('Done')

    # batch detect in dbi training
    print('Detecting pores in PolyU-HRF DBI Training images...')
    load_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Training')
    save_path = os.path.join(FLAGS.results_dir_path, 'DBI', 'Training')
    batch_detect(load_path, save_path, sess, net)
    print('Done')

    # batch detect in dbi test
    print('Detecting pores in PolyU-HRF DBI Test images...')
    load_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Test')
    save_path = os.path.join(FLAGS.results_dir_path, 'DBI', 'Test')
    batch_detect(load_path, save_path,sess,net)
    print('Done')

    # batch detect in dbii
    print('Detecting pores in PolyU-HRF DBII images...')
    load_path = os.path.join(FLAGS.polyu_dir_path, 'DBII')
    save_path = os.path.join(FLAGS.results_dir_path, 'DBII')
    batch_detect(load_path, save_path,sess,net)
    print('Done')



# python3 -m batch_detect_pores --polyu_dir_path polyu_hrf  --results_dir_path log/pores