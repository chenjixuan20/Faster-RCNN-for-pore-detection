# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import math

from utils.timer import Timer
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from model.nms_wrapper import nms
from datasets.voc_eval import parse_rec


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)

    return blobs, im_scale_factors


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""
    for i in range(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, im):
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

    _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])

    boxes = rois[:, 1:5] / im_scales[0]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    # return scores, pred_boxes,one_anchor
    return scores, pred_boxes


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1))[0]
            dets = dets[inds, :]
            if dets == []:
                continue

            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net1(sess, net, imdb, weights_filename):
    np.random.seed(cfg.RNG_SEED)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)

    output_dir = get_output_dir(imdb, weights_filename)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    # 加载GT
    annopath = os.path.join(
        imdb._devkit_path,
        'VOC' + imdb._year, "Annotations", '{:s}.xml')
    cachedir = os.path.join(imdb._devkit_path, 'annotations_cache')
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    imagesetfile = os.path.join(
        imdb._devkit_path,
        'VOC' + imdb._year,
        'ImageSets',
        'Main',
        imdb._image_set + '.txt')
    cachefile = os.path.join(cachedir, 'test_annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annotations
        recs = {}
        for i, imagename in enumerate(imagenames):
            print('image——name',imagename)
            recs[imagename] = parse_rec(annopath.format(imagename))
            # if i % 100 == 0:
            print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')

    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    for i in range(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        _t['im_detect'].tic()
        scores, boxes = im_detect(sess, net, im)
        _t['im_detect'].toc()
        print(scores[1])
        #print(len(boxes))
        print('im_detect: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time))

        # skip j = 0, because it's the background class
        for j in range(1, imdb.num_classes):
            cls_scores = scores[:, j]
            cls_boxes = boxes[:, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

    # validate over thresholds
    print('Evaluating detections')
    _t['misc'].tic()
    inter_thrs = np.arange(0.7, 0, -0.1)
    prob_thrs = np.arange(0, 0.7, 0.1)
    best_f_score = 0
    best_tdr = None
    best_fdr = None
    best_inter_thr = None
    best_prob_thr = None

    # put inference in nms proper format
    for prob_thr in prob_thrs:
        probs = []
        for i in range(num_images):
            prob_tmp=[[]]
            for j in range(1, imdb.num_classes):
                img_preds = all_boxes[j][i]
                pick = (img_preds[:, -1] > prob_thr)
                prob_tmp.append(img_preds[pick])
            probs.append(prob_tmp)
        for inter_thr in inter_thrs:
            total_pores = 0
            total_dets = 0
            true_dets = 0
            dets=[]
            for i, imagename in enumerate(imagenames):
                det_tmp = [[]]
                for j, cls in enumerate(imdb.classes):
                    if cls == '__background__':
                        continue
                    img_preds = probs[i][j]
                    keep = nms(img_preds, cfg.TEST.NMS)
                    img_dets=img_preds[keep]
                    img_det_ctr=trans_ctr(img_dets[:, :-1])
                    det_tmp.append(img_det_ctr)
                    total_dets += len(img_det_ctr)

                    R = [obj for obj in recs[imagename] if obj['name'] == cls]
                    bbox = np.array([x['bbox'] for x in R])
                    bbox_ctr = trans_ctr(bbox)
                    total_pores += len(bbox_ctr)
                    print('AAAAA', len(img_det_ctr), prob_thr)
                    print('BBBBB', len(bbox_ctr), inter_thr)

                    true_dets += len(find_correspondences(img_det_ctr, bbox_ctr))
                dets.append(det_tmp)

            # compute tdr, fdr and f score
            eps = 1e-5
            tdr = true_dets / (total_pores + eps)
            fdr = (total_dets - true_dets) / (total_dets + eps)
            f_score = 2 * (tdr * (1 - fdr)) / (tdr + (1 - fdr))

            # update best parameters
            if f_score > best_f_score:
                best_f_score = f_score
                best_tdr = tdr
                best_fdr = fdr
                best_inter_thr = inter_thr
                best_prob_thr = prob_thr

    _t['misc'].toc()
    print(('best F_SCORE = {:.4f}\n best TDR ={:.4f} \n best FDR = {:.4f}\n '
           'best NMS thr = {:2f}\n best CONF thr = {:2f}'.
           format(best_f_score, best_tdr, best_fdr, best_inter_thr, best_prob_thr)))
    print('{:.3f}s'.format(_t['misc'].average_time))


def trans_ctr(cls_boxes):
    widths = cls_boxes[:, 2] - cls_boxes[:, 0] + 1.0
    heights = cls_boxes[:, 3] - cls_boxes[:, 1] + 1.0
    ctr_x =cls_boxes[:, 0] + 0.5 * widths
    ctr_y = cls_boxes[:, 1] + 0.5 * heights
    ctr_x = np.expand_dims(ctr_x, 1)
    ctr_y = np.expand_dims(ctr_y, 1)
    cls_dets = np.array(np.hstack((ctr_x, ctr_y)).astype(np.float32, copy=False))
    return cls_dets

def find_correspondences(descs1,
                         descs2,
                         pts1=None,
                         pts2=None,
                         euclidean_weight=0,
                         transf=None,
                         thr=None):
  '''
  Finds bidirectional correspondences between descs1 descriptors and
  descs2 descriptors. If thr is provided, discards correspondences
  that fail a distance ratio check with threshold thr. If pts1, pts2,
  and transf are give, the metric considered when finding correspondences
  is
    d(i, j) = ||descs1(j) - descs2(j)||^2 + euclidean_weight *
      * ||transf(pts1(i)) - pts2(j)||^2
  Args:
    descs1: [N, M] np.array of N descriptors of dimension M each.
    descs2: [N, M] np.array of N descriptors of dimension M each.
    pts1: [N, 2] np.array of N coordinates for each descriptor in descs1.
    pts2: [N, 2] np.array of N coordinates for each descriptor in descs2.
    euclidean_weight: weight given to spatial constraint in comparison
      metric.
    transf: alignment transformation that aligns pts1 to pts2.
    thr: distance ratio check threshold.
  Returns:
    list of correspondence tuples (j, i, d) in which index j of
      descs2 corresponds with i of descs1 with distance d.
  '''
  # compute descriptors' pairwise distances
  D = pairwise_distances(descs1, descs2)

  # add points' euclidean distance
  if euclidean_weight != 0:
    assert transf is not None
    assert pts1 is not None
    assert pts2 is not None

    # assure pts are np array
    pts1 = transf(np.array(pts1))
    pts2 = np.array(pts2)

    # compute points' pairwise distances
    euclidean_D = pairwise_distances(pts1, pts2)

    # add to overral keypoints distance
    D += euclidean_weight * euclidean_D

  # find bidirectional corresponding points
  pairs = []
  if thr is None or len(descs1) == 1 or len(descs2) == 1:
    # find the best correspondence of each element
    # in 'descs2' to an element in 'descs1'
    corrs2 = np.argmin(D, axis=0)


    # find the best correspondence of each element
    # in 'descs1' to an element in 'descs2'
    corrs1 = np.argmin(D, axis=1)

    # keep only bidirectional correspondences
    for i, j in enumerate(corrs2):
      if corrs1[j] == i:
        pairs.append((j, i, D[j, i]))
  else:
    # find the 2 best correspondences of each
    # element in 'descs2' to an element in 'descs1'
    corrs2 = np.argpartition(D.T, [0, 1])[:, :2]

    # find the 2 best correspondences of each
    # element in 'descs1' to an element in 'descs2'
    corrs1 = np.argpartition(D, [0, 1])[:, :2]

    # find bidirectional corresponding points
    # with second best correspondence 'thr'
    # worse than best one
    for i, (j, _) in enumerate(corrs2):
      if corrs1[j, 0] == i:
        # discard close best second correspondences
        if D[j, i] < D[corrs2[i, 1], i] * thr:
          if D[j, i] < D[j, corrs1[j, 1]] * thr:
            pairs.append((j, i, D[j, i]))

  return pairs


def pairwise_distances(x1, x2):
  # memory efficient implementation based on Yaroslav Bulatov's answer in
  # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
  sqr1 = np.sum(x1 * x1, axis=1, keepdims=True)
  sqr2 = np.sum(x2 * x2, axis=1)
  D = sqr1 - 2 * np.matmul(x1, x2.T) + sqr2

  return D