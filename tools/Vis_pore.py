
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import os, cv2

def vis_detections(im, dets=None, pores=None, im_name=None):
    """Draw detecte pores and GT pores."""
    # inds = np.where(dets[:, -1] >= thresh)[0]
    if len(dets) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    if pores:
        for i in range(len(pores)):
            ax.add_patch(
                plt.Rectangle((pores[i][1] - 3, pores[i][0] - 3), 7, 7, fill=False,
                              edgecolor='yellow', linewidth=1.5)
            )
    for i in range(len(dets)):
        #print(dets[i])
        #print(dets[i][0])
        ax.add_patch(
            plt.Rectangle((dets[i][1]-4, dets[i][0]-4), 9, 9, fill=False,
                          edgecolor='red', linewidth=1.5)
            )

    ax.set_title(('pores detections(in red）\n GT pores (in yellow） '), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def load( image_path, dets_path, image_name, pores_path=None):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(image_path, image_name+'.jpg')
    im = cv2.imread(im_file)

    dets = load_dets_txt(os.path.join( dets_path, image_name+'.txt'))
    if pores_path:
        pores = load_dets_txt(os.path.join(pores_path, image_name+'.txt'))
    else:
        pores = None

#     Visualize detections for each image
    vis_detections(im=im, dets=dets, pores=pores, im_name=image_name)

def load_dets_txt(pts_path):
  pts = []
  with open(pts_path, 'r') as f:
    for line in f:
      row, col = [int(t) for t in line.split()]
      pts.append((row - 1, col - 1))

  return pts


if __name__ == '__main__':
    image_path = '/data/VOCdevkit2007/VOC2007_new/JPEGImages'
    im_names = ['000100', '000102', '000103', '000107']
    dets_path = '/data/VOCdevkit2007/VOC2007_new/Annotations_txt'
    pores_path=None
    #pores_path = 'polyu_hrf/GroundTruth/PoreGroundTruth/PoreGroundTruthMarked'
    for im_name in im_names:
        load(image_path, dets_path, im_name, pores_path)
    plt.show()
    print('Over')