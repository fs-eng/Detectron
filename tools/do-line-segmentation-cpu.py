#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2
import glob
import logging
import os
import sys
import time
import numpy as np
import json
import re
from datetime import datetime
import xml.etree.ElementTree as etree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom

import pdb

from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import nearest_points
from shapely.ops import unary_union

from caffe2.python import workspace
from caffe2.python import dyndep
from caffe2.proto import caffe2_pb2
from caffe2.python import core

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.timer import Timer
import detectron.utils.c2 as c2_utils

import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
import detectron.modeling.FPN as fpn

import pycocotools.mask as mask_util
from skimage import measure

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

MAX_DIMENSION_SHORT_SIDE = 800

def get_device_option_cpu():
    device_option = core.DeviceOption(caffe2_pb2.CPU)
    return device_option

def import_detectron_ops():
    detectron_ops_lib = '/home/ubuntu/anaconda3/envs/pytorch_p27/lib/python2.7/site-packages/torch/lib/libcaffe2_detectron_ops_gpu.so'
    dyndep.InitOpsLibrary(detectron_ops_lib)


def parse_args():
    parser = argparse.ArgumentParser(description='FamilySearch Mask-R-CNN-based Line Segmenter')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (default: /home/ubuntu/work/detectron/configs/fs/lineseg-6class.yaml)',
        default='/home/ubuntu/work/detectron/configs/fs/lineseg-6class.yaml',
        type=str
    )
    parser.add_argument(
        '--model-dir',
        dest='model_dir',
        help='directory containing prebuilt CPU model files (default: /home/ubuntu/work/detectron/model/)',
        default='/home/ubuntu/work/detectron/model/',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for keeping detections (default: 0.5)',
        default=0.9,
        type=float
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for output (default: /tmp/line_segmentation)',
        default='/tmp/line_segmentation',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--output-type',
        dest='output_type',
        help='Output type (page-xml or coco-json.  default: page-xml)',
        default='page-xml',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


_category_map = {
    'handwritten-line': 1,
    'machine-print-line': 2,
    'separator': 3,
    'line-drawing': 4,
    'graphic': 5
}

_r_category_map = {
    1: 'handwritten-cursive',
    2: 'printed',
    3: 'separator',
    4: 'line-drawing',
    5: 'graphic'
}


def setup_logging(name):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    logging.root.handlers = []
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(name)
    return logger


def prettify(elem):
    rough_string = etree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent='    ')


def new_xml():
    root = Element('top')
    root.tag = 'PcGts'
    root.set('xmlns', 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:schemaLocation',
             'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15/pagecontent.xsd')
    metadata = SubElement(root, 'Metadata')
    creator = SubElement(metadata, 'Creator')
    creator.text = 'FamilySearch International'
    created = SubElement(metadata, 'Created')
    created.text = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    last_change = SubElement(metadata, 'LastChange')
    last_change.text = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    return root


def new_json():
    data = {}

    info = {
        'description': 'FamilySearch Line Segmentation',
        'url': 'http://www.familysearch.org',
        'year': int(datetime.now().strftime('%Y')),
        'contributor': 'FamilySearch International',
        'date_created': datetime.now().strftime('%Y/%m/%d')
    }

    data['info'] = info

    licenses = [
        {
            'url': 'http://www.familysearch.org',
            'id': 1,
            'name': 'Proprietary Data. All Rights Reserved, FamilySearch International, 2019'
        }
    ]

    data['licenses'] = licenses
    data['images'] = []
    data['annotations'] = []

    categories = []
    for (k, v) in _category_map.items():
        category = {
            'supercategory': 'line',
            'id': v,
            'name': k
        }
        categories.append(category)

    data['categories'] = categories
    return data


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _get_blobs(im, rois, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils.get_image_blob(im, target_scale, target_max_size)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    return blobs, im_scale


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn.map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    fpn.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max
    )


def box_results_with_nms_and_limit(scores, boxes):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def im_detect_bbox(workspace, predict_net, im, target_scale, target_max_size, boxes=None):
    """Bounding box object detection for an image with given box proposals.

    Arguments:
        workspace: the caffe2 workspace to use
        predict_net: the prediction network
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals in 0-indexed
            [x1, y1, x2, y2] format, or None if using RPN

    Returns:
        scores (ndarray): R x K array of object class scores for K classes
            (K includes background as object category 0)
        boxes (ndarray): R x 4*K array of predicted bounding boxes
        im_scales (list): list of image scales used in the input blob (as
            returned by _get_blobs and for use with im_detect_mask, etc.)
    """
    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v, get_device_option_cpu())
    workspace.RunNet(predict_net.name)

    # Read out blobs
    #rois = workspace.FetchBlob(core.ScopedName('rois'))
    rois = workspace.FetchBlob(core.ScopedName('rpn_rois'))

    # unscale back to raw image space
    boxes = rois[:, 1:5] / im_scale

    # Softmax class probabilities
    scores = workspace.FetchBlob(core.ScopedName('cls_prob')).squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    # Apply bounding-box regression deltas
    box_deltas = workspace.FetchBlob(core.ScopedName('bbox_pred')).squeeze()
    # In case there is 1 proposal
    box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])

    pred_boxes = box_utils.bbox_transform(
        boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS
    )
    pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))

    return scores, pred_boxes, im_scale


def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                (x_0 - ref_box[0]):(x_1 - ref_box[0])
            ]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms


def im_detect_mask(workspace, predict_net, im_scale, boxes):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        workspace: the caffe2 workspace to use
        predict_net: the prediction network
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, im_scale)}

    _add_multilevel_rois_for_test(inputs, 'mask_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v, get_device_option_cpu())
    workspace.RunNet(predict_net.name)

    pdb.set_trace()

    # Fetch masks TODO: masks are missing from the PB for some reason.
    pred_masks = workspace.FetchBlob(
        core.ScopedName('mask_fcn_probs')
    ).squeeze()

    pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])

    return pred_masks


def im_detect_all(workspace, predict_net, im, box_proposals, timers=None):
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    scores, boxes, im_scale = im_detect_bbox(
        workspace, predict_net, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals
    )
    timers['im_detect_bbox'].toc()

    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    timers['misc_bbox'].toc()

    '''if boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        masks = im_detect_mask(workspace, predict_net, im_scale, boxes)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(
            cls_boxes, masks, boxes, im.shape[0], im.shape[1]
        )
        timers['misc_mask'].toc()
    else:
        cls_segms = None'''

    cls_segms = None
    cls_keyps = None

    return cls_boxes, cls_segms, cls_keyps


def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes


def make_bboxes(boxes):
    if boxes is None:
        return []
    bboxes = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        x0, y0, w, h = (int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))
        bboxes.append(([x0, y0, w, h], score))
    return bboxes

def convert_to_xml_coords(segmentation):
    if len(segmentation) == 0:
        return ''
    new_segmentation = fuse_segmentation(segmentation)
    # new_segmentation = segmentation[0]
    coord_string = ''

    i = 0

    while i < len(new_segmentation):
        x = new_segmentation[i]
        y = new_segmentation[i + 1]
        if i == 0:
            coord_string += '{},{}'.format(int(x), int(y))
        else:
            coord_string += ' {},{}'.format(int(x), int(y))
        i += 2
    return coord_string


def poly2points(poly):
    seg = []
    for (x,y) in list(poly.exterior.coords):
        seg.append(x)
        seg.append(y)
    return seg


def points2poly(pts):
    pts2 = []
    assert len(pts) % 2 == 0
    i = 0
    while i < len(pts):
        x = pts[i]
        y = pts[i + 1]
        pts2.append((x, y))
        i += 2
    poly = Polygon(pts2)
    return poly


def fuse_segmentation(segmentation):
    # TODO: rather than fuse along a line between the two closet points, the better thing would be to connect along darker pixels between the two shapes

    if len(segmentation) == 1:
        return segmentation[0]
    else:
        logging.debug('Fusing {} segments'.format(len(segmentation)))
        polys = []

        for pts in segmentation:
            poly = points2poly(pts)
            polys.append(poly)

        new_poly = polys[0]

        polys.sort(key=lambda x: x.area, reverse=True)

        for p in polys:
            logging.debug('{}'.format(p.area))

        for (i, poly) in enumerate(polys):
            area_before = new_poly.area
            (pt1, pt2) = nearest_points(new_poly, poly)
            bridge = LineString([pt1, pt2]).buffer(4.0)
            geoms = [new_poly, poly, bridge]
            new_poly = unary_union(geoms)
            logging.debug(
                'Combined area should be at least {}.  It is {}'.format(area_before + poly.area, new_poly.area))

        s_pts = list(new_poly.exterior.coords)
        points = []

        for (x, y) in s_pts:
            points.append(x)
            points.append(y)

        return points


def calc_area(segmentation):
    area = 0.0
    for pts in segmentation:
        poly = points2poly(pts)
        area += poly.area
    return area


def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg(cache_urls=False)

    import_detectron_ops()
    init_net = caffe2_pb2.NetDef()
    predict_net = caffe2_pb2.NetDef()
    with open(os.path.join(args.model_dir, "model_init.pb"), 'rb') as f:
        init_net.ParseFromString(f.read())
    with open(os.path.join(args.model_dir, "model.pb"), 'rb') as f:
        predict_net.ParseFromString(f.read())
    workspace.ResetWorkspace()
    workspace.RunNetOnce(init_net)
    for op in predict_net.op:
        for blob_in in op.input:
            if not workspace.HasBlob(blob_in):
                workspace.CreateBlob(blob_in)
    logger.info('Operators Are Loaded')
    workspace.CreateNet(predict_net)
    logger.info('Predictor Net Created')

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    #model = infer_engine.initialize_model_from_cfg(args.weights)

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for i, im_name in enumerate(im_list):

        image_file_name = os.path.basename(im_name)

        if args.output_type == 'coco-json':
            ext = 'json'
        else:
            ext = 'xml'

        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.splitext(image_file_name)[0] + '.' + ext)
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        im_h, im_w, img_c = im.shape
        scale_factor = 1.0
        if im_h < im_w:
            if im_h > MAX_DIMENSION_SHORT_SIDE:
                scale_factor = float(MAX_DIMENSION_SHORT_SIDE) / float(im_h)
        else:
            if im_w > MAX_DIMENSION_SHORT_SIDE:
                scale_factor = float(MAX_DIMENSION_SHORT_SIDE) / float(im_w)

        if scale_factor != 1.0:
            im = cv2.resize(im, (int(round(float(im_w) * scale_factor)), int(round(float(im_h) * scale_factor))))

        timers = defaultdict(Timer)
        t = time.time()
        cls_boxes, cls_segms, cls_keyps = im_detect_all(workspace, predict_net, im, None, timers=timers)
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        if isinstance(cls_boxes, list):

            (boxes, segms, keyps, classes) = convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)

            m = re.match('([0-9]{9})_([0-9]{5})', image_file_name)
            dgs_str = m.group(1)
            img_num_str = m.group(2)

            image_url = 'https://das.familysearch.org/das/v2/dgs:' + dgs_str + '.' + dgs_str + '_' + img_num_str + '/$dist'

            bboxes = make_bboxes(boxes)
            logger.info('{} lines found'.format(len(bboxes)))

            if args.output_type == 'coco-json':
                data = new_json()

                image = {
                    'license': 1,
                    'file_name': image_file_name,
                    'coco_url': image_url,
                    'height': im_h,
                    'width': im_w,
                    'date_captured': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'flickr_url': image_url,
                    'id': 1
                }

                data['images'].append(image)
            else:
                data = new_xml()
                page = SubElement(data, 'Page')
                page.set('imageFilename', image_file_name)
                page.set('imageWidth', str(im_w))
                page.set('imageHeight', str(im_h))
                tr = SubElement(page, 'TextRegion')
                tr.set('id', 'region0')
                coords = SubElement(tr, 'Coords')
                coords.set('points', '0,0 {},0 {},{} 0,{}'.format(im_w, im_w, im_h, im_h))

            next_annotation_id = 1

            next_line_id = 1
            next_sep_id = 1
            next_ld_id = 1
            next_gra_id = 1

            j = 0
            if segms is not None:
                for segm in segms:
                    mask = mask_util.decode(segm)
                    contours = measure.find_contours(mask, 0.5)
                    segmentation = []
                    (bbox, score1) = bboxes[j]
                    score2 = boxes[j, -1]
                    logging.debug('score1: {} score2: {}'.format(score1, score2))

                    if score2 >= args.thresh:
                        bbox = [x / scale_factor for x in bbox]

                        for contour in contours:
                            contour = np.flip(contour, axis=1)
                            seg = contour.ravel().tolist()
                            seg = [x / scale_factor for x in seg]
                            segmentation.append(seg)
                        if args.output_type == 'coco-json':
                            area = calc_area(segmentation)
                            annotation = {
                                'segmentation': segmentation,
                                'score': float(score2),
                                'area': area,
                                'iscrowd': 0,
                                'image_id': 1,
                                'bbox': bbox,
                                'category_id': classes[j],
                                'id': next_annotation_id
                            }
                            data['annotations'].append(annotation)
                            next_annotation_id += 1
                        else:
                            if _r_category_map[classes[j]] == 'handwritten-cursive' or _r_category_map[classes[j]] == 'printed':
                                elem = SubElement(tr, 'TextLine')
                                elem.set('production', _r_category_map[classes[j]], )
                                elem.set('id', 'tl' + str(next_line_id))
                                next_line_id += 1
                            elif _r_category_map[classes[j]] == 'separator':
                                elem = SubElement(page, 'SeparatorRegion')
                                elem.set('id', 'sr' + str(next_sep_id))
                                next_sep_id += 1
                            elif _r_category_map[classes[j]] == 'line-drawing':
                                elem = SubElement(page, 'LineDrawingRegion')
                                elem.set('id', 'ldr' + str(next_ld_id))
                                next_ld_id += 1
                            else:  # graphic
                                elem = SubElement(page, 'GraphicRegion')
                                elem.set('id', 'gr' + str(next_gra_id))
                                next_gra_id += 1

                            coord_string = convert_to_xml_coords(segmentation)
                            coords = SubElement(elem, 'Coords')
                            coords.set('points', coord_string)

                    else:
                        logging.info('Not keeping line with confidence {} below threshold of {}'.format(score2, args.thresh))

                    j += 1


            with open(out_name, 'w') as outfile:
                if args.output_type == 'coco-json':
                    #pdb.set_trace()
                    json.dump(data, outfile, indent=4)
                else:
                    outfile.write(prettify(data))

        else:
            logger.info('Nothing found in image {}'.format(image_file_name))


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
