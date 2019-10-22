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

from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import nearest_points
from shapely.ops import unary_union

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils

import pycocotools.mask as mask_util
from skimage import measure

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

MAX_DIMENSION_SHORT_SIDE = 800


def parse_args():
    parser = argparse.ArgumentParser(description='FamilySearch Mask-R-CNN-based Line Segmenter')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (default: /home/ubuntu/work/detectron/configs/fs/lineseg-e2e_mask_rcnn-R-101-FPN-2x.yaml)',
        default='/home/ubuntu/work/detectron/configs/fs/lineseg-e2e_mask_rcnn-R-101-FPN-2x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (default: /home/ubuntu/work/detectron/models/maskrcnn-lineseg-640k.pkl)',
        default='/home/ubuntu/work/detectron/models/maskrcnn-lineseg-640k.pkl',
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
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)

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
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(model, im, None, timers=timers)
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
