#!/usr/local/bin/python3

import sys
import argparse
import cv2
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Create COCO 2017 "stuff segmentation" PNG files from a COCO JSON file')
    parser.add_argument(
        '--coco-file',
        dest='coco_json_file',
        help='COCO JSON annotation file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--input-image-dir',
        dest='image_dir',
        help='COCO JSON annotation file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='Directory to which PNGs are output',
        default=None,
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main(args):

    coco_json_file = args.coco_json_file
    image_dir = args.image_dir.rstrip('/')
    output_dir = args.output_dir.rstrip('/')

    a_map = {}

    with open(coco_json_file, 'r') as json_file:
        data = json.load(json_file)

        annotations = data['annotations']
        for anno in annotations:
            image_id = anno['image_id']
            try:
                a_list = a_map[image_id]
            except KeyError:
                a_list = []
                a_map[image_id] = a_list
            a_list.append(anno)

        images = data['images']
        total = len(images)

        i = 0
        while i < total:
            image = images[i]
            filename = image['file_name']
            image_id = image['id']

            try:
                a_list = a_map[image_id]
            except KeyError:
                a_list = []

            print('{}: {} of {}'.format(filename, i, total))

            img = cv2.imread(image_dir + '/' + filename)

            try:
                height, width, channels = img.shape
            except:
                print('{} not found! Skipping.'.format(image_dir + '/' + filename))
                i += 1
                continue

            output = np.zeros((height, width, 1), np.uint8)

            for anno in a_list:

                if anno['category_id'] == 1:
                    color = 128
                else:
                    color = 64
                for pts in anno['segmentation']:
                    if isinstance(pts, list):
                        pts2 = np.array(pts, np.int32)
                        pts2 = pts2.reshape((-1, 1, 2))
                        cv2.fillPoly(output, [pts2], color)
                        cv2.polylines(output, [pts2], True, color, thickness=1, lineType=8, shift=0)
                    else:
                        print('Not a list: {}'.format(pts))

            i += 1
            outfile = output_dir + '/' + filename[:-4] + '.png'
            print('writing {}'.format(outfile))
            cv2.imwrite(outfile, output)
            print('done writing {}'.format(outfile))


if __name__ == '__main__':
    args = parse_args()
    main(args)

