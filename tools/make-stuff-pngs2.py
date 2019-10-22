#!/usr/local/bin/python3

import sys
import argparse
import cv2
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Create COCO 2017 "stuff segmentation" PNG files from a COCO JSON file')
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

    image_dir = args.image_dir.rstrip('/')
    output_dir = args.output_dir.rstrip('/')

    file_listing = sorted(os.listdir(image_dir))
    for f in file_listing:
        filename = os.fsdecode(f)
        if filename.endswith('.png'):
            print('working on {}'.format(filename))
            img = cv2.imread(image_dir + '/' + filename,0)
            (h, w) = img.shape
            values = {}
            for y in range(0, h):
                for x in range(0, w):
                    values[img[y, x]] = True

            print('values: {}'.format(values.keys()))

            outfile = output_dir + '/' + filename[:-4] + '.png'
            cv2.imwrite(outfile, img)


if __name__ == '__main__':
    args = parse_args()
    main(args)

