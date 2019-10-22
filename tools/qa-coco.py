#!/usr/local/bin/python3

import sys
import cv2
import json
import numpy as np
import random

def main():

    if len(sys.argv) != 4:
        print('usage: label-icdar.py coco-json-file input-image-directory label-file')
        sys.exit(-1)

    coco_json_file = sys.argv[1]
    image_directory = sys.argv[2].rstrip('/')
    label_file = sys.argv[3]


    previous_labels = {}
    try:
        with open(label_file, 'r') as label_in:
            for line in label_in:
                parts = line.split(',')
                filename = parts[0]
                label = parts[1].rstrip()
                previous_labels[filename] = label
    except FileNotFoundError:
        pass

    labels = {}
    a_map = {}
    try:
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

            images = []
            imgs = data['images']
            for img in imgs:
                filename = img['file_name']
                try:
                    previous_labels[filename]
                except KeyError:
                    images.append(img)

            total = len(images)
            random.shuffle(images)

            i = len(labels.items())
            while i < total:
                image = images[i]
                filename = image['file_name']
                image_id = image['id']

                try:
                    a_list = a_map[image_id]
                except KeyError:
                    a_list = []

                print('{}: {} of {}'.format(filename,i,total))

                img = cv2.imread(image_directory + '/' + filename)

                try:
                    height, width, channels = img.shape
                except:
                    print('{} not found! Skipping.'.format(image_directory + '/' + filename))
                    i += 1
                    continue

                overlay = img.copy()
                output = img.copy()

                for anno in a_list:

                    #draw bbox
                    '''bbox = anno['bbox']
                    (x, y, w, h) = bbox
                    x0 = x
                    y0 = y
                    x1 = x + w
                    y1 = y + h
                    bpts = [x0, y0, x1, y0, x1, y1, x0, y1]
                    bpts = np.array(bpts, np.int32)
                    bpts = bpts.reshape((-1, 1, 2))
                    #cv2.fillPoly(overlay, [bpts], (0,0,128))
                    cv2.polylines(output, [bpts], True, (128,0,128), thickness=1, lineType=8, shift=0)'''

                    if anno['category_id'] == 1:
                        color = (128, 0, 0)
                        #color = (random.randint(1, 128), random.randint(1, 128), random.randint(1, 128))
                    elif anno['category_id'] == 2:
                        color = (0, 128, 0)
                        #color = (random.randint(1, 128), random.randint(1, 128), random.randint(1, 128))
                    elif anno['category_id'] == 3:
                        color = (0, 0, 128)
                    elif anno['category_id'] == 4:
                        color = (128, 128, 0)
                    elif anno['category_id'] == 5:
                        color = (0, 128, 128)
                    else:
                        color = (128, 0, 128)

                    for pts in anno['segmentation']:
                        pts2 = np.array(pts, np.int32)
                        pts2 = pts2.reshape((-1,1,2))
                        cv2.fillPoly(overlay, [pts2], color)
                        cv2.polylines(output, [pts2], True, color, thickness=1, lineType=8, shift=0)

                alpha = 0.5
                cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
                try:
                    prev = labels[filename]
                except KeyError:
                    prev = ''


                text = '{}'.format(prev)
                cv2.putText(output,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,128),1)

                output2 = output.copy()

                if height > 1000:
                    w = int((float(width) * 1000.0) / float(height))
                    h = 1000
                    output2 = cv2.resize(output2, (w, h))

                cv2.namedWindow(filename)
                cv2.moveWindow(filename, 0,0)

                cv2.imshow(filename,output2)
                key = cv2.waitKey(0)

                if key == 44 and i != 0: #left (,)
                    i -= 1
                elif key == 46: #right (.)
                    i += 1
                elif key == 27: #quit (esc)
                    break
                else:
                    i += 1
                    labels[filename] = (chr(key))
                cv2.destroyAllWindows()

    finally:
        with open(label_file, 'w') as label_out:
            for (k, v) in previous_labels.items():
                print('{},{}'.format(k, v))
                label_out.write('{},{}\n'.format(k, v))
            for (k,v) in labels.items():
                print('{},{}'.format(k,v))
                label_out.write('{},{}\n'.format(k,v))
        print('done writing file')

if __name__ == "__main__": main()

