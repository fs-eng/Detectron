#!/usr/local/bin/python3

import re
import sys
import cv2
import os
import json
import xml.etree.ElementTree as etree
from datetime import datetime
from shapely.geometry import Polygon

_category_map = {
    'handwritten-line': 1,
    'machine-print-line': 2,
    'separator': 3,
    'line-drawing': 4,
    'graphic': 5
}

def new_json():
    data = {}

    info = {
        'description': 'FamilySearch Line Segmentation Dataset',
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
    for (k,v) in _category_map.items():
        category = {
            'supercategory': 'line',
            'id': v,
            'name': k
        }
        categories.append(category)

    data['categories'] = categories
    return data

def poly2points(poly):
    seg = []
    for (x,y) in list(poly.exterior.coords):
        seg.append(x)
        seg.append(y)
    return seg

def convert_coords(coords, scale_factor):
    points_string = coords.get('points')
    points = points_string.split(' ')

    seg = []
    parray = []
    for p in points:
        (x, y) = p.split(',')
        new_x = float(x) * scale_factor
        new_y = float(y) * scale_factor
        seg.append(new_x)
        seg.append(new_y)
        parray.append([new_x, new_y])

    try:
        poly = Polygon(parray)
    except ValueError:
        print('error with coords: {}'.format(points_string))
        return [[]], [], 0.0

    poly = poly.simplify(0.01, preserve_topology=True)
    (xmin, ymin, xmax, ymax) = poly.bounds
    area = float(poly.area)

    x0 = float(xmin)
    w = float(xmax-xmin)
    y0 = float(ymin)
    h = float(ymax-ymin)

    segmentation = [poly2points(poly)]
    bbox = [x0, y0, w, h]

    return segmentation, bbox, area


def main():

    if len(sys.argv) != 5:
        print('usage: page2coco.py input-page-xml-directory input-image-directory output-coco-json-file output-image-directory')
        sys.exit(-1)
    
    input_xml_directory = sys.argv[1].rstrip('/')
    input_img_directory = sys.argv[2].rstrip('/')
    output_file = sys.argv[3]
    output_img_directory = sys.argv[4].rstrip('/')
    data = new_json()
    next_annotation_id = 1
    next_image_id = 1

    file_listing = sorted(os.listdir(input_xml_directory))
    for f in file_listing:
        filename = os.fsdecode(f)
        if filename.endswith('.xml'):
            print('working on {}'.format(filename))

            tree = etree.parse(input_xml_directory + '/' + filename)
            root = tree.getroot()
            ns = {'a': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15'}

            page = root.find('a:Page', ns)

            if page == None:
                ns = {'a': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15'}
                page = root.find('a:Page', ns)

            if page == None:
                ns = {'a': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2016-07-15'}
                page = root.find('a:Page', ns)

            if page == None:
                ns = {'a': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
                page = root.find('a:Page', ns)

            multiline = False
            for unicode in page.findall('.//a:Unicode', ns):
                if unicode.text is not None:
                    if '\n' in unicode.text:
                        multiline = True
                        break

            if multiline:
                print('Multiline line regions... skipping file.')
                continue

            height = int(page.attrib['imageHeight'])
            width = int(page.attrib['imageWidth'])
            image_filename = page.attrib['imageFilename']
            img = cv2.imread(input_img_directory + '/' + image_filename)
            if img is None:
                print('Cannot find image {}.  Skipping file.'.format(input_img_directory + '/' + image_filename))
                continue

            new_image_filename = image_filename[:-4] + '.jpg'

            img_h, img_w, img_c = img.shape

            if img_h < img_w:
                if img_h > 800:
                    scale_factor = 800.0 / float(img_h)
            else:
                if img_w > 800:
                    scale_factor = 800.0 / float(img_w)
                else:
                    scale_factor = 1.0

            try:
                assert(height == img_h)
                assert(width == img_w)
            except AssertionError:
                print('{}: dimension mismatch between XML and image ({}x{} vs {}x{}).  Skipping.'.format(filename,height,width,img_h,img_w))
                continue

            img = cv2.resize(img, (round(float(img_w) * scale_factor), round(float(img_h) * scale_factor)))
            cv2.imwrite(output_img_directory + '/' + new_image_filename, img)

            image_id = next_image_id
            next_image_id += 1

            m = re.match('([0-9]{9})_([0-9]{5})',filename)
            if m is not None:
                dgs_str = m.group(1)
                img_num_str = m.group(2)
                image_url = 'https://das.familysearch.org/das/v2/dgs:' + dgs_str + '.' + dgs_str + '_' + img_num_str + '/$dist'
            else:
                image_url = 'https://www.familysearch.org/'

            image = {
                'license': 1,
                'file_name': new_image_filename,
                'coco_url': image_url,
                'height': round(float(height) * scale_factor),
                'width': round(float(width) * scale_factor),
                'date_captured': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'flickr_url': image_url,
                'id': image_id
            }

            data['images'].append(image)


            #old style format using TextRegion to annotate lines, images, etc.
            for tr in page.findall('.//a:TextRegion', ns):
                te = tr.find('a:TextEquiv', ns)
                unicode = tr.find('a:Unicode', ns)
                if unicode is not None or te is not None:
                    try:
                        tr_type = tr.attrib['type']
                    except KeyError:
                        tr_type = 'null'
                    if tr_type == 'text2':
                        category_id = _category_map['handwritten-line']
                    elif tr_type == 'text1':
                        category_id = _category_map['machine-print-line']
                    elif tr_type == 'text1n2':
                        #TODO: is this the best mapping for mixed print?
                        category_id = _category_map['machine-print-line']
                    elif tr_type == 'separator':
                        category_id = _category_map['separator']
                    elif tr_type == 'graphic':
                        category_id = _category_map['graphic']
                    elif tr_type == 'paragraph':
                        category_id = _category_map['machine-print-line']
                    else:
                        continue #ignore marginalia

                    coords = tr.find('a:Coords', ns)
                    (segmentation, bbox, area) = convert_coords(coords, scale_factor)
                    if area == 0.0:
                        print('bad polygon')
                        continue #bad polygon TODO: what to do with these?
                    annotation = {
                        'segmentation': segmentation,
                        'area': area,
                        'iscrowd': 0,
                        'image_id': image_id,
                        'bbox': bbox,
                        'category_id': category_id,
                        'id': next_annotation_id
                    }
                    next_annotation_id += 1
                    data['annotations'].append(annotation)

            for tl in page.findall('.//a:TextLine', ns):
                try:
                    production = tl.attrib['production']
                except KeyError:
                    production = 'handwritten-cursive'

                if production == 'handwritten-cursive' or production == 'handwritten-printscript' or production == 'medieval-manuscript':
                    category_id = _category_map['handwritten-line']
                elif production == 'printed' or production == 'typewritten':
                    category_id = _category_map['machine-print-line']
                else:
                    print('production:{}'.format(production))
                    continue

                coords = tl.find('a:Coords', ns)
                (segmentation, bbox, area) = convert_coords(coords, scale_factor)

                if area == 0.0:
                    print('bad polygon')
                    continue #bad polygon TODO: what to do with these?

                annotation = {
                    'segmentation': segmentation,
                    'area': area,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': bbox,
                    'category_id': category_id,
                    'id': next_annotation_id
                }
                next_annotation_id += 1
                data['annotations'].append(annotation)

            for sr in page.findall('.//a:SeparatorRegion', ns):
                coords = sr.find('a:Coords', ns)
                category_id = _category_map['separator']
                (segmentation, bbox, area) = convert_coords(coords, scale_factor)
                if area == 0.0:
                    print('bad polygon')
                    continue #bad polygon TODO: what to do with these?
                annotation = {
                    'segmentation': segmentation,
                    'area': area,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': bbox,
                    'category_id': category_id,
                    'id': next_annotation_id
                }
                next_annotation_id += 1
                data['annotations'].append(annotation)

            for ld in page.findall('.//a:LineDrawingRegion', ns):
                coords = ld.find('a:Coords', ns)
                category_id = _category_map['line-drawing']
                (segmentation, bbox, area) = convert_coords(coords, scale_factor)
                if area == 0.0:
                    print('bad polygon')
                    continue #bad polygon TODO: what to do with these?
                annotation = {
                    'segmentation': segmentation,
                    'area': area,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': bbox,
                    'category_id': category_id,
                    'id': next_annotation_id
                }
                next_annotation_id += 1
                data['annotations'].append(annotation)

            for gr in page.findall('.//a:GraphicRegion', ns):
                coords = gr.find('a:Coords', ns)
                category_id = _category_map['graphic']
                (segmentation, bbox, area) = convert_coords(coords, scale_factor)
                if area == 0.0:
                    print('bad polygon')
                    continue #bad polygon
                annotation = {
                    'segmentation': segmentation,
                    'area': area,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': bbox,
                    'category_id': category_id,
                    'id': next_annotation_id
                }
                next_annotation_id += 1
                data['annotations'].append(annotation)

            for ir in page.findall('.//a:ImageRegion', ns):
                coords = ir.find('a:Coords', ns)
                category_id = _category_map['graphic']
                (segmentation, bbox, area) = convert_coords(coords, scale_factor)
                if area == 0.0:
                    print('bad polygon')
                    continue #bad polygon TODO: what to do with these?
                annotation = {
                    'segmentation': segmentation,
                    'area': area,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': bbox,
                    'category_id': category_id,
                    'id': next_annotation_id
                }
                next_annotation_id += 1
                data['annotations'].append(annotation)

                #TODO: handle TableRegion?

    with open(output_file, 'w') as outfile:
        json.dump(data,outfile)


if __name__ == "__main__": main()

