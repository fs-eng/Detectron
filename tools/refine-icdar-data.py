#!/usr/local/bin/python3

import os
import re
import sys
import xml.etree.ElementTree as etree
from xml.dom import minidom

import cv2
from shapely.geometry import Polygon


def prettify(elem):
    rough_string = etree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    xmlstr = reparsed.toprettyxml(indent="  ")
    xmlstr = re.sub('ns0:','',xmlstr)
    xmlstr = re.sub('xmlns:ns0=', 'xmlns=', xmlstr)
    xmlstr = re.sub('\n *\n', '\n',xmlstr)
    xmlstr = re.sub('\n *\n', '\n',xmlstr)
    xmlstr = re.sub('\n *\n', '\n',xmlstr)
    return xmlstr

def poly2points(poly):
    seg = []
    for (x,y) in list(poly.exterior.coords):
        seg.append(x)
        seg.append(y)
    return seg

def points2rectangle(points_string):
    points = points_string.split(' ')
    xmin = 999999
    ymin = 999999
    xmax = 0
    ymax = 0
    for p in points:
        (x,y) = p.split(',')
        (x,y) = (int(x),int(y))
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
    x0 = xmin
    x1 = xmax
    y0 = ymin
    y1 = ymax
    return (x0,y0,x1,y1)


def main():
    if len(sys.argv) != 4:
        print(
            'usage: refine-icdar-data.py input-xml-directory input-image-directory output-xml-directory')
        sys.exit(-1)

    input_xml_directory = sys.argv[1].rstrip('/')
    input_img_directory = sys.argv[2].rstrip('/')
    output_xml_directory = sys.argv[3].rstrip('/')

    file_listing = sorted(os.listdir(input_xml_directory))
    for f in file_listing:
        filename = os.fsdecode(f)
        if filename.endswith('.xml'):
            print('working on {}'.format(filename))

            tree = etree.parse(input_xml_directory + '/' + filename)
            root = tree.getroot()

            ns = {'a': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15'}

            page = root.find('a:Page', ns)

            if page is None:
                print('Invalid XML file.  Skipping.')
                continue

            height = int(page.attrib['imageHeight'])
            width = int(page.attrib['imageWidth'])
            image_filename = page.attrib['imageFilename']
            img = cv2.imread(input_img_directory + '/' + image_filename, 0)
            if img is None:
                print('Cannot find image {}.  Skipping file.'.format(input_img_directory + '/' + image_filename))
                continue

            img_h, img_w = img.shape

            try:
                assert(height == img_h)
                assert(width == img_w)
            except AssertionError:
                print('{}: dimension mismatch between XML and image ({}x{} vs {}x{}).  Skipping.'.format(filename, height, width, img_h, img_w))
                continue

            for tr in page.findall('a:TextRegion', ns):
                for tl in tr.findall('a:TextLine', ns):
                    coords = tl.find('a:Coords', ns)
                    (x0, y0, x1, y1) = points2rectangle(coords.get('points'))

                    snippet = img[y0:y1, x0:x1]
                    #snippet2 = cv2.cvtColor(snippet, cv2.COLOR_GRAY2RGB)

                    snip_height = y1-y0
                    snip_width = x1-x0

                    blur = cv2.GaussianBlur(snippet, (5, 5), 0)
                    ret3, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    threshed = cv2.bitwise_not(threshed)
                    contours, hierarchy = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    #cv2.drawContours(snippet2, contours, -1, (0, 255, 0), 3)

                    components = []
                    for i in range(0, len(contours)):
                        c = contours[i]
                        points = []
                        for p1 in c:
                            for p in p1:
                                x = p[0]
                                y = p[1]
                                points.append((x, y))
                        try:
                            poly = Polygon(points)
                            (px0, py0, px1, py1) = poly.bounds
                            pheight = int(py1-py0)
                            pwidth = int(px1-px0)
                            if pheight >= (snip_height * 0.95) and pwidth < (snip_width * 0.25):
                                pass
                            else:
                                components.append(poly)
                        except ValueError:
                            pass

                    maxy = 0
                    miny = 9999999
                    maxx = 0
                    minx = 9999999

                    for c in components:
                        (cx0, cy0, cx1, cy1) = c.bounds
                        if cx0 < minx:
                            minx = int(cx0)
                        if cx1 > maxx:
                            maxx = int(cx1)
                        if cy0 < miny:
                            miny = int(cy0)
                        if cy1 > maxy:
                            maxy = int(cy1)

                    coords_str = '{},{} {},{} {},{} {},{}'.format(x0+minx, y0+miny, x0+minx, y0+maxy, x0+maxx, y0+maxy, x0+maxx, y0+miny)
                    coords.set('points', coords_str)

                    #cv2.line(snippet2, (0, miny), (x1 - x0, miny), (128, 0, 0), thickness=1, lineType=8, shift=0)
                    #cv2.line(snippet2, (0, maxy), (x1 - x0, maxy), (0, 128, 0), thickness=1, lineType=8, shift=0)
                    #cv2.line(snippet2, (minx, 0), (minx, y1 - y0), (0, 0, 128), thickness=1, lineType=8, shift=0)
                    #cv2.line(snippet2, (maxx, 0), (maxx, y1 - y0), (128, 0, 128), thickness=1, lineType=8, shift=0)

                    #cv2.namedWindow('snippet')
                    #cv2.moveWindow('snippet', 0, 0)

                    #cv2.imshow('snippet', snippet2)
                    #key = cv2.waitKey(0)
                    #print(key)
                    #if key == 27:  # quit (esc)
                    #    print('break requested')
                    #    sys.exit(0)
                    #cv2.destroyAllWindows()

            outfilename = output_xml_directory + '/' + filename
            with open(outfilename, 'w') as outfile:
                outfile.write(prettify(root))


if __name__ == "__main__": main()
