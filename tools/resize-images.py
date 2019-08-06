import sys
import cv2
import os

def main():
    if len(sys.argv) != 3:
        print(
            'usage: resize-images.py input-image-directory output-image-directory')
        sys.exit(-1)

    input_img_directory = sys.argv[1].rstrip('/')
    output_img_directory = sys.argv[2].rstrip('/')

    file_listing = sorted(os.listdir(input_img_directory))
    for f in file_listing:
        filename = os.fsdecode(f)
        if filename.endswith('.jpg'):
            print('working on {}'.format(filename))
            img = cv2.imread(input_img_directory + '/' + filename)
            img_h, img_w, img_c = img.shape
            if img_h < img_w:
                if img_h > 800:
                    scale_factor = 800.0 / float(img_h)
            else:
                if img_w > 800:
                    scale_factor = 800.0 / float(img_w)

            img = cv2.resize(img, (round(float(img_w) * scale_factor), round(float(img_h) * scale_factor)))
            cv2.imwrite(output_img_directory + '/' + filename, img)

if __name__ == "__main__": main()


