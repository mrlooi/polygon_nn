import os, glob
import numpy as np
import xml.etree.ElementTree as ET

from dataset.base_dataset import BaseDataset

IMG_EXT = ".jpg"
ANNOT_EXT = ".xml"


def read_annot_file_data(annot_file, valid_classes=[], invalid_classes=[]):
    et = ET.parse(annot_file)
    element = et.getroot()
    # element_file = element.find('filename')
    # element_img_sz = element.find('imagesize')
    
    element_objects = element.findall('object')

    e_data = []
    e_pts_all = []
    e_pts_cls = []
    for e in element_objects:
        if int(e.find('deleted').text) == 1:
            continue
        # print(e.find('name').text)
        e_cls = e.find('name').text
        if len(valid_classes) > 0 and e_cls not in valid_classes:
            continue
        elif len(invalid_classes) > 0 and e_cls in invalid_classes:
            continue

        e_poly = e.find('polygon')
        e_pts = [( float(p.find('x').text), float(p.find('y').text) ) for p in e_poly.findall('pt')]
        e_pts = np.array(e_pts).astype(np.int32)
        e_data.append({'pts': e_pts, 'class': e_cls})
    return e_data

def get_labelme_file_basename(f):  # f 
    bname = f
    if f.endswith(IMG_EXT):
        bname = f.split("/")[-1].replace(IMG_EXT,"")
    elif f.endswith(ANNOT_EXT):
        bname = f.split("/")[-1].replace(ANNOT_EXT,"")
    return bname

class LabelMeDataset(BaseDataset):
    def __init__(self):
        super(LabelMeDataset, self).__init__()

    def name(self):
        return 'LabelMeDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        image_set = opt.image_set
        self.image_set = image_set
        self.image_root = os.path.join(self.root, "Images", image_set)
        self.annot_root = os.path.join(self.root, "Annotations", image_set)

        self.image_files = glob.glob(self.image_root + "/*%s"%(IMG_EXT))
        self.annot_files = glob.glob(self.annot_root + "/*%s"%(ANNOT_EXT))

        assert len(self.annot_files) == len(self.image_files)

        self.valid_classes = opt.valid_classes if hasattr(opt, 'valid_classes') else []
        self.invalid_classes = opt.invalid_classes if hasattr(opt, 'invalid_classes') else []

        self.data = []
        for f in self.annot_files:
            ad = read_annot_file_data(f, valid_classes=self.valid_classes, invalid_classes=self.invalid_classes)
            for d in ad:
                bname = get_labelme_file_basename(f)
                im_path = self.get_image_file(bname)
                d['bname'] = bname
                d['im_path'] = im_path
            self.data += ad

        self.size = len(self.data)

    def get_annot_file(self, bname):
        return os.path.join(self.annot_root, bname + ANNOT_EXT)

    def get_image_file(self, bname):
        return os.path.join(self.image_root, bname + IMG_EXT)

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    import cv2
    import random

    class Opt():
        def __init__(self):
            self.dataroot = "/home/vincent/LabelMe"
            self.image_set = "singulation"

    opt = Opt()
    idd = LabelMeDataset()
    idd.initialize(opt)

    for i in range(10):
        d = idd.data[i]
        im_path = d['im_path']
        img = cv2.imread(im_path)
        ih,iw,_ = img.shape
        poly = d['pts']
        masked = cv2.fillPoly(img, [poly], (0,0,255))
        print(d['class'])

        xmin, ymin=np.amin(poly, axis=0)
        xmax, ymax=np.amax(poly, axis=0)

        buffer_fraction = 0.4
        max_buffer_sz = 80
        x_buffer = min(max_buffer_sz, int((xmax-xmin) * buffer_fraction))
        y_buffer = min(max_buffer_sz, int((ymax-ymin) * buffer_fraction))
        xmin_offset = xmin - max(random.randint(xmin-x_buffer, xmin), 0)   
        ymin_offset = ymin - max(random.randint(ymin-y_buffer, ymin), 0)
        xmax_offset = min(random.randint(xmax, xmax+x_buffer), iw - 1) - xmax
        ymax_offset = min(random.randint(ymax, ymax+y_buffer), ih - 1) - ymax

        xmin2 = xmin - xmin_offset
        ymin2 = ymin - ymin_offset
        xmax2 = xmax + xmax_offset
        ymax2 = ymax + ymax_offset

        img_crop = img[ymin2:ymax2 , xmin2:xmax2]

        cv2.imshow("img", img)
        cv2.imshow("img_crop", img_crop)
        cv2.imshow("masked", masked)
        cv2.waitKey(0)

