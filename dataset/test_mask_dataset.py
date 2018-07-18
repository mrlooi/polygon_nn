import os.path
from dataset.base_dataset import BaseDataset
import random
import numpy as np
import torch
import cv2

MASK_SIZE = (224,224)

def _generate_polygons(mask_size, poly_cnt, mask_padding=(0,0)):
    h,w = mask_size
    ph,pw = mask_padding
    poly = [[random.randint(pw,w-pw-1),random.randint(ph,h-ph-1)] for i in range(poly_cnt)] # [x,y]
    return poly

def write_data(polygons, out_file):
    with open(out_file,'w') as f: 
        np.save(f, polys)
    print("Wrote to %s"%(out_file))

def read_data(mask_file):
    with open(mask_file,'r') as f: 
        return np.load(f)

class TestMaskDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(num_classes=len(LABELS))
        return parser


    def initialize(self, opt):
        self.opt = opt

        self.root = opt.dataroot

        self.mask = np.zeros((MASK_SIZE[0], MASK_SIZE[1], 3), np.uint8)
        self.mask_size = MASK_SIZE

        poly_file = os.path.join(self.root, "polys.npy")
        self.polys = read_data(poly_file)

        sz = len(self.polys)

        train_test_split = 0.8
        if opt.phase == "train":
            self.polys = self.polys[ : int(sz * train_test_split)]
        else:
            self.polys = self.polys[int(sz * train_test_split) : ]
        self.size = len(self.polys)


    def __getitem__(self, index):
        idx = index % self.size
        if not self.opt.serial_batches:
            idx = random.randint(0, self.size - 1) 
        polys = self.polys[idx]

        m = cv2.fillConvexPoly(self.mask.copy(), polys, (255, 255, 255))
        mh,mw,_ = m.shape
        # # resize 
        # osize = (self.opt.loadSize, self.opt.loadSize)
        # A_img = cv2.resize(A_img, osize)

        # transpose from H,W,3 to 3,H,W
        A = np.transpose(m, [2,0,1])
        # to tensor
        A = torch.from_numpy(A).float()
        A /= 255

        return {'data': A, 'polys': polys}

    def __len__(self):
        return self.size

    def get_class(self, index):
        return self.labels[index]

    def name(self):
        return 'TestMaskDataset'

def sample():
    total_polygons = 1000
    polygon_cnt = 10
    mask_size = MASK_SIZE
    mask_padding = (30,30)
    polys = np.array([_generate_polygons(mask_size, polygon_cnt, mask_padding) for i in range(total_polygons)])
    polys = np.array([p[p[:,1].argsort()] for p in polys])

    mask_file = "test/polys.npy"    
    write_data(polys, mask_file)

    polys = read_data(mask_file)

    for ix, p in enumerate(polys):
        mask = np.zeros(mask_size, np.uint8)
        p = np.array(polys[ix])
        # p = p[p[:,1].argsort()]
        m = cv2.fillConvexPoly(mask, p, 255)
        cv2.imshow("m", m)
        cv2.waitKey(0)
    

if __name__ == '__main__':
    # sample()

    class Opt():
        def __init__(self):
            self.dataroot = "./test"
            self.phase = "train"
            self.resize_or_crop = 'resize_and_crop'
            self.serial_batches = False
            self.no_flip = False
            self.isTrain = True

    opt = Opt()

    idd = TestMaskDataset()
    idd.initialize(opt)

    for i in range(10):
        data = idd[i]
        A = data['data']
        A_polys = data['polys']

        A_np = A.cpu().numpy()
        A_np = np.transpose(A_np, [1,2,0])

        cv2.imshow("m", A_np)
        # cv2.imshow("resized", cv2.resize(A_np, (128,128)))
        cv2.waitKey(0)
