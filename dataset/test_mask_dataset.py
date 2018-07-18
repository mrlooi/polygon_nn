import os.path
import random
import numpy as np
import torch
import cv2

from dataset.base_dataset import BaseDataset
from models import networks


MASK_SIZE = (224,224)

def _generate_polygons(mask_size, poly_cnt, mask_padding=(0,0)):
    h,w = mask_size
    ph,pw = mask_padding
    poly = [[random.randint(pw,w-pw-1),random.randint(ph,h-ph-1)] for i in range(poly_cnt)] # [x,y]
    return poly

def write_data(polygons, out_file):
    with open(out_file,'w') as f: 
        np.save(f, polygons)
    print("Wrote to %s"%(out_file))

def read_data(mask_file):
    with open(mask_file,'r') as f: 
        return np.load(f)

class TestMaskDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(model='simple_unet')
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
        osize = (self.opt.fineSize, self.opt.fineSize)
        m = cv2.resize(m, osize)
        m_gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

        # transpose from H,W,3 to 3,H,W
        A = np.transpose(m, [2,0,1])
        # to tensor
        A = torch.from_numpy(A).float()
        A /= 255

        # single channel i.e. grayscale version, but for masks still 0 and 255
        _,m_gray = cv2.threshold(m_gray, 10, 255, cv2.THRESH_BINARY)  # thresh the results of interpolation
        A_gt = torch.from_numpy(m_gray).float()
        # from H,W to 1,H,W
        A_gt = A_gt.unsqueeze(0)
        A_gt /= 255


        polys = polys.astype(np.float32)
        polys[:,0] *= (self.opt.fineSize/float(mh))
        polys[:,1] *= (self.opt.fineSize/float(mw))
        polys = polys.astype(np.int32)

        return {'data': A, 'gt': A_gt, 'polys': polys}

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

    for ix, p in enumerate(polys[:10]):
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
            self.loadSize = 256
            self.fineSize = 256
            self.no_flip = False
            self.isTrain = True
            self.ngf = 64
            self.norm = 'instance'
            self.no_dropout = False
            self.init_type = 'normal'
            self.init_gain = 0.02

    opt = Opt()

    netG = networks.define_G(3, 1, opt.ngf, 
                              "unet_256", opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, [0])

    idd = TestMaskDataset()
    idd.initialize(opt)

    dd = idd[0]
    sample_img = dd['data'].unsqueeze(0).cuda()
    sample_gt = dd['gt'].unsqueeze(0).cuda()
    # torch.cat((sample_gt,sample_gt), 0)

    pred_mask = netG.forward(sample_img)

    bce_criterion = torch.nn.BCEWithLogitsLoss(weight=None, reduce=True)
    bce_criterion = torch.nn.BCEWithLogitsLoss(weight=None, reduce=False)
    bce_criterion2 = torch.nn.BCELoss(weight=None, reduce=False)

    bce_loss_gt = torch.mean(bce_criterion2(sample_gt, sample_gt))
    bce_loss = torch.mean(bce_criterion2(torch.sigmoid(pred_mask), sample_gt))
    bce_loss = bce_criterion(pred_mask, sample_gt)

    img = sample_img.cpu().detach().numpy().squeeze()
    img = np.transpose(img, [1,2,0])
    gt_mask = sample_gt.cpu().detach().numpy().squeeze()

    pred_mask_cpu = pred_mask.cpu().detach().numpy()
    m = pred_mask_cpu.squeeze()
    m = (m+1)/2  # tanh output to 0 - 1 values
    m[m<0.5] = 0
    m[m>=0.5] = 1

    cv2.imshow("img", img)
    cv2.imshow("gt", gt_mask)
    cv2.imshow("pred_mask", m)
    cv2.waitKey(0)

    # for i in range(10):
    #     data = idd[i]
    #     A = data['data']
    #     A_polys = data['polys']

    #     A_np = A.cpu().numpy()
    #     A_np = np.transpose(A_np, [1,2,0])
    #     A_np *= 255
    #     A_np = A_np.astype(np.uint8).copy()

    #     for p in A_polys:
    #         cv2.circle(A_np, tuple(p), 2, (255,0,0))

    #     cv2.imshow("m", A_np)
    #     # cv2.imshow("resized", cv2.resize(A_np, (128,128)))
    #     cv2.waitKey(0)


