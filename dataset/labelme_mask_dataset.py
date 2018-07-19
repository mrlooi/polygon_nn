import os
import numpy as np
import random
import cv2
import torch

from dataset.labelme_dataset import LabelMeDataset


class LabelMeMaskDataset(LabelMeDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(model='simple_unet')
        return parser

    def initialize(self, opt):
        super(LabelMeMaskDataset, self).initialize(opt)

    def __getitem__(self, index):
        idx = index % self.size
        if not self.opt.serial_batches:
            idx = random.randint(0, self.size - 1) 
        data = self.data[idx]
        polys = data['pts']
        img_path = data['im_path']

        img = cv2.imread(img_path)
        assert img is not None

        # resize 
        osize = (self.opt.fineSize, self.opt.fineSize)
        
        A, A_gt = LabelMeMaskDataset.convert_data(img, polys, osize)

        # perform random flips
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            A_gt = A_gt.index_select(2, idx)

        return {'data': A, 'gt': A_gt, 'path': img_path}

    @staticmethod   
    def convert_data(img, polys, resize_shape):
        ih,iw,_ = img.shape
        
        xmin,ymin = np.amin(polys, axis=0)
        xmax,ymax = np.amax(polys, axis=0)

        mask = np.zeros((ih,iw), dtype=np.uint8)

        mask = cv2.fillPoly(mask, [polys], 255)

        # randomly crop around the target region up to a certain range
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

        img = img[ymin2:ymax2,xmin2:xmax2]
        mask = mask[ymin2:ymax2,xmin2:xmax2]

        # resize 
        mask = cv2.resize(mask, resize_shape)
        img = cv2.resize(img, resize_shape)

        # transpose from H,W,3 to 3,H,W
        A = np.transpose(img, [2,0,1])
        # to tensor
        A = torch.from_numpy(A).float()
        A /= 255

        A_gt = torch.from_numpy(mask).float()
        # from H,W to 1,H,W
        A_gt = A_gt.unsqueeze(0)
        A_gt /= 255

        return A, A_gt

    def __len__(self):
        return len(self.data)

    def name(self):
        return 'LabelMeMaskDataset'

if __name__ == '__main__':
    from models import networks

    class Opt():
        def __init__(self):
            self.dataroot = "/home/vincent/LabelMe"
            self.phase = "train"
            self.serial_batches = False
            self.fineSize = 256
            self.no_flip = False
            self.isTrain = True
            self.ngf = 64
            self.norm = 'instance'
            self.no_dropout = False
            self.init_type = 'normal'
            self.init_gain = 0.02
            self.image_set = "singulation"
            self.invalid_classes = 'barcode'

    opt = Opt()
    idd = LabelMeMaskDataset()
    idd.initialize(opt)
    idd[0]

    netG = networks.define_G(3, 1, opt.ngf, 
                              "unet_256", opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, [0])

    for i in range(10):
        dd = idd[i]
        sample_img = dd['data'].unsqueeze(0).cuda()
        sample_gt = dd['gt'].unsqueeze(0).cuda()

        pred_mask = netG.forward(sample_img)
        pred_mask_cpu = pred_mask.cpu().detach().numpy()

        bce_criterion = torch.nn.BCELoss(weight=None, reduce=False)
        bce_loss_gt = torch.mean(bce_criterion(sample_gt, sample_gt))
        bce_loss = torch.mean(bce_criterion(torch.sigmoid(pred_mask), sample_gt))

        img = sample_img.cpu().detach().numpy().squeeze()
        img = np.transpose(img, [1,2,0])
        gt_mask = sample_gt.cpu().detach().numpy().squeeze()

        m = pred_mask_cpu.squeeze()
        m = (m+1)/2  # tanh output to 0 - 1 values
        m[m<0.5] = 0
        m[m>=0.5] = 1

        cv2.imshow("img", img)
        cv2.imshow("gt", gt_mask)
        cv2.imshow("pred_mask", m)
        cv2.waitKey(0)
