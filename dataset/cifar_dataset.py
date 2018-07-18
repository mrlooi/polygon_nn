import os.path
from dataset.base_dataset import BaseDataset
import random
import numpy as np
import torch
import torchvision.transforms as ttransforms
import cv2

LABELS = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def read_img(img_file):
    img_data = cv2.imread(img_file) 
    # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY) # convert to grayscale 
    # img_data = np.expand_dims(img_data,axis=-1)  # (H,W,1)
    return img_data

class CifarDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(num_classes=len(LABELS))
        return parser

    def initialize(self, opt):
        self.opt = opt

        # self.opt.input_nc = 1 # grayscale
        # self.opt.output_nc = 1 # depth

        self.labels = LABELS
        self.num_classes = len(LABELS)

        self.root = opt.dataroot

        label_file = os.path.join(opt.dataroot, "%s_labels.txt"%(opt.phase))
        self.A_data = []
        with open(label_file, "r") as f: 
            data = [i.strip("\n") for i in f.readlines()]
            for d in data:
                path, cls = d.split(",")
                self.A_data.append([os.path.join(opt.dataroot, path), int(cls)])

        self.A_size = len(self.A_data)

    def __getitem__(self, index):
        idx = index % self.A_size
        if not self.opt.serial_batches:
            idx = random.randint(0, self.A_size - 1) 
        A_data = self.A_data[idx]
        A_path = A_data[0]
        A_cls = A_data[1]
        A_img = read_img(A_path) 
        A_img = cv2.resize(A_img, (224,224))

        # # resize 
        # osize = (self.opt.loadSize, self.opt.loadSize)
        # A_img = cv2.resize(A_img, osize)

        # transpose from H,W,3 to 3,H,W
        A = np.transpose(A_img, [2,0,1])
        # to tensor
        A = torch.from_numpy(A).float()
        A /= 255

        # cropping
        # w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        # A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        # A = ttransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        # A = ttransforms.Normalize((0.5,), (0.5,))(A)

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)

        return {'data': A, 'paths': A_path, 'labels': A_cls}

    def __len__(self):
        return self.A_size

    def get_class(self, index):
        return self.labels[index]

    def name(self):
        return 'CifarDataset'

if __name__ == '__main__':
    class Opt():
        def __init__(self):
            self.dataroot = "/home/vincent/hd/datasets/cifar"
            self.phase = "train"
            self.resize_or_crop = 'resize_and_crop'
            self.loadSize = 286
            self.fineSize = 256
            self.serial_batches = False
            self.no_flip = False
            self.isTrain = True

    opt = Opt()

    idd = CifarDataset()
    idd.initialize(opt)

    # A_path = idd.A_paths[0]
    # B_path = idd.B_paths[0]
    # A_img = cv2.imread(A_path) # Image.open(A_path).convert('RGB')
    # B_img = read_depth(B_path)
    
    # A = idd.A_transform(A_img)
    # B = idd.B_transform(B_img)

    for i in range(10):
        data = idd[i]
        A = data['data']
        A_path = data['paths']
        A_cls = data['labels']

        print("Showing %s"%(A_path))
        cls = idd.get_class(A_cls)
        print("Class: %s"%(cls))
        A_np = A.cpu().numpy()
        A_np = np.transpose(A_np, [1,2,0])

        cv2.imshow(cls, A_np)
        # cv2.imshow("resized", cv2.resize(A_np, (128,128)))
        cv2.waitKey(0)
