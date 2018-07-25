import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

IMG_SIZE = 64

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

class persistent_locals(object):
    def __init__(self, func):
        self._locals = {}
        self.func = func

    def __call__(self, *args, **kwargs):
        def tracer(frame, event, arg):
            if event=='return':
                l = frame.f_locals.copy()
                self._locals = l
                for k,v in l.items():
                    globals()[k] = v

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            # trace the function call
            res = self.func(*args, **kwargs)
            
        finally:
            # disable tracer and replace with old one
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}

    @property
    def locals(self):
        return self._locals

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        kernel_sz = 3
        stride = 1
        pad = 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_sz, stride=stride, padding=pad)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_sz, stride=stride, padding=pad)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=kernel_sz, stride=stride, padding=pad)
        self.upconv2 = nn.ConvTranspose2d(16, 2, kernel_size=kernel_sz, stride=stride, padding=pad)

    def forward(self, x):
        batch_sz = len(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.upconv1(x))
        x = self.upconv2(x)
        x = x.view(-1, IMG_SIZE*IMG_SIZE)  # B,2,D,D -> Bx2,DxD
        x = F.log_softmax(x, dim=1)
        return x

"""
Creates a 
"""
class DataGenerator():
    def __init__(self):
        self.H = IMG_SIZE
        self.W = IMG_SIZE
        self.padding = 3

    def next_batch(self, batch_size=8):
        data = [self._get_random_data() for i in range(batch_size)]
        return data

    def _get_random_data(self):
        p = self.padding
        m = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        r = np.random.randint(p, self.H - p)
        c = np.random.randint(p, self.W - p)
        m[r-p:r+p+1, c-p:c+p+1, 2] = 255
        kr, kc = r, c
        # if c - p 

        # ADD BLUE
        r = np.random.randint(p, self.H - p)
        c = np.random.randint(p, self.W - p)
        m[r-p:r+p+1, c-p:c+p+1, 0] = 255

        return [m, (kr,kc), (r,c)]

    def convert_data_batch_to_tensor(self, data, use_cuda=False):
        keypts_idx = []
        m_data = []
        for d in data:
            m = d[0]
            r, c = d[1]
            keypts_idx.append(r*self.W + c)
            r, c = d[2]
            keypts_idx.append(r*self.W + c)

            md = np.transpose(m, [2,0,1]).astype(np.float32)
            md /= 255
            m_data.append(md)
        tki = torch.LongTensor(keypts_idx)
        tm = torch.FloatTensor(m_data)
        if use_cuda:
            tki = tki.cuda() 
            tm = tm.cuda()
        return tm, tki

@persistent_locals
def train(model, dg):

    epochs = 10
    n_iters = 100
    batch_size = 16
    lr = 3e-3

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    for iter in range(n_iters):
        data = dg.next_batch(batch_size)

        train_x, train_y = dg.convert_data_batch_to_tensor(data, use_cuda=True)

        optimizer.zero_grad()
        output = model(train_x)
        loss = F.nll_loss(output, train_y)
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        print(loss.item())

@persistent_locals
def test(model, dg):
    model.eval()
    test_batch_sz = 8
    test_data = dg.next_batch(test_batch_sz)
    test_x, test_y = dg.convert_data_batch_to_tensor(test_data, use_cuda=True)

    output = model(test_x)
    test_loss = F.nll_loss(output, test_y).item()

    predl = output.max(1, keepdim=True)[1]
    correct = predl.eq(test_y.view_as(predl)).sum().item()
    pred = predl.cpu().numpy().squeeze()
    pred = pred.reshape((test_batch_sz, 2))

    for ix in range(len(test_data)):
        d = test_data[ix]
        m = d[0]
        gt_pts = d[1:]
        pred_pts = pred[ix]

        m_copy = m.copy()

        for pix in range(len(gt_pts)):
            gt_pt = gt_pts[pix]
            pred_pt_idx = pred_pts[pix]
            pred_pt = (pred_pt_idx / dg.W, pred_pt_idx % dg.W)
            print("GT: %d %d, Pred: %d %d"%(gt_pt[0],gt_pt[1],pred_pt[0],pred_pt[1]))

            cv2.circle(m, tuple(gt_pt)[::-1], 1, (0,255,0))
            cv2.circle(m_copy, tuple(pred_pt)[::-1], 1, (0,255,0))

        cv2.imshow("gt", m)
        cv2.imshow("pred", m_copy)
        cv2.waitKey(0)


if __name__ == '__main__':
    dg = DataGenerator()
    data = dg.next_batch(8)

    # m = data[0][0]
    # cv2.imshow("m", m)
    # cv2.waitKey(0)

    model = ConvNet()
    model.cuda()
    train_x, train_y = dg.convert_data_batch_to_tensor(data, use_cuda=True)
    model(train_x)

    train(model, dg)
    test(model, dg)

