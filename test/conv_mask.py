import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
Keypoints tutorial
1. Change the data padding size
2. Change the conv kernel size according to padding size
"""

IMG_SIZE = 64

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
        m_gt = np.zeros((self.H, self.W), dtype=np.uint8)

        # ADD BLUE
        r = np.random.randint(p, self.H - p)
        c = np.random.randint(p, self.W - p)
        m[r-p:r+p+1, c-p:c+p+1, 0] = 255
        r1, c1 = r, c

        # ADD GREEN
        r = np.random.randint(p, self.H - p)
        c = np.random.randint(p, self.W - p)
        m[r-p:r+p+1, c-p:c+p+1, 1] = 255
        r2, c2 = r, c

        # ADD RED
        r = np.random.randint(p, self.H - p)
        c = np.random.randint(p, self.W - p)
        m[r-p:r+p+1, c-p:c+p+1, 2] = 255
        r3, c3 = r, c

        # input id
        input_id = np.random.randint(0,3)
        # if input_id==0:
        #     r, c = r1, c1
        # elif input_id==1:
        #     r, c = r2, c2
        # else:
        #     r, c = r3, c3
        r, c = r3, c3

        m_gt[r-p:r+p+1, c-p:c+p+1] = 255

        return [m, input_id, m_gt]

    def convert_data_batch_to_tensor(self, data, use_cuda=False):
        in_ids = []
        m_data = []
        m_gt_data = []
        for d in data:
            m = d[0]
            iid = d[1]
            m_gt = d[2]

            md = np.transpose(m, [2,0,1]).astype(np.float32)
            md /= 255
    
            m_gtd = m_gt.astype(np.float32) / 255

            m_data.append(md)
            m_gt_data.append(m_gtd)
            in_ids.append([iid])

        ti = torch.FloatTensor(in_ids)
        tm = torch.FloatTensor(m_data)
        tmgt = torch.FloatTensor(m_gt_data)
        if use_cuda:
            ti = ti.cuda() 
            tm = tm.cuda()
            tmgt = tmgt.cuda()
        return [tm, ti], tmgt

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        kernel_sz = 3
        stride = 1
        pad = 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_sz, stride=stride, padding=pad)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_sz, stride=stride, padding=pad)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=kernel_sz, stride=stride, padding=pad)
        self.upconv2 = nn.ConvTranspose2d(16, 1, kernel_size=kernel_sz, stride=stride, padding=pad)

        self.fc1 = nn.Linear(IMG_SIZE*IMG_SIZE, IMG_SIZE*IMG_SIZE)
        # self.fc1 = nn.Linear(IMG_SIZE*IMG_SIZE, 128)
        # self.fc2 = nn.Linear(128, IMG_SIZE*IMG_SIZE)

    def forward(self, x_img, x_id):
        batch_sz = len(x_img)
        x = F.relu(self.conv1(x_img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.upconv1(x))
        x = self.upconv2(x)
        x = x.view(-1, IMG_SIZE*IMG_SIZE)  # B,2,D,D -> Bx2,DxD

        # x = torch.cat((x, x_id), dim=1)

        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc1(x)

        x = F.sigmoid(x)
        x = x.view(-1, IMG_SIZE, IMG_SIZE)
        return x

def create_log_label_weights(log_label):
    assert len(log_label.shape) == 2
    h,w = log_label.shape
    instance_weight = np.zeros((h,w), np.float32)
    sumP = len(log_label[log_label==1])
    sumN = h*w - sumP
    # 'balanced' case only for instance weights
    weightP = 0.5 / sumP
    weightN = 0.5 / sumN
    for r in xrange(h):
        for c in xrange(w):
            instance_weight[r,c] = weightP if log_label[r,c] == 1 else weightN
    return instance_weight


@persistent_locals
def train(model, dg):

    epochs = 10
    n_iters = 100
    batch_size = 32
    lr = 3e-3

    model.train()

    criterion = nn.BCELoss(reduce=False)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    for iter in range(n_iters):
        data = dg.next_batch(batch_size)
        train_weights = torch.FloatTensor([create_log_label_weights(d[2]/255) for d in data]).cuda()
        train_x, train_y = dg.convert_data_batch_to_tensor(data, use_cuda=True)
        train_img, train_in_id = train_x

        optimizer.zero_grad()
        output = model(train_img, train_in_id)
        loss = criterion(output, train_y)
        weighted_loss = torch.mul(loss, train_weights)
        loss = torch.sum(weighted_loss.view(-1, IMG_SIZE*IMG_SIZE), dim=1)
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
    test_img, test_in_id = test_x

    output = model(test_img, test_in_id)

    pred = output.detach().cpu().numpy().squeeze()

    for ix in range(len(test_data)):
        d = test_data[ix]
        m = d[0]
        in_id = d[1]
        m_gt = d[2]

        pred_mask = pred[ix].copy()
        pred_mask[pred_mask>0.5]=1
        pred_mask[pred_mask<=0.5]=0

        print("In ID %d"%(in_id))
        cv2.imshow("im", m)
        cv2.imshow("gt", m_gt)
        cv2.imshow("pred", pred_mask)
        cv2.waitKey(0)


if __name__ == '__main__':
    dg = DataGenerator()
    data = dg.next_batch(8)

    model = ConvNet()
    model.cuda()

    train(model, dg)
    test(model, dg)

