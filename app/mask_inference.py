import numpy as np
import cv2
import torch
import glob

from util.mask import get_mask_approx_poly
from options.test_options import TestOptions
from models.simple_unet_model import SimpleUnetModel
from dataset.labelme_mask_dataset import LabelMeMaskDataset


def inference(model, img):  # assumes the img is after bounding box
    # transpose from H,W,3 to 3,H,W
    A = np.transpose(img, [2,0,1])
    # to tensor
    A = torch.from_numpy(A).float()
    A /= 255
    A = A.unsqueeze(0)  # batchsize 1
    pred = model.inference(A)
    pred = pred.detach().cpu().numpy().squeeze()
    return pred

def inference_with_polys(model, img, polys): # inference on a full image, with custom cropping basd on input polygon
    resize_shape = (model.opt.fineSize, model.opt.fineSize)
    A, A_gt = LabelMeMaskDataset.convert_data(img, polys, resize_shape)  
    A = A.unsqueeze(0)  # batchsize 1

    pred = model.inference(A)
    pred = pred.detach().cpu().numpy().squeeze()

    return pred


def get_init_op():
    class Opt():
        pass

    opt = Opt()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.no_flip = True  # no flip
    opt.gpu_ids = [0]
    opt.isTrain = False
    opt.checkpoints_dir = "./checkpoints"
    opt.name = "experiment_name"
    opt.resize_or_crop = None
    opt.ngf = 64
    opt.norm = "instance"
    opt.init_type = "normal"
    opt.init_gain = 0.02
    opt.phase = "test"
    opt.which_epoch = "latest"
    opt.fineSize = 256
    opt.verbose = False
    opt.no_dropout = True
    
    return opt

def test_on_dir(model):
    opt = model.opt
    eps = 0.01

    img_dir = "./results/experiment_name/test_latest/images"
    for img_file in glob.glob(img_dir + "/*_data.png"):
        img = cv2.imread(img_file)
        print(img_file)
        
        osize = (opt.fineSize, opt.fineSize)

        img = cv2.resize(img, osize)
        img_copy = img.copy()

        pred = inference(model, img)
        mask = (pred + 1) / 2.0 #* 255
        # mask = pred_mask.astype(np.uint8)

        contours, approx = get_mask_approx_poly(mask, eps=eps)
        if len(contours) == 0:
            print("Could not find contours!")
            continue
        cnts = contours[0]
        cv2.drawContours(img_copy, [cnts], 0, (0,0,255), 1)
        print("cnt area: %.3f"%(cv2.contourArea(cnts)))
        
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if len(approx) > 0:
            for i,p in enumerate(approx):
                p = tuple(p)
                pt2 = tuple(approx[(i+1)%len(approx)])
                cv2.line(img_copy, p, pt2, (0,255,0), 2)
                cv2.circle(img_copy, p, 3, (255,0,0))
                cv2.circle(mask, p, 3, (255,0,0))

        cv2.imshow("img", img_copy)
        cv2.imshow("pred", mask)

        f = img_file.replace("_data.png","")
        mask_file = f + "_pred_mask.png"
        out_file = f + "_out.png"

        cv2.imshow("pred2", cv2.imread(mask_file))
        # cv2.imwrite(out_file, img_copy)
        cv2.waitKey(1)


if __name__ == '__main__':
    import sys

    opt = get_init_op()
    model = SimpleUnetModel()
    model.initialize(opt)
    model.setup(opt)

    # test_on_dir(model) 

    eps = 0.01
    
    img_file = "/home/vincent/LabelMe/Images/singulation_test/rgb2_0_281.jpg"
    # img_file = "/home/vincent/LabelMe/Images/unity_boxes/boxes_201708311714499595.jpg"
    # img_file = "/home/vincent/Documents/deep_learning/polyrnn-pp/imgs2/horse.png"

    # Read image
    img = cv2.imread(img_file)
    if img is None:
        print("Could not read %s"%(img_file))
        sys.exit(1)
    img_copy = img.copy()
     
    # Select ROI
    fromCenter = False

    resize_shape = (opt.fineSize, opt.fineSize)

    while True:
        r = cv2.selectROI(img_copy, fromCenter)
         
        # Crop image
        print(r)
        img_crop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        
        h_ratio = r[3] / float(resize_shape[0])
        w_ratio = r[2] / float(resize_shape[1])

        img_crop = cv2.resize(img_crop, resize_shape)
        img_crop_copy = img_crop.copy()

        pred = inference(model, img_crop)
        mask = (pred + 1) / 2.0 * 255
        mask = mask.astype(np.uint8)

        contours, approx = get_mask_approx_poly(mask, eps=eps)
        if len(contours) == 0:
            print("Could not find contours!")
            sys.exit(1)
        cnts = contours[0]
        cv2.drawContours(img_crop_copy, [cnts], 0, (0,0,255), 1)
        print("cnt area: %.3f"%(cv2.contourArea(cnts)))
        
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if len(approx) == 0:
            continue

        # on cropped image
        for i,p in enumerate(approx):
            p = tuple(p)
            pt2 = tuple(approx[(i+1)%len(approx)])
            cv2.line(img_crop_copy, p, pt2, (0,255,0), 2)
            cv2.circle(img_crop_copy, p, 3, (255,0,0))
            cv2.circle(mask, p, 3, (255,0,0))

        # on original image
        approx = approx.astype(np.float32)
        approx[:,0] *= w_ratio
        approx[:,1] *= h_ratio
        approx[:,0] += r[0]
        approx[:,1] += r[1]
        approx = approx.astype(np.int32)
        for i,p in enumerate(approx):
            p = tuple(p)
            pt2 = tuple(approx[(i+1)%len(approx)])
            cv2.line(img_copy, p, pt2, (0,255,0), 2)
            cv2.circle(img_copy, p, 3, (255,0,0))

        # cv2.imshow("Image", img)
        # cv2.imshow("Crop", img_crop)
        # cv2.imshow("Pred Mask", mask)
        # cv2.imshow("Polygons", img_crop_copy)
        # cv2.waitKey(0)
