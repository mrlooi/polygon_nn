import numpy as np
import cv2
# import pycocotools.coco as coco
from pycocotools.coco import maskUtils as mutils

def compute_mask_iou(x,y):
	iscrowd = [0] #[0 for i in range(len(x))]
	# if x 
	xx = np.asfortranarray(x)
	rle_x = mutils.encode(xx)
	yy = np.asfortranarray(y)
	rle_y = mutils.encode(yy)

	iou = mutils.iou([rle_x],[rle_y],iscrowd)[0][0]
	return iou

def approx_poly(cnts, eps=0.01):
    arclen = cv2.arcLength(cnts, True)
    epsilon = arclen * eps
    approx = cv2.approxPolyDP(cnts, epsilon, True)
    return approx

def get_mask_contours(mask, is_sorted=True):
    m = mask.copy()
    dims = len(m.shape)
    if dims == 3 and m.shape[2] == 3: # color, convert to gray
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    _,m = cv2.threshold(m, 20, 255, cv2.THRESH_BINARY) # thresh it 
    _, contours, hierarchy = cv2.findContours(m,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if is_sorted:
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
    return contours

def get_mask_approx_poly(mask, eps=0.01):
    contours = get_mask_contours(mask, is_sorted=True)
    if len(contours) == 0:
        return ([], [])
    cnts = contours[0]
    approx = approx_poly(cnts, eps=eps)
    return (contours, approx.squeeze())

if __name__ == '__main__':
	x = np.ones((10,10), dtype=np.uint8)
	y = np.ones((10,10), dtype=np.uint8)
	x *= 255
	y *= 255
	y[0][0] = 0
	x[0][1] = 0

	iou = compute_mask_iou(x,y)
	print(iou)