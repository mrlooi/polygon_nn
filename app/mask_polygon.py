import numpy as np
import cv2

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
    import sys
    import os, glob

    img_dir = "../checkpoints/experiment_name/web/images"
    # img_file = "../checkpoints/experiment_name/web/images/epoch063_data.png"

    for img_file in glob.glob(img_dir + "/*_data.png"):
        f = img_file.replace("_data.png","")
        mask_file = f + "_pred_mask.png"

        img = cv2.imread(img_file)
        mask = cv2.imread(mask_file)
        img_copy = img.copy()

        if mask is None or img is None:
            print("Could not read %s or %s"%(mask_file, img_file))
            continue

        print("Showing %s and %s"%(img_file, mask_file))

        eps = 0.01
        contours, approx = get_mask_approx_poly(mask, eps=eps)
        if len(contours) == 0:
            print("Could not find contours!")
            continue
        cnts = contours[0]
        cv2.drawContours(img_copy, [cnts], 0, (0,0,255), 1)
        print("cnt area: %.3f"%(cv2.contourArea(cnts)))
    
        if len(approx) > 0:
            for i,p in enumerate(approx):
                p = tuple(p)
                pt2 = tuple(approx[(i+1)%len(approx)])
                cv2.line(img_copy, p, pt2, (0,255,0), 2)
                cv2.circle(img_copy, p, 3, (255,0,0))
                cv2.circle(mask, p, 3, (255,0,0))
        # print(approx)

        cv2.imshow("img", img)
        cv2.imshow("pred mask", mask)
        cv2.imshow("im_copy", img_copy)
        cv2.waitKey(0)

        