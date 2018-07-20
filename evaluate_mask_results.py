import numpy as np
import cv2

from util.mask import compute_mask_iou, get_mask_approx_poly

if __name__ == '__main__':
    import sys
    import os, glob

    img_dir = "./checkpoints/unet1/web/images"
    # img_file = "../checkpoints/experiment_name/web/images/epoch063_data.png"

    iou_list = []

    for img_file in glob.glob(img_dir + "/*_data.png"):
        f = img_file.replace("_data.png","")
        mask_file = f + "_pred_mask.png"
        gt_mask_file = f + "_mask_gt.png"

        img = cv2.imread(img_file)
        gt_mask = cv2.imread(gt_mask_file)
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        _,gt_mask = cv2.threshold(gt_mask, 130, 255, cv2.THRESH_BINARY)

        mask = cv2.imread(mask_file)
        img_copy = img.copy()

        if mask is None or img is None or gt_mask is None:
            print("Could not read %s or %s or %s"%(mask_file, img_file, gt_mask_file))
            continue

        print("Showing %s"%(img_file))#, mask_file, gt_mask_file))

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

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _,mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)

        iou = compute_mask_iou(gt_mask / 255, mask / 255)
        iou_list.append(iou)

        print("Iou: %.3f, mean iou: %.3f"%(iou, np.mean(iou_list)))

        cv2.imshow("img", img)
        cv2.imshow("pred mask", mask)
        cv2.imshow("gt mask", gt_mask)
        cv2.imshow("im_copy", img_copy)
        cv2.waitKey(0)

        
    print("Total mean IOU: %.3f"%(np.mean(iou_list)))