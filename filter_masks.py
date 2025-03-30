# turn nonebinary mask  to  binary masks
import os
import cv2
import numpy as np
indir= "/media/deadcrow/SSD_480GB/segment_anything/semantic_shapes/"
for fl in os.listdir(indir):
    im=cv2.imread(indir+"/"+fl)
    im[im<125]=0
    im[im > 125] = 255
    # cv2.imshow("",im)
    # cv2.waitKey()
    cv2.imwrite(indir+"/"+fl,im)
    print(indir+"/"+fl)
