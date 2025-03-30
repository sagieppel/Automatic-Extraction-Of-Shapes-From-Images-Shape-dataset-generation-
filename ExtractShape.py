#  Extract shapes from set of images

import os
import numpy as np
import cv2
######################################################################################################

# Turn image into binary segmentation map, by picking image property and thresholding it (both random picked)

######################################################################################################
def get_random_shape_from_image(img_origin,max_occupancy_fracttion=0.5,min_area=20000,erode_size=11, eroded_min_area=15000):
    for kk in range(15):
        # pick random image property
        if np.random.rand() < 0.5:
            img = cv2.cvtColor(img_origin.copy(), cv2.COLOR_BGR2HSV)
        else:
            img = img_origin.copy()


        map=img[:,:,np.random.randint(3)].astype(np.float32)

        # normalize map value

        map = (map / 255)
        # inverse
        if np.random.rand()<0.5:
            map= 1-map

        for ii in range(20): # Try random threshold
            thresh = np.random.rand()#*(map.max()-map.min())+map.min()
            binmap = (map>thresh).astype(np.uint8)

            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binmap, connectivity=8)

            # Exclude background (label 0)
            if num_labels < 2:
                continue  # No objects found

            # Get the largest component (excluding the background)
            sizes = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = np.argmax(sizes) + 1

            # Extract the bounding box of the largest component
            x, y, w, h, area = stats[largest_idx]

            # Check if it touches the image boundary
            if x < 5 or y < 5 or x + w > map.shape[1]-5 or y + h > map.shape[0]-5:
                continue # Component touches the boundary

            # Check if it is larger than the minimum size

            if area < min_area:
                continue

            # Extract and erode the largest component
            component_mask = (labels == largest_idx).astype(np.uint8) * 255
            if component_mask.mean()>max_occupancy_fracttion*255: continue
            s1=component_mask.sum()
            component_mask=component_mask[y:y+h,x:x+w]
            s2 = component_mask.sum()
            if s1!=s2:
                print("ddddd")
            kernel = np.ones((erode_size, erode_size), np.uint8)
            eroded_mask = cv2.erode(component_mask, kernel, iterations=1)

            # Check if the eroded component is still larger than the eroded_min_size
            if cv2.countNonZero(eroded_mask) < eroded_min_area:continue
            # if component_mask[0:10,:].sum()>0 or component_mask[:,0:10].sum()>0 or  component_mask[-10:-1,:].sum()>0 or component_mask[:,-10:-1].sum()>0:
            #        print("somthing wrong")
            #        cv2.imshow("", cv2.resize(component_mask, (int(component_mask.shape[1] / 3), int(component_mask.shape[0] / 3))))
            #        cv2.waitKey()
            #        xx=4
            return component_mask, True
    return None,False
######################################################################################################################################3333

#  Run extraction on folder of images and get shapes  (should run out of the box with supplied folders)

#######################################################################################################################
if __name__=="__main__":
     indir = r"images//" # folder of input images (see supplied "images" folder)
     outdir = r"shapes//" # folder where the extracted shapes will be saved (see supplied "shapes" folder()
     if not os.path.exists(outdir): os.mkdir(outdir)
     max_size=512
     for fl in os.listdir(indir):
         if ".jpg" not in fl: continue
         im = cv2.imread(indir+"/"+fl)
         r=512/np.min(im.shape[:2])
         if r<1:
             im=cv2.resize(im,[int(im.shape[1]*r),int(im.shape[0]*r)])

         mask, success = get_random_shape_from_image(im)
         print(indir+"/"+fl,"Sucsses:",success)
         if success:
             cv2.imwrite(outdir+"/"+fl[:-4]+".png",mask)
             print(outdir+"/"+fl[:-4]+".png")
             # cv2.imshow("",cv2.resize(mask,(int(mask.shape[1]/3),int(mask.shape[0]/3))))
             # cv2.waitKey()