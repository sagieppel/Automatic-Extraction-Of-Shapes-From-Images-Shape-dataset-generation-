import cv2
import os
import numpy as np
import random
import numpy as np
import colorsys
from skimage.color import lab2rgb
###########################################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
# Create multiple images with same texture but overlay on different orientations/shapes/backgrounds/
# For this you need one folder of textures and one folder of shaps saved as binary 0,255 masks (work of out the box with the mask and shapes folders supply)
# Part of the Large shape & textures dataset (LAS&T) generation  code

def random_color_lab():
    """Generates a perceptually uniform random color in RGB format."""
    L = np.random.uniform(20, 90)
    a = np.random.uniform(-80, 80)
    b = np.random.uniform(-80, 80)

    rgb = lab2rgb(np.array([[[L, a, b]]]))[0, 0]  # Convert Lab to RGB
    return rgb

#################################################################################################################################
def create_uniform_color_image(height=512, width=512,black=False,white=False):
    """
    Creates an image filled with a single uniform random color.

    Args:
        height (int): Image height.
        width (int): Image width.

    Returns:
        np.ndarray: A (height, width, 3) image with RGB values.
    """
    color = random_color_lab()
    image = np.ones((height, width, 3), dtype=np.uint8) * (color*255)  # Fill image with color
    if black:  image = np.ones((height, width, 3), dtype=np.uint8)
    if white:  image = np.ones((height, width, 3), dtype=np.uint8)*255
    return image

#############################################################################################################################
def rotate_image_without_crop(image, angle):
    # Get image size
    h, w = image.shape[:2]

    # Compute the new bounding dimensions
    diagonal = int(np.sqrt(h ** 2 + w ** 2))

    # Pad the image to fit the rotated shape
    pad_h = (diagonal - h) // 2
    pad_w = (diagonal - w) // 2
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)

    # Get rotation matrix
    center = (padded_image.shape[1] // 2, padded_image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform rotation
    rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (padded_image.shape[1], padded_image.shape[0]),
                                   borderValue=0)

    return rotated_image
####################################################################################################################################################33

# Prepare shape for deployment

#####################################################################33############################
def prep_shape(sp,img_sz,rotate_shape=True,keep_shape_size=True):
    if rotate_shape:
        angle= np.random.randint(0,360)
        sp = rotate_image_without_crop(sp,angle)

    r=img_sz/max(sp.shape[:2])
    if r<1:
        sp = cv2.resize(sp, (int( r * sp.shape[1]), int( r * sp.shape[0])))
    if not  keep_shape_size:
        r = img_sz / max(sp.shape[:2])
        if r > 1:
            sp = cv2.resize(sp, (int(r * sp.shape[1]), int(r * sp.shape[0])))

    d=img_sz-sp.shape[0]
    if d>0:
        x1=random.randint(0,d)
        x2=d-x1
        if x1>0:
            pad =np.zeros([x1,sp.shape[1],3],dtype=np.uint8)
            sp=np.concatenate([pad,sp],axis=0)
        if x2>0:
            pad = np.zeros([x2, sp.shape[1],3], dtype=np.uint8)
            sp = np.concatenate([sp,pad], axis=0)


    d = img_sz - sp.shape[1]
    if d > 0:
        x1 = random.randint(0, d)
        x2 = d - x1
        if x1 > 0:
            pad = np.zeros([sp.shape[0],x1,3], dtype=np.uint8)
            sp = np.concatenate([pad, sp], axis=1)
        if x2 > 0:
            pad = np.zeros([sp.shape[0], x2,3], dtype=np.uint8)
            sp = np.concatenate([sp,pad], axis=1)
    return sp

####################################################################################################################

# prepare texture for deployment

################################################################################################################################################
def prep_texture(tx,img_sz,rotate_texture):
    r=img_sz/min(tx.shape[:2])
    if r>1:
        tx=cv2.resize(tx,(int(1+r*tx.shape[1]),int(1+r*tx.shape[0])))
    r0 =  random.randint(0,tx.shape[0]-img_sz)
    r1 = random.randint(0, tx.shape[1] - img_sz)
    tx = tx[r0:r0+img_sz,r1:r1+img_sz]
    if rotate_texture:
        num_rot=random.randint(0,4)
        for i in range(num_rot):
                     tx = np.rot90(tx)
    return tx

#################################################################################################3

# Overlay textures on shape and background to create final image

##################################################################################################################################################
def deploy_texture(sp,tx1,tx2,img_sz,rotate_shape=True,keep_shape_size=True,rotate_texture=True):
    tx1=prep_texture(tx1, img_sz, rotate_texture)
    tx2=prep_texture(tx2, img_sz, rotate_texture)
    sp=prep_shape(sp, img_sz, rotate_shape, keep_shape_size)
    sp[sp>0]=1
    im=tx1.copy()
    im[sp>0]=tx2[sp>0]
    return im




########################################################################################################################################################3
# Create multiple images of the same textures in different setting/variations (like shape/bacground/rotations)
if __name__=="__main__":
    #----------------Input parameters------------------------------------------------------------------------------------------------
    shape_dir= "shapes/" #input folder of shapes saved as binary masks (see supplied shapes folder)
        #r"/media/deadcrow/6TB/Data_zoo/shapes//"
    texture_dir=r"textures" # input Folder of textures  saved  (see supplied shapes folder)
    out_main_dir=r"Texture_Matching_test/" # output folder of images of the same texture with different variationn will be saved
    num_inst=3 #  number of images to render with each shape
    img_sz=512 # Size of generated image

    num_inst=5 # number of images per texture
    img_sz=512 # outpput image size
    # Parameters that will be changed between different images of the same shape
    rotate_shape=False
    keep_shape_size=False
    rotate_texture=True
    uniform_shape=False
    shape_number=2
    uniform_background=True
#------------Create images -------------------------------------------------------------------
    if not os.path.exists(out_main_dir):
        os.mkdir(out_main_dir)
    all_textures = os.listdir(texture_dir)
    all_shapes = os.listdir(shape_dir)
    for indx1 in range(len(all_textures)):
         # if uniform_shape_texture:
         #    tx1 = create_uniform_color_image(img_sz, img_sz)
         # else:
          tx1 = cv2.imread(texture_dir + "//" + all_textures[indx1]) # load texture
          out_dir = out_main_dir+"//"+all_textures[indx1][:-4] +"//"
          if not os.path.exists(out_dir): os.mkdir(out_dir)
          for i in range(num_inst):
              if uniform_shape:
                  sp = cv2.imread(shape_dir + "/" + all_shapes[shape_number]) # load shape
              else:
                  sp=cv2.imread(shape_dir+"/"+random.choice(all_shapes))  # load random shape
              while(True):
                  indx2 = np.random.randint(0, len(all_textures)) # load background texture
                  if indx1!=indx2: break

              if uniform_background:
                  tx2 = create_uniform_color_image(img_sz, img_sz,black=True)
              else:
                  tx2 = cv2.imread(texture_dir + "//" + all_textures[indx2])
              im=deploy_texture(sp,tx2,tx1,img_sz,rotate_shape,keep_shape_size,rotate_texture) # combine elements for final image
              cv2.imwrite(out_dir+"/"+str(i)+".jpg",im)
              print(out_dir+"/"+str(i)+".jpg")



