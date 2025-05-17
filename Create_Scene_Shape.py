# Create multiple images with same shape but with different orientations/colors/textures/backgrounds/
# For this you need one folder of textures and one folder of shaps saved as binary 0,255 masks (work of the box with the mask and shapes folders supply)
# Part of the Large shape & textures dataset (LAS&T) generation  code
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

###########################################################################################################################################
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

# Rotate image on the image plane

#######################################################################################################33
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
###############################################################################################################################

#

####################################################################################################################################################33
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


################################################################################################################################################

# Prepare texture  for deployment

########################################################################################################################
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
#######################################################################################################3333

# Overlay texture on shape (mask) and background

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
# Main script generate for each shape multiple variations and save as image
if __name__=="__main__":
    #----------------Input parameters------------------------------------------------------------------------------------------------
    shape_dir= "shapes/" #input folder of shapes saved as binary masks (see supplied shapes folder)
    texture_dir=r"textures" # input Folder of textures  saved  (see supplied shapes folder)
    out_main_dir=r"Shape_Matching_test/" # output folder of images of the same shape with different variations (orientation/texture/color/background)
    #***********************************
    # shape_dir = r"/home/deadcrow/Downloads/SHAPES_2D_350k_UNIFIED/"  # input folder of shapes saved as binary masks (see supplied shapes folder)
    # texture_dir = r"/media/deadcrow/SSD_480GB/Extracted_textures_larger_512_pix_55k_Set2/textures_larger_512_pix_50k//"  # input Folder of textures  saved  (see supplied shapes folder)
    # out_main_dir = r"/media/deadcrow/SSD_480GB/segment_anything/2D_Shape_Matching_Tests/2D_Shapes_Recognition_Textured_Synthetic//"  # output folder of images of the same shape with different variations (orientation/texture/color/background)

    #***************************************


    num_inst=5 #  number of images to render with each shape
    img_sz=512 # Size of generated image
    resize_range = [int(img_sz*0.32), img_sz] # Size range for the shape (if you use resize
    #  Which element of the shape will be modified between different images:
    rotate_shape=True # randomly rotate shape
    keep_shape_size=True # if this is false the shape will be resize to max size feet the image
    rotate_texture=True # randomly rotate exture
    uniform_shape_texture=False  # if False the background will be cover with a  random texture
    uniform_background=False # if False the background will be cover with a  random texture
    black_and_white=False # shape will be white background will be black else each will have random color only work if  uniform_background and or uniform_shape_texture
    resize_shape=True # randomly resize shape in sizes in resize_range
    if resize_shape: keep_shape_size=True


    #-------------------Main script---------------------------------------------------------
    if not os.path.exists(out_main_dir):
         os.mkdir(out_main_dir)
    all_texutres= os.listdir(texture_dir)

    for spfile in os.listdir(shape_dir):
          sp_origin = cv2.imread(shape_dir+"/"+spfile) # load shape


          out_dir = out_main_dir+"//"+spfile[:-4] +"//"
          if not os.path.exists(out_dir): os.mkdir(out_dir)
          for i in range(num_inst):
              sp=sp_origin.copy()
              if resize_shape:  # random resize
                  mx = np.max(sp.shape)
                  szrange = resize_range / np.max(sp.shape)
                  rs = np.random.rand() * (szrange[1] - szrange[0]) + szrange[0]
                  sp = cv2.resize(sp, (int(rs * sp.shape[1]), int(rs * sp.shape[0])), cv2.INTER_NEAREST)

              indx1=np.random.randint(0,len(all_texutres))
              while(True):
                  indx2 = np.random.randint(0, len(all_texutres)) # load shape texture
                  if indx1!=indx2: break
              if uniform_background: # uniforemd  color background
                  tx1=create_uniform_color_image(img_sz,img_sz,black=black_and_white) # load background texture
              else:
                  tx1 = cv2.imread(texture_dir+"//"+all_texutres[indx1])
              if uniform_shape_texture: # unform colored shape
                  tx2 = create_uniform_color_image(img_sz, img_sz,white=black_and_white)
              else:
                  tx2 = cv2.imread(texture_dir + "//" + all_texutres[indx2])
              im=deploy_texture(sp,tx1,tx2,img_sz,rotate_shape,keep_shape_size,rotate_texture) # add texture to shape and background
              cv2.imwrite(out_dir+"/"+str(i)+".jpg",im) # save
              print(out_dir+"/"+str(i)+".jpg")



