# Automatic Extraction natural 2D shapes from images 

## Extracting shapes from images: ExtractShape.py

ExtractShape.py run over folder of images and extract 2D shapes, should run out of the box with the supplied sampled folders and work with any images.
This will generate an unlimited amount of 2D shapes. 
A sample of 12,000 extracted 2D shapes can be downloaded from [this url.](https://drive.google.com/file/d/1Mb6aYvcwqRGdydCY7AFdvs1zwR8JpOwQ/view?usp=drive_link)  

For large set of images it possible to use the [Segement anything dataset](https://segment-anything.com/dataset/index.html) or the [open images  dataset](This will generate an unlimited amount of 2D shapes.) 

![Sampled extracted shapes](shapes.jpg)

## Creating shape matching test: Create_Scene_Shape.py

Create_Scene_Shape.py will generate images of the same shape but with different orientations/textures/color/backgound. 
This can be used for the 2D shape matching dataset of the [large shape and texture dataset](https://arxiv.org/pdf/2503.23062).

Samples of the generated images can be downloaded from [this url](https://icedrive.net/s/tDNwSabRZfAbx8yiXxb15xag529v).

This code can run on simple cpu it need set of textures saved as a rgb images, and set of shapes saved as binary images (see above scripts.)

Shapes can be extracted from images using the above scripts or downloaded from [this url.]([https://icedrive.net/s/gTbNa4BaCRGAijRBvW4AVihZ8y8h])  

Textures can be downloaded from the vastexture datasets (Mirror1)[https://sites.google.com/view/infinitexture/home],  (Mirror2)[https://zenodo.org/records/12629301].



## Creating texture matching test: Create_Scene_Textures.py

Create_Scene_Textures.py will create images of the same texture but deploy on different 2D shapes and with different background, this can be used for the textures matching dataset of the [large shape and texture dataset](https://arxiv.org/pdf/2503.23062). 


Samples of the generated images can be downloaded from [this url](https://icedrive.net/s/tDNwSabRZfAbx8yiXxb15xag529v).

This code can run on simple cpu it need set of textures saved as a rgb images, and set of shapes saved as binary images (see above scripts.)

Shapes can be extracted from images using the above scripts or downloaded from [this url.](https://drive.google.com/file/d/1Mb6aYvcwqRGdydCY7AFdvs1zwR8JpOwQ/view?usp=drive_link)  

Textures can be downloaded from the vastexture datasets (Mirror1)[https://sites.google.com/view/infinitexture/home],  (Mirror2)[https://zenodo.org/records/12629301].
