# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:53:48 2022

@author: jamiu
"""

from PIL import Image
import glob


#display characteristics of images
img_path='C:/Users/Desktop/Research Images/crack/Un_resized/Capture7.JPG'
#img_path='C:/Users/Desktop/Research Images/spalling/Un_resized/spalling1.jpg'
#img_path='C:/Users/Desktop/shapes/corrosion2.jpg'
im= Image.open(img_path)
print('{}'.format(im.format))
print('size:{}'.format(im.size))
print('image mode: {}'.format(im.mode))



##%%empty lists
image_list=[]
resized_images=[]

##%% append images to list
for filename in glob.glob('C:/Users/Desktop/Research Images/crack/Un_resized/*.jpg'):
#for filename in glob.glob('C:/Users/Desktop/shapes/*.jpg'):
    print(filename)
    img=Image.open(filename)
    image_list.append(img)
    
    
##%% append resized images to list
for image in image_list:
    #image.show()
    image=image.resize((128,128))
    resized_images.append(image)
    
    
    
##save to new folder. in this wexample, we name it New_Shapes
#create a new folder manaually in destop or whatever. 
for (i, new) in enumerate(resized_images):
    new.save('{}{}{}'.format('C:/Users/Desktop/Research Images/crack/resized/crack',i+1,'.jpg'))
    
