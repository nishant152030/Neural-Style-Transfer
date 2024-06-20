import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
import keras.preprocessing.image as process_im
from PIL import Image
import matplotlib.pyplot as plt
from keras.applications import vgg19
from keras.models import Model
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import functools
import IPython.display

import functions as f
content_path='./images/bridge.jpg' # path to content image
style_path = './images/curls.jpg' # path to style image

content = f.load_file(content_path)
style = f.load_file(style_path)

plt.figure(figsize=(10,10))
content = load_file(content_path)
style = load_file(style_path)
plt.subplot(1,2,1)
f.show_im(content,'Content Image')
plt.subplot(1,2,2)
f.show_im(style,'Style Image')
plt.show()

im=f.img_preprocess(content_path)

content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
number_content=len(content_layers)
number_style =len(style_layers)

model=tf.keras.applications.vgg19.VGG19(include_top=False,weights='imagenet')
model.summary()

model=f.get_model()
model.summary()

model=tf.keras.applications.vgg19.VGG19(include_top=False,weights='imagenet')

best, best_loss,image = f.run_style_transfer(content_path, 
                                     style_path, epochs=500)

plt.figure(figsize=(15,15))
plt.subplot(1,3,3)
plt.imshow(best)
plt.title('Style transfer Image')
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,1)
f.show_im(content,'Content Image')
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,2)
f.show_im(style,'Style Image')
plt.xticks([])
plt.yticks([])
plt.show()
