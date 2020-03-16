import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from process import *
from Network import *
from training import *

model = get_model()
"""5:47 / 12:13￼￼￼￼￼
The BIGGEST FLAW about Breath of the Wild's DLC5:47 / 12:13￼￼￼￼￼
The BIGGEST FLAW about Breath of the Wild's DLC
    input the path of image you want as content on load_image(path) if on the desktop else use load_image_url(name, URL) to get image from net
    input the path of the image in style image on load_image(path) if on desktop else use load_image_url(name, URL) to get image from net  
"""
content_image = load_image('/home/youdit/Pictures/Kushagra.jpeg')
style_image = load_image_url('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


"""
    train will take addition argument optimizer for changing optimizer, learning rate or making custom optimizer 
"""

image = train(model, content_image, style_image, epochs=100, epochs_per_steps=2)

plt.subplot(1,3,1)
imgshow(content_image,'content image')

plt.subplot(1,3,2)
imgshow(style_image,'style image')

plt.subplot(1,3,3)
imgshow(image,'result image')

plt.show()
