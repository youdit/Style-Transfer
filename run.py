import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from process import *
from Network import *
from training import *

model = get_model()

content_image = load_image('/home/youdit/Pictures/Kushagra.jpeg')
style_image = load_image_url('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

image = train(model, content_image, style_image, epochs=100, epochs_per_steps=2)

plt.subplot(1,3,1)
imgshow(content_image,'content image')

plt.subplot(1,3,2)
imgshow(style_image,'style image')

plt.subplot(1,3,3)
imgshow(image,'result image')

plt.show()
