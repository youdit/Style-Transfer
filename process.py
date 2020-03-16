import PIL
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

def load_image_url(name, URL):
    """
        use when we have url of the image and we need to upload it
    """
    max_dim = 512
    path = tf.keras.utils.get_file(name, URL)
    img= tf.io.read_file(path)
    img = tf.image.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    ratio = max_dim/long_dim
    new_shape = tf.cast(shape*ratio, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, : ]
    return img

def load_image(path):
    """
        Use when the image is present in our own system, We use the path of the sysytem
    """
    max_dim = 512
    img = PIL.Image.open(path)
    img = np.asarray(img)
    img = tf.convert_to_tensor(img)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    ratio = max_dim/long_dim
    new_shape = tf.cast(shape*ratio, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, : ]
    return img

def imgshow(img, title=None):
    """
        To show image using matplotlib
    """
    if len(img.shape)>3:
        img = tf.squeeze(img, axis=0)
    plt.imshow(img)
    if title!=None:
        plt.title(title)
