import tensorflow as tf 
import numpy as np 

def vgg_net(layers):

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    output = [vgg.get_layer(name).output for name in layers]
    model = tf.keras.Model([vgg.input], output)
    return model



def gram_matrix(input):
    """
    gram matrix is is the 2D matrix formed from 3D tensor
    """
    channel = int(input.shape[-1])
    a = tf.reshape(input, [-1, channel]) # to convert image tensor into matrix
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram/tf.cast(n, tf.float32)



class TransferModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(TransferModel, self).__init__()
        self.vgg = vgg_net(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.size_content_layer = len(content_layers)
        self.size_style_layer = len(style_layers)
        self.vgg.trainable = False


    def call(self, input):
        input = input*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(input)
        print(tf.shape(preprocessed_input))
        output = self.vgg([preprocessed_input])

        style_output, content_output = (output[:self.size_style_layer], output[self.size_style_layer: ])

        style_output = [gram_matrix(out) for out in style_output]

        content_dict = {content_name: value
                        for content_name, value in zip(self.content_layers, content_output)}

        style_dict = {style_name: value
                        for style_name, value in zip(self.style_layers, style_output)}

        return {'content': content_dict, 'style': style_dict}



def get_model():
    content_layers = ["block5_conv2"]
    style_layers = ["block1_conv1","block2_conv1","block3_conv1","block4_conv1"]
    return TransferModel(style_layers, content_layers)
