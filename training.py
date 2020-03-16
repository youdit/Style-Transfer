import tensorflow as tf
import numpy as np 


def losses(output, style_target, content_target, style_weight=1e-2, content_weight=1e4):
    """
    loss function we use in our style transfer
    Input: Arguments
        output: dictionary of dictionary it contain the value of 3D tensor of vgg19 at each conv layer {'content':Content_dictionary, 'style': Style_dictionary}
                content_dictionary conatin the 3D tensor of image on each content layer, style_dictionary conatin the 3D tensor of image on each style_layer in vgg19
        style_target: dictionary contain the 3D tensor style layer when style image is passed in model 
        content_target: dictionary contain the 3D tensor of content layer when contetn image is passed in model
        style_weight, content_weight: initial value
    
    return : 
        loss = total loss in our model
    
    """
    style_output = output['style']
    content_output = output['content']
    style_loss = tf.add_n([tf.reduce_mean((style_output[name] - style_target[name])**2) for name in style_output.keys()])
    style_loss *= style_weight/len(["block1_conv1","block2_conv1","block3_conv1","block4_conv1"])#size_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_output[name] - content_target[name])**2) for name in content_output.keys()])
    content_loss *= content_weight/len(["block5_conv2"])#size_content_layers

    loss = style_loss + content_loss
    return loss


def clip_image(image):
    """
        To clip the pixel value of image 3D tensor between [0,1]

    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)



@tf.function()
def train_step(image, optimize, model, style_target, content_target):
    """
    step at which the gradient descend occur
    Input: Arguments
        image: 3D tensor on which the image to create
        optimize : optimizer to use
        model: model to use in our training usually vgg19
        style_target: style image we have to super impose 
        contnet_target: image on which we want to impose the style
    
    it will perform gradient descend on image tensor using loss function
    """
    with tf.GradientTape() as tape:
        output = model(image)
        loss = losses(output, style_target, content_target)
    
    grad = tape.gradient(loss ,image)
    optimize.apply_gradients([(grad, image)])
    image.assign(clip_image(image))


# call only train function to start training process 
def train(model,  content_image , style_image, optimizer=tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1), epochs = 10, epochs_per_steps = 100):
    """
    Input: Arguments
        model : model to be used in style transfer by default we use a pre-train vgg19 model
        content_image : image on which style to impose, a 3D tensor between [0,1]
        style_image : image whose style to use, a 3D tensor between [0,1]
        optimizer : optimizer to use in Gradient descent, We use Adam by default 
        epochs : # of epochs to go through on data
        epochs_per_steps :  # of steps in each epochs
    output:
        image : a 3D tensor between [0,1] which contain our required image
    """
   

    style_target = model(style_image)['style']
    content_target = model(content_image)['content']

    image = tf.Variable(content_image)

    for epoch in range(epochs):
        for step in range(epochs_per_steps):
            print(".",end="")
            train_step(image, optimizer, model,style_target, content_target)
    
    return image

