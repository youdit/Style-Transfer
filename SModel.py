import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
import PIL.Image

def load_img(path):
  max_dim = 512
  img = tf.io.read_file(path)
  img = tf.image.decode_image(img)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  ratio = max_dim/long_dim
  new_shape = tf.cast(shape*ratio, tf.int32)
  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, : ]
  return img

def imgshow(img, title=None):
  if len(img.shape)>3:
    img = tf.squeeze(img, axis=0)
  plt.imshow(img)
  if title!=None:
    plt.title(title)
  

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)



def vgg_net(layers):

  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  output = [vgg.get_layer(name).output for name in layers]
  model = tf.keras.Model([vgg.input], output)
  return model

def gram_matrix(input):
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


def losses(output, style_target, content_target, style_weight=1e-2, content_weight=1e4):
  style_output = output['style']
  content_output = output['content']
  style_loss = tf.add_n([tf.reduce_mean((style_output[name] - style_target[name])**2) for name in style_output.keys()])
  style_loss *= style_weight/size_style_layers

  content_loss = tf.add_n([tf.reduce_mean((content_output[name] - content_target[name])**2) for name in content_output.keys()])
  content_loss *= content_weight/size_content_layers

  loss = style_loss + content_loss
  return loss

def clip_image(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


@tf.function()
def train_step(image, optimize, model, style_target, content_target):
  with tf.GradientTape() as tape:
    output = model(image)
    loss = losses(output, style_target, content_target)
  
  grad = tape.gradient(loss ,image)
  optimize.apply_gradients([(grad, image)])
  image.assign(clip_image(image))

def train(model,  content_image , style_image, optimizer=tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1), epochs = 10, epochs_per_steps = 100):
  #content_image = load_img(content_path)
  #style_image = load_img(style_path)

  style_target = model(style_image)['style']
  content_target = model(content_image)['content']

  image = tf.Variable(content_image)

  for epoch in range(epochs):
    for step in range(epochs_per_steps):
      print(".",end="")
      train_step(image, optimizer, model,style_target, content_target)
  
  return image



content_layers = ["block5_conv2"]
style_layers = ["block1_conv1","block2_conv1","block3_conv1","block4_conv1"]
size_content_layers = len(content_layers)
size_style_layers = len(style_layers)

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

optimize = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

model = TransferModel(style_layers, content_layers )

content_image=load_img(content_path)
style_image=load_img(style_path)


image = train(model, content_image, style_image, optimizer=optimize, epochs=2, epochs_per_steps= 2)
image = tensor_to_image(image)

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1,2,1)
imgshow(content_image, 'Content Image')

plt.subplot(1,2,2)
imgshow(style_image, 'Style Image')
plt.show()

image.show()