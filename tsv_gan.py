# ## Generator

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, losses


def generator_model():
    
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(3,(5,5), strides = (2,2) , padding = "same"  ,input_shape = (224,224,3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)
    
    model.add(tf.keras.layers.Conv2D(16,(5,5), strides = (2,2) , padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)
    
    model.add(tf.keras.layers.Conv2D(32,(5,5), strides = (2,2) , padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)

    model.add(tf.keras.layers.Conv2D(64,(5,5), strides = (2,2) , padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)

    model.add(tf.keras.layers.Conv2D(128,(5,5), strides = (2,2) , padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)
    
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(7*7*128 , use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Reshape((7,7,128)))

    model.add(tf.keras.layers.UpSampling2D())  #14X14
    model.add(tf.keras.layers.Conv2D(128 , (5,5) , strides = (1,1), padding = "same")) #28x28
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.UpSampling2D())  #28X28
    model.add(tf.keras.layers.Conv2D(64 , (5,5) , strides = (1,1), padding = "same")) #28x28
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.UpSampling2D())  #56X56
    model.add(tf.keras.layers.Conv2D(32 , (5,5) , strides = (1,1), padding = "same")) #56x56
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
        
    model.add(tf.keras.layers.UpSampling2D())  #112X112
    model.add(tf.keras.layers.Conv2D(16 , (5,5) , strides = (1,1), padding = "same")) #112x112
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
        
    model.add(tf.keras.layers.UpSampling2D())  #224X224
    model.add(tf.keras.layers.Conv2D(3 , (5,5) , strides = (1,1), padding = "same" , activation = "tanh")) #28x28

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32,(5,5), strides = (2,2) , padding = "same"  ,input_shape = (224,224,3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2D(64,(5,5), strides = (2,2) , padding = "same"  ,))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    
    model.add(tf.keras.layers.Conv2D(128,(5,5), strides = (2,2) , padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2D(256,(5,5), strides = (2,2) , padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1028,activation = "relu"))
    model.add(tf.keras.layers.Dense(1))
    
    
    return model

# ## Losses and optimizers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    
    return total_loss


def generator_loss(real_output, fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    #return tf.norm(real_output - fake_output)


def optimizers_and_checkpoint(gen_opt = tf.keras.optimizers.Adam(1e-4), disc_opt = tf.keras.optimizers.Adam(1e-4))
    generator_optimizer = gen_opt
    discriminator_optimizer = disc_opt
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)




import time


@tf.function
def train_step(images):
    if DEBUG :
        tf.print("Inside train_step")

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(images, training=True)
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)
      assert generated_images.shape == images.shape, "Shape of images and generated images do not match"

      gen_loss = generator_loss(real_output, fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
            
      tf.print("gen_loss : ", gen_loss, "    ", "disc_loss : ", disc_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    if DEBUG :
        tf.print("Gradients calculated and applied")


def train(dataset, epochs, display_batch_number = False, DEBUG = False):
  for epoch in range(epochs):
    start = time.time()

    for j in range(len(dataset)):
      if display_batch_number :
          print("Going through batch number : " + str(j+1))
      train_step(dataset[j][0])

    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for this epoch {} is {} sec'.format(epoch + 1, time.time()-start))  


def generate_and_save_images(model, epoch, test_input):

  predictions = model(test_input, training=False)

  print(predictions[0].shape)    

  fig = plt.figure(figsize=(8,8))

  for i in range(predictions[:16].shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[0])
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


# ## Loading the already trained generator and discriminator

def load_gen_disc(gen_dir, disc_dir, gen_opt = None, disc_output = None):

    generator = tf.keras.models.load_model(gen_dir)
    discriminator = tf.keras.models.load_model(disc_dir)
    model_loaded = True
    
    if gen_opt and disc_output :
        optimizers_and_checkpoint(gen_opt = gen_opt, disc_opt = disc_output)
    else:
        optimizers_and_checkpoint()
    
    return model_loaded


# ## Training the model

def train_model(training_data, model_loaded, epochs, DEBUG = False):
    
    if model_loaded == True:
        train(training_data, train_generator, epochs)
    
    if(model_loaded == False):
        generator = generator_model()
        discriminator = make_discriminator_model()
        
        if DEBUG :
            generator.summary()
            discriminator.summary()
        
        train(training_data, train_generator, epochs)


# ## Results

def compare_results(seed, seed_number):    
    
    aa = tf.squeeze(seed[seed_number]).numpy()
    print(aa.shape)

    h = [75, 100]
    v = [100, 125]
    for i in range(v[0], v[1]):
      for j in range(h[0], h[1]):
        aa[i][j] = -1

    c = tf.convert_to_tensor(aa)
    c = tf.reshape(c, [1, 224, 224, 3])
    c.shape

    o = generator(c, training = False)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(7, 8))
    
    ax1.set_title("Incomplete face")
    ax1.imshow(aa)
    ax2.set_title("Generated Face by our Model")
    ax2.imshow(tf.squeeze(o))
    ax3.set_title("Complete Face")
    ax3.imshow(tf.squeeze(seed[seed_number]).numpy())
    ax4.set_title("Face Generated when \n the complete image is input")
    ax4.imshow(tf.squeeze(generator(seed[seed_number].reshape(1,224,224,3))))
    fig.savefig('/Users/tsanjevvishnu/Downloads/ica/gime8/epoch'+str(8)+'-image_number'+str(seed_number))


def results(seed_number)
    for image_batch in training_generator :
        seed = image_batch
    
    seed_number = seed_number
    assert seed_number < 10, "seed_number greater than size of the seed"
    
    compare_results(seed = seed, 
                    seed_number = seed_number)



# # Script

# ## Saving the model

def model_save(generator_name, discriminator_name):
    generator.save(generator_name)
    discriminator.save(discriminator_name)

