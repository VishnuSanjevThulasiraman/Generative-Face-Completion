import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, losses

from tsv_img_ds import create_celeba_dataset
from tsv_gan import load_gen_disc, model_save
from tsv_gan import train_model
from tsv_gan import results

batch_size = 64

train_generator = create_celeba_dataset('Users/tsanjevvishnu/Downloads/ica', 0.95, batch_size = batch_size)

model_loaded = load_gen_disc('/Users/tsanjevvishnu/Downloads/ica/model/generator_6',
                             '/Users/tsanjevvishnu/Downloads/ica/model/discriminator_6',)

train_model(training_data = training_generator, 
            model_loaded = model_loaded, 
            epochs = 1, 
            DEBUG = True)

results(7)

model_save('/Users/tsanjevvishnu/Downloads/ica/model/generator_sanjev',
           '/Users/tsanjevvishnu/Downloads/ica/model/discriminator_sanjev')