import tensorflow as tf

def create_celeba_dataset(data_dir, validation_split, batch_size):
    
    data_dir = data_dir
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, validation_split= validation_split)   
    train_generator = image_generator.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')
    
    return train_generator