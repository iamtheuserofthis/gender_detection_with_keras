import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
Before using this script the datasets must be created in the following manner:
            /Root
                |-----|Train/--------|female/
                |     |              |male/
                |     |
                |-----|Test/---------|female/
                |     |              |male/
                |     |
                |-----|Validation/---|female/
                                     |male/

'''


class GenCreator:

    def __init__(self, location, batch_size, target_size):
        self.train_dir = os.path.join(location, 'Train')
        self.validation_dir = os.path.join(location, 'Validation')
        self.test_dir = os.path.join(location, 'Test')
        self.train_datagen = ImageDataGenerator(rescale=1. / 255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True, )
        self.batch_size = batch_size
        self.target_size = target_size
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

    def get_data_generator(self, sets=2):
        train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.target_size, self.target_size),
            batch_size=self.batch_size,
        # class_mode='binary'
        )
        validation_generator = self.test_datagen.flow_from_directory(self.validation_dir,
                                                                     target_size=(self.target_size, self.target_size),
                                                                     # class_mode='binary'
                                                                     )
        test_generator = self.test_datagen.flow_from_directory(self.test_dir,
                                                               target_size=(self.target_size, self.target_size),
                                                               # class_mode='binary'
                                                               )
        if sets == 2:
            return train_generator, validation_generator
        else:
            return train_generator, validation_generator,test_generator
