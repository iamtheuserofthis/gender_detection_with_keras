import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
import logging
from tensorflow.keras.layers import Dense, Flatten
from dataset import GenCreator


class MultiClassKerasClassifier(Model):

    def __init__(self, base_model='resnet50', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        self.logger = logging.getLogger(__name__)
        self.logger.info('INIT WITH base model: %s' % base_model)

        # self.resnet_base = tf.keras.applications.ResNet50(input_shape=(150, 150, 3),
        #                                           include_top=False,
        #                                           weights='imagenet')
        self.xception_base = tf.keras.applications.Xception(weights='imagenet',  # Load weights pre-trained on ImageNet.
                                                            input_shape=(150, 150, 3),
                                                            include_top=False)
        self.xception_base.trainable = False
        # self.resnet_base = False
        self.flatten = tf.keras.layers.GlobalAvgPool2D()
        # self.flatten = tf.keras.layers.Flatten()
        self.dense01 = Dense(512, activation='relu', name="DENSE-512-RELU-1")
        self.dense02 = Dense(512, activation='relu', name="DENSE-512-RELU-2")
        self.dense1 = Dense(256, activation='relu', name="DENSE-256-RELU-1")
        self.dense2 = Dense(256, activation='relu', name="DENSE-256-RELU-2")
        self.output3 = Dense(3, activation='softmax', name="SOFTMAX-3")
        # self.resnet_preprocessor = tf.keras.applications.resnet50.preprocess_input


    def call(self, inputs, **kwargs):

        # x = self.resnet_preprocessor(inputs)
        x = self.xception_base(inputs, training=False)
        x = self.flatten(x)
        x = self.dense01(x)
        x = self.dense02(x)
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.output3(x)
        tf.summary.histogram('outputs', outputs)
        return outputs



if __name__ == '__main__':
    """
    USAGE OF MultiClassKerasClassifier
    """
    model = MultiClassKerasClassifier()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    gc = GenCreator('/home/iamtheuserofthis/python_workspace/img_processing/jpeg_images_set',
                    batch_size=32,
                    target_size=150)

    train_generator, validation_generator = gc.get_data_generator()

    model = MultiClassKerasClassifier()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'], run_eagerly=True)

    model.build(input_shape=(16, 150, 150, 3))

    model.summary()
