import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras
import sys
sys.path.append('./utils')
from create_model import MultiClassKerasClassifier
from dataset import GenCreator
import matplotlib.pyplot as plt
import numpy
import tensorflow.keras.models
import os

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq=10)


if __name__ == '__main__':
    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    batch_size=8
    gc = GenCreator('/home/iamtheuserofthis/python_workspace/img_processing/jpeg_images_set', #path to Train, Test, Validation dirs
                    batch_size=batch_size,
                    target_size=150)

    train_generator, validation_generator = gc.get_data_generator()
    tf_callback = tf.keras.callbacks.TensorBoard('./logs')
    model = MultiClassKerasClassifier()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['acc'])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=5 * batch_size)

    history = model.fit_generator(
        train_generator,
        epochs=2,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[tf_callback]
    )

    model.save('models/gend_detectx')
    #
    #
    #
    #
    # model_s = tf.keras.models.load_model('/home/iamtheuserofthis/python_workspace/mod1')

    model = tf.keras.models.load_model('./models/gend_detect2')
    img_samp, lab_samp = next(validation_generator)
    res = []
    pred_labs = model(img_samp)
    print(img_samp.shape)
    #print(pred_labs)
    print(lab_samp)

    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    # imgs , labels = next(train_generator)

    # for i in range(4):
    #     plt.imshow(imgs[i])
    #     plt.title(labels[i])
    #     plt.show()




