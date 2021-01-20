import tensorflow as tf
import os
import sys
sys.path.append('./utils')
from create_model import MultiClassKerasClassifier
from dataset import GenCreator


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




