import os
import tensorflow as tf


def get_callbacks(base_path):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(base_path, "model_epoch{epoch:03d}.hdf5"),
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        verbose=0,
    )
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(base_path, "train_info.csv"), append=True)
    callbacks = [checkpoint, csv_logger]
    return callbacks
