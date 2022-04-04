import os
import tensorflow as tf


def get_model(model, base_path, load=False, last_epoch='000'):
    if load:
        post_path = f"model_epoch{last_epoch}.hdf5"
        load_path = os.path.join(base_path, post_path)
        model = tf.keras.models.load_model(load_path)
    else:
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy'],
        )
    return model
