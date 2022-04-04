import os
import numpy as np
import tensorflow as tf

INPUT_SHAPE = 48, 48, 3


def encode_single_sample(img_path, label, input_shape):
    height, width, channels = input_shape
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode
    img = tf.io.decode_png(img, channels=channels)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize the image
    img = tf.image.resize(
        img, size=[height, width]
    )
    return img, label


def get_list_files(x_dir, y_dir, n_classes):
    x_list = [name.split('.jpg')[0] for name in os.listdir(x_dir) if name.endswith('.jpg')]
    y_list = [name for name in os.listdir(y_dir) if name.endswith('_exp.npy')]

    temp = {}
    for label in y_list:
        temp[label.split('_exp.npy')[0]] = np.squeeze(np.load(os.path.join(y_dir, label)))
    y = []
    for i, img_name in enumerate(x_list):
        y.append(temp[img_name])
        x_list[i] = os.path.join(x_dir, str(img_name) + '.jpg')
    x = np.array(x_list)
    y = tf.keras.utils.to_categorical(np.array(y).astype('int'), n_classes)
    return x, y


def get_dataset(x, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    dataset = (
        dataset.map(
            lambda p, l: encode_single_sample(p, l, INPUT_SHAPE),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return dataset


def get_datasets(direction, n_classes, batch_size):
    x_dir_train = direction + 'train_set/images/'
    y_dir_train = direction + 'train_set/annotations/'
    x_dir_val = direction + 'val_set/images/'
    y_dir_val = direction + 'val_set/annotations/'

    x_train, y_train = get_list_files(x_dir_train, y_dir_train, n_classes)
    x_val, y_val = get_list_files(x_dir_val, y_dir_val, n_classes)

    train_set = get_dataset(x_train, y_train, batch_size)
    val_set = get_dataset(x_val, y_val, batch_size)

    return train_set, val_set
