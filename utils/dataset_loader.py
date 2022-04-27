import numpy as np
import tensorflow as tf

INPUT_SHAPE = 224, 224, 3


def encode_single_sample(img_path, label, input_shape):
    height, width, channels = input_shape
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode
    img = tf.io.decode_png(img, channels=channels)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize the image
    img = tf.image.resize(img, size=[height, width])
    return img, label


def load_np_arrays():
    direction = './data/affectnet_mini/'
    x_dir_train = direction + 'x_train.npy'
    y_dir_train = direction + 'y_train.npy'
    x_dir_val = direction + 'x_val.npy'
    y_dir_val = direction + 'y_val.npy'

    mode = 'r'
    x_train = np.load(x_dir_train, mmap_mode=mode)
    y_train = np.load(y_dir_train, mmap_mode=mode)
    x_val = np.load(x_dir_val, mmap_mode=mode)
    y_val = np.load(y_dir_val, mmap_mode=mode)

    return (x_train, y_train), (x_val, y_val)


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


def get_datasets(batch_size):
    (x_train, y_train), (x_val, y_val) = load_np_arrays()
    train_set = get_dataset(x_train, y_train, batch_size)
    val_set = get_dataset(x_val, y_val, batch_size)

    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_set = train_set.with_options(options)
    val_set = val_set.with_options(options)

    return train_set, val_set
