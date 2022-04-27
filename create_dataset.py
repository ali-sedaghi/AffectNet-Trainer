import os
import numpy as np
import tensorflow as tf


def get_list_files(x_dir, y_dir, n_classes):
    x_list = [name.split('.jpg')[0] for name in os.listdir(x_dir) if name.endswith('.jpg')]
    y_list = [name for name in os.listdir(y_dir) if name.endswith('_exp.npy')]

    temp = {}
    for i, label in enumerate(y_list):
        temp[label.split('_exp.npy')[0]] = np.squeeze(np.load(os.path.join(y_dir, label)))

    y = []
    for i, img_name in enumerate(x_list):
        y.append(temp[img_name])
        x_list[i] = os.path.join(x_dir, str(img_name) + '.jpg')

    x = np.array(x_list)
    y = tf.keras.utils.to_categorical(np.array(y).astype('int'), n_classes)
    return x, y


if __name__ == "__main__":
    DIRECTION = './data/affectnet_mini/'
    N_CLASSES = 11

    x_dir_train = DIRECTION + 'train_set/images/'
    y_dir_train = DIRECTION + 'train_set/annotations/'
    x_dir_val = DIRECTION + 'val_set/images/'
    y_dir_val = DIRECTION + 'val_set/annotations/'

    x_train, y_train = get_list_files(x_dir_train, y_dir_train, N_CLASSES)
    x_val, y_val = get_list_files(x_dir_val, y_dir_val, N_CLASSES)

    np.save(DIRECTION + 'x_train.npy', x_train, allow_pickle=False)
    np.save(DIRECTION + 'y_train.npy', y_train, allow_pickle=False)
    np.save(DIRECTION + 'x_val.npy', x_val, allow_pickle=False)
    np.save(DIRECTION + 'y_val.npy', y_val, allow_pickle=False)
