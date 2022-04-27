import os
import tensorflow as tf
from models.resnet import resnet_v2
from utils.get_callbacks import get_callbacks
from utils.dataset_loader import get_datasets, INPUT_SHAPE

# Silent TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Remove memory limit on GPUs
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

# Train on multiple GPUs
strategy = tf.distribute.MirroredStrategy()

# Hyperparams
BASE_PATH = './checkpoints/'
LOAD = False
LAST_EPOCH = '000'
DEPTH = 56
N_CLASSES = 11
BATCH_SIZE = 32
EPOCHS = 100

# Train size: 287651    Val size: 3999
train_set, val_set = get_datasets(BATCH_SIZE)
print("Dataset loaded")

# Model
with strategy.scope():
    if LOAD:
        post_path = f"model_epoch{LAST_EPOCH}.hdf5"
        load_path = os.path.join(BASE_PATH, post_path)
        model = tf.keras.models.load_model(load_path)
    else:
        model = resnet_v2(
            input_shape=INPUT_SHAPE,
            depth=DEPTH,
            num_classes=N_CLASSES,
        )
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy'],
        )
print("Model loaded")

# Training
history = model.fit(
    x=train_set,
    validation_data=val_set,
    epochs=EPOCHS,
    callbacks=get_callbacks(BASE_PATH),
    initial_epoch=int(LAST_EPOCH),
)
