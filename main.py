import tensorflow as tf
from models.resnet import resnet_v2
from utils.get_model import get_model
from utils.get_callbacks import get_callbacks
from utils.dataset_loader import get_datasets, INPUT_SHAPE

physical_devices = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

DATA_DIR = './data/affectnet_mini/'
BASE_PATH = './checkpoints/'

LOAD = False
LAST_EPOCH = '000'

DEPTH = 56
CLASSES = 11
BATCH_SIZE = 4
EPOCHS = 100

train_set, val_set = get_datasets(DATA_DIR, CLASSES, BATCH_SIZE)

model = resnet_v2(
    input_shape=INPUT_SHAPE,
    depth=DEPTH,
    num_classes=CLASSES
)

compiled_model = get_model(model, BASE_PATH, load=LOAD, last_epoch=LAST_EPOCH)

history = compiled_model.fit(
    x=train_set,
    validation_data=val_set,
    epochs=EPOCHS,
    callbacks=get_callbacks(BASE_PATH),
    initial_epoch=int(LAST_EPOCH),
)
