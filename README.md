# AffectNet-Trainer
A simple bootstrap for training custom models on [AffectNet](http://mohammadmahoor.com/affectnet/) dataset on a system with multiple GPUs using [Mirrored Distributed Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy).


## Dataset

For downloading AffectNet, only Lab Managers, Professors or Students using their academic email account should fill out the request form [HERE](http://mohammadmahoor.com/affectnet-request-form/).

Dataset must be inside ```data``` folder of the project.


## Adding Models

For adding custom keras models, add a new python file inside ```models``` folder and use it inside ```main.py```.

Keras callbacks will save weights and logs inside ```checkpoints``` folder.
