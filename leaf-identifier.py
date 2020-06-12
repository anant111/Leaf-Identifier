from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import cv2
from keras import callbacks as cbks
from keras import metrics
import argparse
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton

parser = argparse.ArgumentParser(description='Perform leaf training.')
parser.add_argument("--id", dest='testimage',
                   help='run a test on a single image.')

args = parser.parse_args()

epochs = 100
runmodel = "MNET"
testimage = args.testimage

import datetime
now = datetime.datetime.now().isoformat()
print("START",now)

logdir = "logs/%s-%s" % (runmodel, now)

#K.set_image_dim_ordering("tf")
seed = 7
np.random.seed(seed)

image_size=224

batch_size=64
lrate = 0.08

train_dir = "data/train"
test_dir = "data/test"

train_datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')


test_datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode='nearest')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

num_classes = len(train_generator.class_indices)
input_shape = train_generator.image_shape

print("CLASSES")

classes_dict = {}
for name, val in train_generator.class_indices.items():
  classes_dict[val] = name
classes = []
for i in range(0, num_classes):
  print(i, classes_dict[i])
  classes.append(classes_dict[i])

K.clear_session()

from keras.utils.data_utils import Sequence

def is_sequence(seq):
    
    return (getattr(seq, 'use_sequence_api', False)
            or set(dir(Sequence())).issubset(set(dir(seq) + ['use_sequence_api'])))


def evaluate_gen(model, generator,
                       steps=None,
                       callbacks=None,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=False,
                       verbose=0):
    model._make_test_function()
    steps_done = 0
    outs_per_batch = []
    batch_sizes = []

    use_sequence_api = is_sequence(generator)
    if steps is None:
        steps = len(generator)
        
    enqueuer = None

    try:
        if workers > 0:
            if use_sequence_api:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if use_sequence_api:
                output_generator = iter_sequence_infinite(generator)
            else:
                output_generator = generator

        if verbose == 1:
            progbar = Progbar(target=steps)
        count=0
        while steps_done < steps:
            generator_output = next(output_generator)
            if not hasattr(generator_output, '__len__'):
                raise ValueError('Output of generator should be a tuple '
                                 '(x, y, sample_weight) '
                                 'or (x, y). Found: ' +
                                 str(generator_output))
            if len(generator_output) == 2:
                x, y = generator_output
                sample_weight = None
            elif len(generator_output) == 3:
                x, y, sample_weight = generator_output
            else:
                raise ValueError('Output of generator should be a tuple '
                                 '(x, y, sample_weight) '
                                 'or (x, y). Found: ' +
                                 str(generator_output))

            if x is None or len(x) == 0:
                batch_size = 1
            elif isinstance(x, list):
                batch_size = x[0].shape[0]
            elif isinstance(x, dict):
                batch_size = list(x.values())[0].shape[0]
            else:
                batch_size = x.shape[0]
            if batch_size == 0:
                raise ValueError('Received an empty batch. '
                                 'Batches should contain '
                                 'at least one item.')

            batch_logs = {'batch': steps_done, 'size': batch_size}
            y_pred= model.predict_on_batch(x)
            success_result = K.eval(metrics.top_k_categorical_accuracy(y, y_pred,
                            k=3))
            
            steps_done += 1

            if verbose == 1:
                progbar.update(steps_done)

    finally:
        if enqueuer is not None:
            enqueuer.stop()
    return success_result


if testimage is not None:
    model = keras.models.load_model("leaf.%s.h5" % runmodel)
    test_data = ImageDataGenerator(
        rescale=1./255,
        fill_mode='nearest')

    test_gen = test_data.flow_from_directory(
        testimage,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')
    evaltest =  model.evaluate_generator(test_gen, 1)
    eval=evaluate_gen(model,test_gen,1)
    print("\nTOP-1 ACCURACY")
    for name, val in zip(model.metrics_names, evaltest):
        print(name, val)
    print("\nTOP-3 ACCURACY")
    print("acc {}\n".format(eval))
    exit(0)

from model1 import MobileNet
model = MobileNet(input_shape=input_shape, weights=None, classes=num_classes)

model.summary()

print("TRAINING PHASE")

decay = lrate/epochs

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True)
tensorboard.set_model(model)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=333,
    epochs=epochs,
    callbacks=[tensorboard])

model.save("leaf.%s.h5" % runmodel)

print("TESTING PHASE")

evaltest =  model.evaluate_generator(test_generator, 1)
for name, val in zip(model.metrics_names, evaltest):
    print(name, val)

print("END", datetime.datetime.now().isoformat())
