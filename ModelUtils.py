import random

import dlib
import keras.backend.tensorflow_backend as tfback
import numpy as np
import tensorflow as tf

from constants import *


def _get_available_gpus():
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


def _get_random_peak_photo(data):
    subj = random.choice(list(data.subjects.values()))
    sess = random.choice(list(subj.sessions.values()))

    while sess.emotion is None:
        subj = random.choice(list(data.subjects.values()))
        sess = random.choice(list(subj.sessions.values()))

    return sess.getPeakPhoto(), sess.emotion


def _crop_photo(photo):
    photo.loadData()
    img = photo.data

    crop_img = img[0:480, 80:560]
    return dlib.resize_image(crop_img, DATA_RESOLUTION, DATA_RESOLUTION)


def _train_model_common(model, model_name, x_train, x_test, y_train, y_test):
    print("Starting training")

    model.fit(np.array(x_train).reshape((len(x_train), DATA_RESOLUTION, DATA_RESOLUTION, 1)), np.array(y_train),
              epochs=EPOCHS, batch_size=8)
    model.save("models_eval\\{}.h5".format(model_name))

    ev = model.evaluate(np.array(x_test).reshape((len(x_test), DATA_RESOLUTION, DATA_RESOLUTION, 1)), np.array(y_test))

    print("Evaluation: " + str(ev))

