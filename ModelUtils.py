import random

import dlib
import keras.backend.tensorflow_backend as tfback
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from DataLoader import DataLoader
from EmotionModel import EmotionModel
from constants import *
from data_model.emotion_map import emotion_map
from data_model.photo import Photo


def _get_available_gpus():
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


x_train, x_test, y_train, y_test = None, None, None, None



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



def _train_model():
    print("Starting training")
    model = get_model()
    model.fit(np.array(x_train).reshape((len(x_train), DATA_RESOLUTION, DATA_RESOLUTION, 1)), np.array(y_train),
              epochs=EPOCHS)
    model.save('model.h5')

    ev = model.evaluate(np.array(x_test).reshape((len(x_test), DATA_RESOLUTION, DATA_RESOLUTION, 1)), np.array(y_test))

    print("Evaluation: " + str(ev))
    print("Prediction: ")

    print(model.predict(np.array(x_test[22]).reshape((1, DATA_RESOLUTION, DATA_RESOLUTION, 1))))
