from keras.models import Sequential
from keras.layers import *
from keras.utils import plot_model
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as tfback
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from data_model.photo import Photo
from DataLoader import DataLoader
from sklearn.model_selection import train_test_split
import random
import numpy as np
import dlib
from EmotionModel import EmotionModel
from data_model.emotion_map import emotion_map
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import to_categorical

model = dlib.cnn_face_detection_model_v1('resources/mmod_human_face_detector.dat')

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

#PHOTOS_PER_SESSION = 7
#EPOCHS = 15
PHOTOS_PER_SESSION = 3
EPOCHS = 3
DATA_RESOLUTION = 160
shrink_ratio = 480 / DATA_RESOLUTION

data = DataLoader()
emotion_model = EmotionModel(data)
x_train, x_test, y_train, y_test = None, None, None, None


def _get_random_photo():
    subj = random.choice(list(data.subjects.values()))
    sess = random.choice(list(subj.sessions.values()))
    pic = random.choice(sess.photos)
    return pic


def _get_random_peak_photo():
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


def get_model():
    model = Sequential()

    #model.add(Conv2D(258, kernel_size=8, activation='relu', input_shape=(DATA_RESOLUTION,DATA_RESOLUTION, 1)))
    model.add(Conv2D(64, kernel_size=8, activation='relu', input_shape=(DATA_RESOLUTION, DATA_RESOLUTION, 1)))
    model.add(MaxPooling2D(pool_size=2))
    #model.add(Conv2D(384, kernel_size=5, activation='relu'))
    model.add(Conv2D(128, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    #model.add(Conv2D(196, kernel_size=5, activation='relu'))
    model.add(Conv2D(64, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())


    model.add(Dense(units=2048, activation='relu'))
    #model.add(Dropout(rate=0.1))
    model.add(Dense(units=1024, activation='relu'))
    #model.add(Dropout(rate=0.1))
    #model.add(Dense(units=136, activation='linear'))
    model.add(Dense(units=8, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        #metrics=['mae'],
        metrics=['accuracy'],
        #loss='mean_squared_error'
        #loss='mean_absolute_error',
        loss='categorical_crossentropy'
    )

    plot_model(model, show_shapes=True, expand_nested=True)

    return model


def _preprocess_data():
    global x_train, x_test, y_train, y_test
    x = []
    y = []
    for subj_name, subject in data.subjects.items():
        for session_name, session in subject.sessions.items():
            if session.emotion is None:
                continue

            #photos = random.sample(session.photos, min(len(session.photos), PHOTOS_PER_SESSION))
            photos = session.get_last_n_photos(PHOTOS_PER_SESSION)
            for photo in photos:
                crop_photo = _crop_photo(photo)
                landmarks = list(map(lambda l: [(l[0] - 80) / shrink_ratio, l[1] / shrink_ratio], photo.landmarks))
                # p = Photo(None, crop_photo, landmarks, None)
                # p.show()

                x.append(np.array(crop_photo).reshape(DATA_RESOLUTION, DATA_RESOLUTION))
                #y.append(np.array(landmarks).flatten())
                y.append(session.emotion)

    y = to_categorical(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    print("Train size: {}".format(len(x_train)))
    print("Test size: {}".format(len(x_test)))


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

def test_on_single_photo(landmark_model):
    photo, actual_emotion = _get_random_peak_photo()
    crop_photo = _crop_photo(photo)
    original_landmarks = photo.landmarks
    shrinked_landmarks = list(map(lambda l: [(l[0] - 80) / shrink_ratio, l[1]/shrink_ratio] , original_landmarks))
    # shrinked_landmarks = None
    predicted_landmarks = landmark_model.predict(np.array(crop_photo).reshape((1, DATA_RESOLUTION, DATA_RESOLUTION, 1)))
    predicted_landmarks = predicted_landmarks.reshape(68, 2)
    print("Actual emotion: {} {}".format(actual_emotion, emotion_map[str(actual_emotion)]))
    predicted_emotion = emotion_model.predict_by_landmarks(predicted_landmarks)
    print("Predicted emotion: {} {}".format(predicted_emotion, emotion_map[str(predicted_emotion)]))
    test_photo = Photo(None, crop_photo, predicted_landmarks, None)
    test_photo.show(overlap_landmarks=shrinked_landmarks)


def _test_model():
    model = load_model('model.h5')
    pred = model.predict(np.array(x_test).reshape((len(x_test), DATA_RESOLUTION, DATA_RESOLUTION, 1)))
    conf_mat = confusion_matrix(y_test, pred)
    plt.imshow(conf_mat)
    #print("Evaluation: " + str(ev))

    #for i in range (0, 15):
    #    test_on_single_photo(model)


_preprocess_data()
#_train_model()
_test_model()
