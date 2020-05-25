import random

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from keras.layers import *
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import keras.backend as K
import keras


from DataLoader import DataLoader
from EmotionModel import EmotionModel
from ModelUtils import _crop_photo
from ModelUtils import _get_random_peak_photo
from ModelUtils import _train_model_common
from constants import *
from data_model.emotion_map import emotion_map
from data_model.photo import Photo


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



MODEL_NAME = "model_landmarks"

low_prio_indecies = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
normal_indecies = [18, 19, 20, 23, 24, 25, 27,28,29,30,31,32,33,34,35,
                   37,38,40,41,43,44,46,47,50,51,52,55,56,57,58,59,60,61,62,63,64,65,66,67]
high_prio_indecies = [17, 21, 22, 26, 36, 39, 42, 45, 48, 49, 53, 54]

weight_lp = 1
weight_normal = 2
weight_hp = 4
total_weight = len(low_prio_indecies) * weight_lp + \
               len(normal_indecies) * weight_normal + \
               len(high_prio_indecies) * weight_hp

def get_model():
    model = Sequential()

    model.add(Conv2D(258, kernel_size=8, activation='relu', input_shape=(DATA_RESOLUTION,DATA_RESOLUTION, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(384, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(196, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())

    model.add(Dense(units=2048, activation='relu'))
    #model.add(Dropout(rate=0.1))
    model.add(Dense(units=1024, activation='relu'))
    #model.add(Dropout(rate=0.1))
    model.add(Dense(units=136, activation='linear'))

    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        metrics=['mae'],
        #loss='mean_squared_error'
        loss=_custom_loss
    )

    return model

def _apply_weight_to_tensor(input, indecies, weight):
    for i in indecies:
        #x point
        tf.multiply(input[i * 2], weight)
        #y point
        tf.multiply(input[(i * 2) + 1], weight)

def _custom_metric(yTrue, yPred):
    out = tf.Variable(0, dtype='float32')

    for i in high_prio_indecies:
        out.assign_add(tf.abs(yTrue[i * 2] - yPred[i * 2]))
        out.assign_add(tf.abs(yTrue[(i * 2) + 1] - yPred[(i * 2) + 1]))

    return out / (2 * len(high_prio_indecies))

def _custom_loss(yTrue,yPred):
    _apply_weight_to_tensor(yTrue, low_prio_indecies, weight_lp)
    _apply_weight_to_tensor(yPred, low_prio_indecies, weight_lp)

    _apply_weight_to_tensor(yTrue, normal_indecies, weight_normal)
    _apply_weight_to_tensor(yPred, normal_indecies, weight_normal)

    _apply_weight_to_tensor(yTrue, high_prio_indecies, weight_hp)
    _apply_weight_to_tensor(yPred, high_prio_indecies, weight_hp)

    yTrue = yTrue / total_weight
    yPred = yPred / total_weight

    return K.sum(K.square(yPred - yTrue))


def _preprocess_data(data):
    x = []
    y = []
    for subj_name, subject in data.subjects.items():
        for session_name, session in subject.sessions.items():
            if session.emotion is None:
                continue

            photos = random.sample(session.photos, min(len(session.photos), PHOTOS_PER_SESSION_MODEL_2))
            #photos = session.get_last_n_photos(PHOTOS_PER_SESSION)
            for photo in photos:
                crop_photo = _crop_photo(photo)
                landmarks = list(map(lambda l: [(l[0] - 80) / shrink_ratio, l[1] / shrink_ratio], photo.landmarks))
                # p = Photo(None, crop_photo, landmarks, None)
                # p.show()

                x.append(np.array(crop_photo).reshape(DATA_RESOLUTION, DATA_RESOLUTION))
                y.append(np.concatenate((np.array(landmarks).flatten(), [session.emotion])))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, shuffle=True)
    y_train_emotion = list(map(lambda n: n[136], y_train))
    y_test_emotion = list(map(lambda n: n[136], y_test))
    y_train = list(map(lambda n: n[0:136], y_train))
    y_test = list(map(lambda n: n[0:136], y_test))
    print("Train size: {}".format(len(x_train)))
    print("Test size: {}".format(len(x_test)))
    print("Total size: {}".format(len(x)))
    return x_train, x_test, y_train, y_test, y_train_emotion, y_test_emotion


def test_on_landmakrs_on_single_photo(landmark_model, emotion_model, data):
    photo, actual_emotion = _get_random_peak_photo(data)
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

def _load_data_and_model(skip_model_load = False):
    data = DataLoader()

    if skip_model_load:
        return data, None, None

    landmark_model = load_model("models_resources\\{}.h5".format(MODEL_NAME))
    emotion_model = EmotionModel(data)
    return data, landmark_model, emotion_model

def _manual_test():
    data, landmark_model, emotion_model = _load_data_and_model()

    for i in range (0, 15):
        test_on_landmakrs_on_single_photo(landmark_model, emotion_model, data)

def _evaluation_test():
    data, landmark_model, emotion_model = _load_data_and_model()
    _, x_test, _, y_test, _, y_test_emotion = _preprocess_data(data)

    pred_lm = landmark_model.predict(np.array(x_test).reshape((len(x_test), DATA_RESOLUTION, DATA_RESOLUTION, 1)))
    mae = mean_absolute_error(y_test, pred_lm)
    print("Mean absolute error: {}".format(mae))

    pred_emotion = []
    for lm in pred_lm:
        pred_emotion.append(emotion_model.predict_by_landmarks(lm.reshape(68, 2)))

    conf_mat = confusion_matrix(y_test_emotion, pred_emotion)
    acc = accuracy_score(y_test_emotion, pred_emotion)
    print("Accuracy: {}".format(acc))

    plt.imshow(conf_mat)
    plt.show()

def _plot_model():
    _, landmark_model, _ = _load_data_and_model()
    plot_model(landmark_model, to_file="models_resources\\{}.png".format(MODEL_NAME), show_shapes=True, expand_nested=True)

def _train_model():
    with tf.device('/GPU:0'):
        data, _, _ = _load_data_and_model(True)
        x_train, x_test, y_train, y_test, _, _ = _preprocess_data(data)
        _train_model_common(get_model(), MODEL_NAME, EPOCHS_MODEL_2, x_train, x_test, y_train, y_test)

def _camera_test():
    _, model, _ = _load_data_and_model()
    vid = cv2.VideoCapture(0)

    while (True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray_frame_croped = gray_frame[0:480, 80:560]
        gray_frame_croped = gray_frame[60:420, 140:500]
        gray_frame_resized = cv2.resize(gray_frame_croped, (DATA_RESOLUTION, DATA_RESOLUTION))
        data = [gray_frame_resized]

        pred = model.predict(np.array(data).reshape((len(data), DATA_RESOLUTION, DATA_RESOLUTION, 1)))
        pred = pred.reshape(68, 2)

        for t in pred:
            cv2.circle(gray_frame_resized, tuple(t), 1, (255, 255, 255), 1)

        cv2.line(frame, (80, 0), (80, 480), (255, 255, 255) , 1)
        cv2.line(frame, (560, 0), (560, 480), (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('resized', gray_frame_resized)
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #_manual_test()
    _evaluation_test()
    #_camera_test()
    #_train_model()
    #_plot_model()
    pass
