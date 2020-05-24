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

from DataLoader import DataLoader
from EmotionModel import EmotionModel
from ModelUtils import _crop_photo
from ModelUtils import _get_random_peak_photo
from ModelUtils import _train_model_common
from constants import *
from data_model.emotion_map import emotion_map
from data_model.photo import Photo

MODEL_NAME = "model_landmarks"

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
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=136, activation='linear'))

    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        metrics=['mae'],
        loss='mean_squared_error'
    )

    return model

def _preprocess_data(data):
    x = []
    y = []
    for subj_name, subject in data.subjects.items():
        for session_name, session in subject.sessions.items():
            if session.emotion is None:
                continue

            photos = random.sample(session.photos, min(len(session.photos), PHOTOS_PER_SESSION))
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

def _load_data_and_model():
    data = DataLoader()
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
    data, _ = _load_data_and_model()
    x_train, x_test, y_train, y_test = _preprocess_data(data)
    _train_model_common(get_model(), MODEL_NAME, x_train, x_test, y_train, y_test)

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
    #_evaluation_test()
    #_camera_test()
    #_train_model()
    #_plot_model()
    pass
