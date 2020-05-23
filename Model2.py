import random

import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from DataLoader import DataLoader

from data_model.emotion_map import emotion_map
from data_model.photo import Photo
from ModelUtils import _get_random_peak_photo
from ModelUtils import _crop_photo
from constants import *
from EmotionModel import EmotionModel

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

def _preprocess_data_for_landmarks(data):
    global x_train, x_test, y_train, y_test
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
                y.append(np.array(landmarks).flatten())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, shuffle=True)
    print("Train size: {}".format(len(x_train)))
    print("Test size: {}".format(len(x_test)))
    print("Total size: {}".format(len(x)))
    return x_train, x_test, y_train, y_test


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
    _, x_test, _, y_test = _preprocess_data_for_landmarks(data)

    pred = landmark_model.predict(np.array(x_test).reshape((len(x_test), DATA_RESOLUTION, DATA_RESOLUTION, 1)))
    mae = mean_absolute_error(y_test, pred)
    print("Mean absolute error: {}".format(mae))

def _plot_model():
    _, landmark_model, _ = _load_data_and_model()
    plot_model(landmark_model, to_file="models_resources\\{}.png".format(MODEL_NAME), show_shapes=True, expand_nested=True)

if __name__ == "__main__":
    #_manual_test()
    #_evaluation_test()
    #_plot_model()
    pass
