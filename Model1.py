import numpy as np
from keras.layers import *
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from DataLoader import DataLoader
from ModelUtils import _crop_photo
from ModelUtils import _get_random_peak_photo
from ModelUtils import _train_model_common
from constants import *
from data_model.emotion_map import emotion_map

MODEL_NAME = "model"
np.set_printoptions(precision=4, floatmode='fixed')

def get_model():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=8, activation='relu', input_shape=(DATA_RESOLUTION, DATA_RESOLUTION, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())


    model.add(Dense(units=2048, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.1))
    #model.add(Dense(units=136, activation='linear'))
    model.add(Dense(units=8, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        metrics=['accuracy'],
        loss='categorical_crossentropy'
    )

    plot_model(model, to_file="models_resources\\{}.png".format(MODEL_NAME), show_shapes=True, expand_nested=True)

    return model

def _preprocess_data(data):
    x = []
    y = []
    for subj_name, subject in data.subjects.items():
        for session_name, session in subject.sessions.items():
            if session.emotion is None:
                continue

            photos = session.get_last_n_photos(PHOTOS_PER_SESSION)
            for photo in photos:
                crop_photo = _crop_photo(photo)
                # p = Photo(None, crop_photo, landmarks, None)
                # p.show()

                x.append(np.array(crop_photo).reshape(DATA_RESOLUTION, DATA_RESOLUTION))
                y.append(session.emotion)

    y = to_categorical(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, shuffle=True)
    print("Train size: {}".format(len(x_train)))
    print("Test size: {}".format(len(x_test)))
    return x_train, x_test, y_train, y_test

def _test_model_on_single_photo(model, data):
    photo, actual_emotion = _get_random_peak_photo(data)
    crop_photo = _crop_photo(photo)
    predicted_emotion = model.predict(np.array(crop_photo).reshape((1, DATA_RESOLUTION, DATA_RESOLUTION, 1)))
    predicted_emotion = list(map(lambda n: round(n, 3), predicted_emotion[0]))
    emotion_index = np.argmax(predicted_emotion)

    print("Actual emotion: {} {}".format(actual_emotion, emotion_map[str(actual_emotion)]))
    print("Predicted emotion list: {}".format(str(predicted_emotion)))
    print("Predicted emotion: {} {} score: {}".format(emotion_index, emotion_map[str(float(emotion_index))], predicted_emotion[emotion_index]))
    photo.show(showLandmarks = False)

def _load_data_and_model():
    data = DataLoader()
    model = load_model("models_resources\\{}.h5".format(MODEL_NAME))
    return data, model

def _manual_test():
    data, model = _load_data_and_model()
    for i in range (0, 15):
        _test_model_on_single_photo(model, data)

def _evaluation_test():
    data, model = _load_data_and_model()
    _, x_test, _, y_test = _preprocess_data(data)
    pred = model.predict(np.array(x_test).reshape((len(x_test), DATA_RESOLUTION, DATA_RESOLUTION, 1)))
    pred = list(map(lambda n: np.argmax(n),pred))
    y_test = list(map(lambda n: np.argmax(n), y_test))
    conf_mat = confusion_matrix(y_test, pred)

    acc = accuracy_score(y_test, pred)
    print("Accuracy: {}".format(acc))

    plt.imshow(conf_mat)
    plt.show()

def _plot_model():
    _, model = _load_data_and_model()
    plot_model(model, to_file="models_resources\\{}.png".format(MODEL_NAME), show_shapes=True, expand_nested=True)

def _train_model():
    data, _ = _load_data_and_model()
    x_train, x_test, y_train, y_test = _preprocess_data(data)
    _train_model_common(get_model(), MODEL_NAME, x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    #_manual_test()
    #_evaluation_test()
    _train_model()
    #_plot_model()
    pass