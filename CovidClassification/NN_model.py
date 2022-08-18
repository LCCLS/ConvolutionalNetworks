from tensorflow.keras import Sequential
from tensorflow.keras import layers


def build_model():
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(256, 256, 1)))

    model.add(layers.Conv2D(3, 5, strides=3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))

    model.add(layers.Conv2D(4, 3, strides=1, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

    model.add(layers.Conv2D(3, 3, strides=1, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(3, activation='softmax'))

    return model
