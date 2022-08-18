import tensorflow
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, AUC

OPTIMIZER = Adam(learning_rate=0.005)
LOSS_FUNCTION = SparseCategoricalCrossentropy()
EVAL_METRIC = SparseCategoricalAccuracy()
BATCH_SIZE = 5

train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                          zoom_range=0.2,
                                          rotation_range=15,
                                          width_shift_range=0.05,
                                          height_shift_range=0.05)

validation_data_generator = ImageDataGenerator(rescale=1. / 255,
                                               zoom_range=0.2,
                                               rotation_range=15,
                                               width_shift_range=0.05,
                                               height_shift_range=0.05)

training_iterator = train_data_generator.flow_from_directory('Covid19-dataset/train',
                                                             class_mode='sparse',
                                                             color_mode='grayscale',
                                                             batch_size=BATCH_SIZE)
validation_iterator = validation_data_generator.flow_from_directory('Covid19-dataset/test',
                                                                    class_mode='sparse',
                                                                    color_mode='grayscale',
                                                                    batch_size=BATCH_SIZE)

sample_batch_input, sample_batch_label = training_iterator.next()
# print(list(sample_batch_input[0]), sample_batch_label[0])
# print(validation_iterator)

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

early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

model.compile(optimizer=OPTIMIZER, loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

print(model.summary())

history = model.fit(training_iterator,
                    steps_per_epoch=training_iterator.samples / BATCH_SIZE,
                    epochs=60,
                    validation_data=validation_iterator,
                    validation_steps=validation_iterator.samples / BATCH_SIZE,
                    callbacks=[early_stop])

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['sparse_categorical_accuracy'])
ax1.plot(history.history['val_sparse_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

fig.savefig('Covid19-dataset/CNN_performance/performance.png')
fig.savefig('Covid19-dataset/CNN_performance/performance.jpeg')

plt.show()
