import tensorflow
import numpy as np
from sklearn.metrics import classification_report

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, AUC

from utils import data_augmentation, plot_performance
from NN_model import build_model

# defining basic hyperparameters
OPTIMIZER = Adam(learning_rate=0.005)
LOSS_FUNCTION = SparseCategoricalCrossentropy()
EVAL_METRIC = SparseCategoricalAccuracy()
BATCH_SIZE = 5

# loading the datasets and augmenting them based on the function in utils.py
training_iterator = data_augmentation('Covid19-dataset/train')
validation_iterator = data_augmentation('Covid19-dataset/test')
sample_batch_input, sample_batch_label = training_iterator.next()

# defining the model based on the build_model() function imported from NN_model.py
XRay_Classifier = build_model()
XRay_Classifier.compile(optimizer=OPTIMIZER, loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
# print(XRay_Classifier.summary())

# defining the early_stop callback function to be passed into the .fit() function and training the model
early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
history = XRay_Classifier.fit(training_iterator,
                              steps_per_epoch=training_iterator.samples / BATCH_SIZE,
                              epochs=60,
                              validation_data=validation_iterator,
                              validation_steps=validation_iterator.samples / BATCH_SIZE,
                              callbacks=[early_stop])

plot_performance(history, 'accuracy')
plot_performance(history, 'loss')

score = XRay_Classifier.evaluate(validation_iterator, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

y_pred = XRay_Classifier.predict(validation_iterator.samples)
y_pred = np.argmax(y_pred, axis=1)
class_names = ['Spruce/Fir', 'Lodgepole Pine',
               'Ponderosa Pine', 'Cottonwood/Willow',
               'Aspen', 'Douglas-fir', 'Krummholz']

print(classification_report(validation_iterator.labels, y_pred, target_names=class_names))
plot_heatmap(class_names, y_pred, validation_iterator)