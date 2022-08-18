import tensorflow
import numpy as np
from sklearn.metrics import classification_report

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, AUC

from utils import *
from NN_model import build_model

# defining basic hyperparameters
OPTIMIZER = Adam(learning_rate=0.008)
EPOCHS = 60
TRAINING_BATCH_SIZE = 32
VAL_BATCH_SIZE = 12

# loading the datasets and augmenting them based on the function in utils.py
training_iterator = data_augmentation('Covid19-dataset/train', TRAINING_BATCH_SIZE)
validation_iterator = data_loader('Covid19-dataset/test', VAL_BATCH_SIZE)

# defining the model based on the build_model() function imported from NN_model.py
XRay_Classifier = build_model()
XRay_Classifier.compile(optimizer=OPTIMIZER,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
# print(XRay_Classifier.summary())

# defining the early_stop callback function to be passed into the .fit() function and training the model
early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='loss',
                                                      patience=5)
history = XRay_Classifier.fit(training_iterator,
                              steps_per_epoch=training_iterator.samples / TRAINING_BATCH_SIZE,
                              epochs=EPOCHS,
                              validation_data=validation_iterator,
                              validation_steps=validation_iterator.samples / VAL_BATCH_SIZE,
                              callbacks=[early_stop])

plot_performance(history, 'accuracy')
plot_performance(history, 'loss')

score = XRay_Classifier.evaluate(validation_iterator, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

X_test, y_test = validation_iterator.next()

y_pred = XRay_Classifier.predict(X_test, steps=validation_iterator.samples/VAL_BATCH_SIZE)
y_pred = np.argmax(y_pred, axis=1)
true_classes = validation_iterator.classes
class_labels = list(validation_iterator.class_indices.keys())

print(classification_report(y_test, y_pred, target_names=class_labels))
plot_heatmap(class_labels, y_pred, y_test)
