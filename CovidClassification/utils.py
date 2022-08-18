import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def data_augmentation(directory_path, batch_size):
    data_generator = ImageDataGenerator(rescale=1.0 / 255,
                                        zoom_range=0.2,
                                        rotation_range=15,
                                        width_shift_range=0.05,
                                        height_shift_range=0.05)

    data_iterator = data_generator.flow_from_directory(directory_path,
                                                       class_mode='sparse',
                                                       color_mode='grayscale',
                                                       target_size=(256, 256),
                                                       batch_size=batch_size)
    return data_iterator


def data_loader(directory_path, batch_size):
    data_generator = ImageDataGenerator(rescale=1.0 / 255)

    data_iterator = data_generator.flow_from_directory(directory_path,
                                                       class_mode='sparse',
                                                       color_mode='grayscale',
                                                       target_size=(256, 256),
                                                       batch_size=batch_size)
    return data_iterator


def plot_performance(history, param):
    if param == 'accuracy':

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('model_performance/accuracy.png')
        # plt.show()

    elif param == 'loss':

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig('model_performance/loss.png')
        # plt.show()


def plot_heatmap(class_names, y_pred, y_test):
    """
    Function to compute a Confusion Matrix and plot a heatmap based on the matrix.
    input: class names, y-predicted, y-test (ground-truth)
    output: a PNG file of the heatmap.
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(15, 15))
    heatmap = sns.heatmap(cm, fmt='g', cmap='Blues', annot=True, ax=ax)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    heatmapfig = heatmap.get_figure()
    heatmapfig.savefig('model_performance/confusion_matrix.png')
