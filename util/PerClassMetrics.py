
from tensorflow import keras
import numpy as np

from util.custom_functions import plot_confusion_matrix, calc_confusion_matrix


class PerClassMetrics(keras.callbacks.Callback):

    def __init__(self, model, val_data, batch_size, model_id):
        self.model = model
        self.model_id = model_id
        self.validation_data = val_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, epoch, logs):
        import matplotlib.pyplot as plt

        x_test, y_test = self.validation_data[0], self.validation_data[1]
        nr_classes = self.validation_data[1].shape[1]
        if epoch % 200 == 0:
            calc_confusion_matrix(self.model, x_test, y_test, self.model_id)

        accs = self.cm.diagonal()

        print("\n")
        for cl in range(nr_classes):
            print(" " * 40 + "C{}: {}".format(cl, accs[cl]))

        return

    def get_data(self):
        return self._data