import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow import keras
import numpy as np


class PerClassMetrics(keras.callbacks.Callback):

    def __init__(self, model, val_data, batch_size):
        self.model = model
        self.validation_data = val_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, epoch, logs):
        import matplotlib.pyplot as plt

        nr_classes = self.validation_data[1].shape[1]

        x_test, y_test = self.validation_data[0], self.validation_data[1]

        y_predict = np.asarray(self.model.predict(x_test))

        true = np.argmax(y_test, axis=1)
        pred = np.argmax(y_predict, axis=1)

        self.cm = confusion_matrix(true, pred)
        self.cm_raw = self.cm
        self.cm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
        self.cm = np.round(self.cm, 2)
        self._data.append({
            'classLevelaccuracy': self.cm.diagonal(),
        })

        if epoch % 200 == 0:
            self.plot_confusion_matrix(self.cm)
            self.plot_confusion_matrix(self.cm_raw)

        accs = self.cm.diagonal()

        print("\n")
        for cl in range(nr_classes):
            print(" " * 40 + "C{}: {}".format(cl, accs[cl]))

        return



    def get_data(self):
        return self._data