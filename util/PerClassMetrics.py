import numpy as np
from tensorflow import keras

from util.custom_functions import calc_confusion_matrix


class PerClassMetrics(keras.callbacks.Callback):

    def __init__(self, model, val_data, batch_size, model_id):
        self.model = model
        self.model_id = model_id
        self.validation_data = val_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, epoch, logs):
        from sklearn.metrics import classification_report

        x_test, y_test = self.validation_data[0], self.validation_data[1]

        y_predict = np.asarray(self.model.predict(x_test))

        if epoch % 200 == 0:
            calc_confusion_matrix(y_predict, y_test, self.model_id)

            y_labels_pred = np.argmax(y_predict, axis=1)
            y_labels = np.argmax(y_test, axis=1)
            print('\n')
            print(classification_report(y_labels, y_labels_pred, zero_division=0))

        # accs = self.cm.diagonal()
        #
        # print("\n")
        # for cl in range(nr_classes):
        #     print(" " * 40 + "C{}: {}".format(cl, accs[cl]))

        return

    def get_data(self):
        return self._data
