import matplotlib.pyplot as plt

from TrainingPlot import PlotLosses
from util.PerClassMetrics import PerClassMetrics


def train_model(model, model_path, batch, epoch, x_traincnn, y_train, x_testcnn, y_test, get_emotion_label):
    cl_backs = [PlotLosses(model_path, get_emotion_label), PerClassMetrics(model, (x_testcnn, y_test), 64, int(get_emotion_label))]
    cnnhistory = model.fit(x_traincnn, y_train, batch_size=batch, epochs=epoch, validation_data=(x_testcnn, y_test), callbacks=cl_backs)
    # Save the weights
    model.save(model_path)
    figure, axis = plt.subplots(2)
    axis[0].plot(cnnhistory.history['loss'])
    axis[0].plot(cnnhistory.history['val_loss'])
    axis[0].set_title('Loss')
    axis[0].set_ylabel('loss')
    axis[0].set_xlabel('epoch')
    axis[0].legend(['train', 'test'], loc='upper left')
    axis[1].plot(cnnhistory.history['accuracy'])
    axis[1].plot(cnnhistory.history['val_accuracy'])
    axis[1].set_title('Accuracy')
    axis[1].set_ylabel('accuracy')
    axis[1].set_xlabel('epoch')
    axis[1].legend(['train', 'test'], loc='upper left')
    plt.subplots_adjust(hspace=0.7)
    plt.show()
