import scipy
import numpy as np
import matplotlib.pyplot as plt
import librosa

from TrainingPlot import PlotLosses

from util.PerClassMetrics import PerClassMetrics


def train_model(model, model_path, x_traincnn, y_train, x_testcnn, y_test, get_emotion_label):
    cl_backs = [PlotLosses(get_emotion_label), PerClassMetrics(model, (x_testcnn, y_test), 16)]
    cnnhistory = model.fit(x_traincnn, y_train, batch_size=16, epochs=1200, validation_data=(x_testcnn, y_test), callbacks=cl_backs)
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


def reject_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def replace_outliers_by_std(data, m=3.):

    u = np.mean(data)
    s = np.std(data)
    f1 = u - m * s
    f2 = u + m * s
    dt_median = np.median(data)
    data1 = np.where(data > f1, data, f1)
    data2 = np.where(data1 < f2, data1, f2)

    return data2


def show_spectrogram(file):
    sr, x = scipy.io.wavfile.read(file)
    ## Parameters: 10ms step, 30ms window
    nstep = int(sr * 0.01)
    nwin = int(sr * 0.03)
    nfft = nwin
    window = np.hamming(nwin)
    ## will take windows x[n1:n2].  generate
    ## and loop over n2 such that all frames
    ## fit within the waveform
    nn = range(nwin, len(x), nstep)
    X = np.zeros((len(nn), nfft // 2))
    for i, n in enumerate(nn):
        xseg = x[n - nwin:n]
        z = np.fft.fft(window * xseg, nfft)
        X[i, :] = np.log(np.abs(z[:nfft // 2]))
    plt.imshow(X.T, interpolation='nearest',
               origin='lower',
               aspect='auto')
    plt.show()


def show_amplitude(file):
    data, sampling_rate = librosa.load(file)
    plt.figure(figsize=(15, 5))
    librosa.display.waveplot(data, sr=sampling_rate)
    plt.show()