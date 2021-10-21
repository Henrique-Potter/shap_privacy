import glob
from pathlib import Path

import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import tensorflow as tf
from matplotlib.pyplot import specgram

from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.keras import regularizers
import os

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def main():
    audio_files_path = "G:\\NNDatasets\\audio"
    audio_files = glob.glob("{}/**/*.wav".format(audio_files_path), recursive=True)

    feeling_list = []

    parse_emotion_tags(audio_files, feeling_list)

    labels = pd.DataFrame(feeling_list)

    mel_features_df = pd.DataFrame(columns=['feature'])
    extract_mel_features(audio_files, mel_features_df)

    print(feeling_list)
    print(mel_features_df)

    print(mel_features_df[:5])

    df3 = pd.DataFrame(mel_features_df['feature'].values.tolist())
    labels = pd.DataFrame(feeling_list)

    newdf = pd.concat([df3, labels], axis=1)
    newdf = newdf[:1879]
    rnewdf = newdf.rename(index=str, columns={'0': "label"})

    print(rnewdf[:5])

    from sklearn.utils import shuffle
    rnewdf = shuffle(newdf)

    rnewdf = rnewdf.fillna(0)

    # rnewdf.drop(rnewdf[rnewdf == 50].index, inplace=True)

    newdf1 = np.random.rand(len(rnewdf)) < 0.8
    train = rnewdf[newdf1]
    test = rnewdf[~newdf1]

    trainfeatures = train.iloc[:, :-1]
    trainlabel = train.iloc[:, -1:]
    testfeatures = test.iloc[:, :-1]
    testlabel = test.iloc[:, -1:]

    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder

    X_train = np.array(trainfeatures)
    y_train = np.array(trainlabel)
    X_test = np.array(testfeatures)
    y_test = np.array(testlabel)

    lb = LabelEncoder()

    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    print(y_test)

    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    model = Sequential()

    model.add(Conv1D(256, 5, padding='same',
                     input_shape=(216, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    # model.add(Conv1D(128, 5,padding='same',))
    # model.add(Activation('relu'))
    # model.add(Conv1D(128, 5,padding='same',))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    opt = optimizers.Adam(learning_rate=0.1)
    opt = optimizers.RMSprop(lr=0.00001, decay=1e-6)

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    cnnhistory = model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))

    plt.plot(cnnhistory.history['loss'])
    plt.plot(cnnhistory.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



def extract_mel_features(audio_files, df):
    bookmark = 0
    for index, y in enumerate(audio_files):
        if audio_files[index][6:-16] != '01' and audio_files[index][6:-16] != '07' and audio_files[index][6:-16] != '08' \
                and audio_files[index][:2] != 'su' and audio_files[index][:1] != 'n' and audio_files[index][:1] != 'd':

            audio_bin, sample_rate = librosa.load(y, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=audio_bin, sr=sample_rate, n_mfcc=13), axis=0)
            feature = mfccs

            # [float(i) for i in feature]
            # feature1=feature[:135]
            df.loc[bookmark] = [feature]
            bookmark = bookmark + 1


def parse_emotion_tags(audio_files, feeling_list):

    for file in audio_files:
        # print(file)
        file_name = Path(file).name

        if file_name[6:-16] == '02' and int(file_name[18:-4]) % 2 == 0:
            feeling_list.append('female_calm')
        elif file_name[6:-16] == '02' and int(file_name[18:-4]) % 2 == 1:
            feeling_list.append('male_calm')
        elif file_name[6:-16] == '03' and int(file_name[18:-4]) % 2 == 0:
            feeling_list.append('female_happy')
        elif file_name[6:-16] == '03' and int(file_name[18:-4]) % 2 == 1:
            feeling_list.append('male_happy')
        elif file_name[6:-16] == '04' and int(file_name[18:-4]) % 2 == 0:
            feeling_list.append('female_sad')
        elif file_name[6:-16] == '04' and int(file_name[18:-4]) % 2 == 1:
            feeling_list.append('male_sad')
        elif file_name[6:-16] == '05' and int(file_name[18:-4]) % 2 == 0:
            feeling_list.append('female_angry')
        elif file_name[6:-16] == '05' and int(file_name[18:-4]) % 2 == 1:
            feeling_list.append('male_angry')
        elif file_name[6:-16] == '06' and int(file_name[18:-4]) % 2 == 0:
            feeling_list.append('female_fearful')
        elif file_name[6:-16] == '06' and int(file_name[18:-4]) % 2 == 1:
            feeling_list.append('male_fearful')
        elif file_name[:1] == 'a':
            feeling_list.append('male_angry')
        elif file_name[:1] == 'f':
            feeling_list.append('male_fearful')
        elif file_name[:1] == 'h':
            feeling_list.append('male_happy')
        # elif file_name[:1]=='n':
        # feeling_list.append('neutral')
        elif file_name[:2] == 'sa':
            feeling_list.append('male_sad')


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


if __name__ == '__main__':
    main()
