import glob
import time
from pathlib import Path

import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import tensorflow as tf
from pandas import Series
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split

from TrainingPlot import PlotLosses

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model_path = './checkpoints/trained_model'


def main():
    audio_files_path = "G:\\NNDatasets\\audio"
    audio_files = glob.glob("{}/**/*.wav".format(audio_files_path), recursive=True)

    x_testcnn, x_traincnn, y_test, y_train = pre_process_data(audio_files)

    model = Sequential()

    model.add(Dense(259), )
    model.add(Activation('relu'))

    model.add(Conv1D(256, 5, padding='same', ))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=8))

    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))

    # #
    # model.add(Conv1D(128, 5, padding='same',))
    # model.add(Activation('relu'))
    # model.add(Conv1D(128, 5, padding='same',))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # #
    model.add(Conv1D(256, 1, padding='same', ))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    opt = optimizers.Adam(learning_rate=0.000005)
    #opt = optimizers.Adam()
    #opt = optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.summary()

    plot_losses = PlotLosses()

    if not Path(model_path).exists():
        cnnhistory = model.fit(x_traincnn, y_train, batch_size=128, epochs=1600, validation_data=(x_testcnn, y_test), callbacks=plot_losses)
        # Save the weights
        model.save(model_path)

        plt.plot(cnnhistory.history['loss'])
        plt.plot(cnnhistory.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    else:
        # Restore the weights
        model = tf.keras.models.load_model(model_path)
        cnnhistory = model.evaluate(x_testcnn, y_test, batch_size=128)


def pre_process_data(audio_files):

    audio_mel_df_path = 'mel_df.pkl'
    audio_label_df_path = 'label_df.pkl'
    rnewdf = None

    if not Path(audio_mel_df_path).exists():

        mel_features_df, labels_df = extract_mel_features(audio_files)
        mel_features_df.fillna(0, inplace=True)

        # # -1 and 1 scaling
        for column in mel_features_df.columns:
            mel_features_df[column] = mel_features_df[column] / mel_features_df[column].abs().max()

        mel_features_df.to_pickle(audio_mel_df_path)
        labels_df.to_pickle(audio_label_df_path)
    else:
        mel_features_df = pd.read_pickle(audio_mel_df_path)
        labels_df = pd.read_pickle(audio_label_df_path)

    print(mel_features_df)
    print(labels_df)
    X_train, X_test, y_train, y_test = train_test_split(mel_features_df, labels_df, test_size=0.2)

    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    print(y_test)
    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    return x_testcnn, x_traincnn, y_test, y_train


def extract_mel_features(audio_files):
    mel_feature_df = pd.DataFrame()
    feeling_list = []
    start_time = time.time()

    for index, full_fname in enumerate(audio_files):
        file_name = Path(full_fname).name
        if file_name[6:-16] != '01' and file_name[6:-16] != '08' \
                and file_name[:2] != 'su' and file_name[:1] != 'n' and file_name[:1] != 'd' and file_name[6:-16] != '07':

            feeling_list.append(parse_fname_to_label(file_name))

            audio_bin, sample_rate = librosa.load(full_fname, duration=3, sr=22050 * 2, offset=0.5)
            sample_rate = np.array(sample_rate)
            mfccs = librosa.feature.mfcc(y=audio_bin, sr=sample_rate, n_mfcc=4, dct_type=3)
            mfccs_mean = np.mean(mfccs, axis=0)

            # fig, ax = plt.subplots()
            # # Ploting a single strip of the Me
            # img = librosa.display.specshow(mfccs[0:1], x_axis='time', ax=ax)
            # fig.colorbar(img, ax=ax)
            # ax.set(title='MFCC')
            # plt.show()

            # [float(i) for i in feature]
            # feature1=feature[:135]
            mel_feature_df = mel_feature_df.append(Series(mfccs_mean), ignore_index=True)
            #mel_feature_df.loc[index] = [mfccs]

    print("Time to extract Mel features:  %s seconds." % (time.time() - start_time))
    return mel_feature_df, pd.DataFrame(feeling_list)


# def parse_emotion_tags(audio_files):
#     start_time = time.time()
#
#     for file in audio_files:
#         # print(file)
#         file_name = Path(file).name
#         feeling_list = parse_fname(file_name)
#
#     print("Time to parse emotion tags:  %s seconds." % (time.time() - start_time))
#     return pd.DataFrame(feeling_list)


def parse_fname_to_label(file_name):
    label = ""
    if file_name[6:-16] == '01' and int(file_name[18:-4]) % 2 == 1:
        label = 'male_neutral'
    elif file_name[6:-16] == '01' and int(file_name[18:-4]) % 2 == 0:
        label = 'female_neutral'
    elif file_name[6:-16] == '02' and int(file_name[18:-4]) % 2 == 0:
        label = 'female_calm'
    elif file_name[6:-16] == '02' and int(file_name[18:-4]) % 2 == 1:
        label = 'male_calm'
    elif file_name[6:-16] == '03' and int(file_name[18:-4]) % 2 == 0:
        label = 'female_happy'
    elif file_name[6:-16] == '03' and int(file_name[18:-4]) % 2 == 1:
        label = 'male_happy'
    elif file_name[6:-16] == '04' and int(file_name[18:-4]) % 2 == 0:
        label = 'female_sad'
    elif file_name[6:-16] == '04' and int(file_name[18:-4]) % 2 == 1:
        label = 'male_sad'
    elif file_name[6:-16] == '05' and int(file_name[18:-4]) % 2 == 0:
        label = 'female_angry'
    elif file_name[6:-16] == '05' and int(file_name[18:-4]) % 2 == 1:
        label = 'male_angry'
    elif file_name[6:-16] == '06' and int(file_name[18:-4]) % 2 == 0:
        label = 'female_fearful'
    elif file_name[6:-16] == '06' and int(file_name[18:-4]) % 2 == 1:
        label = 'male_fearful'
    elif file_name[6:-16] == '07' and int(file_name[18:-4]) % 2 == 0:
        label = 'female_disgust'
    elif file_name[6:-16] == '07' and int(file_name[18:-4]) % 2 == 1:
        label = 'male_disgust'
    elif file_name[6:-16] == '08' and int(file_name[18:-4]) % 2 == 0:
        label = 'female_surprised'
    elif file_name[6:-16] == '08' and int(file_name[18:-4]) % 2 == 1:
        label = 'male_surprised'
    elif file_name[:1] == 'a':
        label = 'male_angry'
    elif file_name[:1] == 'f':
        label = 'male_fearful'
    elif file_name[:1] == 'h':
        label = 'male_happy'
    # elif file_name[:1]=='n':
    # label = 'neutral')
    elif file_name[:2] == 'sa':
        label = 'male_sad'

    return label


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
