import glob

import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import time

from numpy.random import seed
from tqdm import tqdm

seed(42)# keras seed fixing import


def pre_process_data(audio_files_path, get_emotion_label):

    audio_files = glob.glob("{}/**/*.wav".format(audio_files_path), recursive=True)

    audio_mel_df_path = './data/audio_data_df.pkl'

    if not Path(audio_mel_df_path).exists():
        print("Loading files and extracting features.")

        audiol_features_df = extract_mel_features(audio_files)
        audiol_features_df.fillna(0, inplace=True)

        # # # -1 and 1 scaling
        # for column in audiol_features_df.columns:
        #     audiol_features_df[column] = audiol_features_df[column] / audiol_features_df[column].abs().max()

        audiol_features_df.to_pickle(audio_mel_df_path)
    else:
        print("Loading pre_extracted features from file.")
        audiol_features_df = pd.read_pickle(audio_mel_df_path)

    print(audiol_features_df)

    if get_emotion_label:
        X_train, X_test, y_train, y_test = train_test_split(audiol_features_df.iloc[:, :-2], audiol_features_df.iloc[:, -2:-1],
                                                            test_size=0.2, random_state=5)
    else:
        X_train, X_test, y_train, y_test = train_test_split(audiol_features_df.iloc[:, :-2], audiol_features_df.iloc[:, -1:],
                                                            test_size=0.2, random_state=5)

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


def pre_process_fseer_data(audio_files_path, get_emotion_label):

    audio_files = glob.glob("{}/**/*.wav".format(audio_files_path), recursive=True)

    audio_mel_df_path = 'data/fser_audio_data_df.pkl'

    if not Path(audio_mel_df_path).exists():
        print("Loading files and extracting features.")

        audiol_features_df = extract_2d_mel_features(audio_files)
        audiol_features_df.fillna(0, inplace=True)

        # # # -1 and 1 scaling
        # for column in audiol_features_df.columns:
        #     audiol_features_df[column] = audiol_features_df[column] / audiol_features_df[column].abs().max()

        audiol_features_df.to_pickle(audio_mel_df_path)
    else:
        print("Loading pre_extracted features from file.")
        audiol_features_df = pd.read_pickle(audio_mel_df_path)

    print(audiol_features_df)

    if get_emotion_label:
        X_train, X_test, y_train, y_test = train_test_split(audiol_features_df.iloc[:, :-2], audiol_features_df.iloc[:, -2:-1],
                                                            test_size=0.2, random_state=6)
    else:
        X_train, X_test, y_train, y_test = train_test_split(audiol_features_df.iloc[:, :-2], audiol_features_df.iloc[:, -1:],
                                                            test_size=0.2, random_state=6)

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

    return x_traincnn, x_testcnn, y_train, y_test


def extract_mel_features(audio_files):
    from pandas import Series

    audio_features_df = pd.DataFrame()
    audio_labels_df = pd.DataFrame()
    start_time = time.time()

    for index, full_fname in tqdm(enumerate(audio_files)):
        file_name = Path(full_fname).name
        if file_name[6:-16] != '01' and file_name[6:-16] != '08' \
                and file_name[:2] != 'su' and file_name[:1] != 'n' and file_name[:1] != 'd' and file_name[6:-16] != '07':

            emo_label = parse_fname_to_emo_label(file_name)
            gen_label = parse_fname_to_gender_label(file_name)

            if emo_label is None:
                t =232323

            audio_bin, sample_rate = librosa.load(full_fname, duration=3, sr=22050 * 2, offset=1.1)
            sample_rate = np.array(sample_rate)
            mfccs = librosa.feature.mfcc(y=audio_bin, sr=sample_rate, n_mfcc=4, dct_type=3)
            mfccs_mean = np.mean(mfccs, axis=0)

            #plot_mfccs(full_fname)

            audio_labels_df = audio_labels_df.append(Series([emo_label, gen_label]), ignore_index=True)
            audio_features_df = audio_features_df.append(Series(mfccs_mean), ignore_index=True)

    print("Time to extract Mel features:  %s seconds." % (time.time() - start_time))
    full_audio_data_df = pd.concat([audio_features_df, audio_labels_df], ignore_index=True, axis=1)
    return full_audio_data_df


def plot_mfccs(full_fname):
    import matplotlib.pyplot as plt
    import librosa.display

    fig, ax = plt.subplots()
    audio_bin, sample_rate = librosa.load(full_fname, duration=2.5, sr=22050 * 2, offset=1.1)
    sample_rate = np.array(sample_rate)
    mfccs = librosa.feature.mfcc(y=audio_bin, sr=sample_rate, n_mfcc=128, dct_type=2)
    # Ploting a single strip of the Me
    mfccs3 = librosa.power_to_db(mfccs)
    img = librosa.display.specshow(mfccs3, x_axis='time', ax=ax, cmap=plt.get_cmap('inferno'))
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')
    plt.show()


def plot_mel_frequencies(full_fname):
    import matplotlib.pyplot as plt
    import librosa.display
    fig, ax = plt.subplots()
    audio_bin, sample_rate = librosa.load(full_fname, duration=3, sr=22050 * 2, offset=0.5)
    sample_rate = np.array(sample_rate)
    n_fft = 2048
    mel = librosa.feature.melspectrogram(y=audio_bin, sr=sample_rate, n_fft=n_fft, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)

    hop_length = 512
    img = librosa.display.specshow(mel, x_axis='time', ax=ax, hop_length=hop_length, cmap=plt.get_cmap('inferno'))
    fig.colorbar(img, ax=ax)
    ax.set(title='Mel')
    plt.show()


def extract_2d_mel_features(audio_files):
    from pandas import Series

    audio_features_df = pd.DataFrame()
    audio_labels_df = pd.DataFrame()
    start_time = time.time()

    for index, full_fname in tqdm(enumerate(audio_files)):
        file_name = Path(full_fname).name
        # if file_name[6:-16] != '01' and file_name[6:-16] != '08' \
        #         and file_name[:2] != 'su' and file_name[:1] != 'n' and file_name[:1] != 'd' and file_name[6:-16] != '07':

        emo_label = parse_fname_to_only_emo_label(file_name)
        gen_label = parse_fname_to_gender_label(file_name)

        if emo_label is None or emo_label == '':
            continue

        if len(file_name) > 13:
            audio_bin, sample_rate = librosa.load(full_fname, duration=3, sr=22050 * 2, offset=1.1)
        else:
            audio_bin, sample_rate = librosa.load(full_fname, duration=3, sr=22050 * 2, offset=0.1)

        sample_rate = np.array(sample_rate)
        mfccs = librosa.feature.mfcc(y=audio_bin, sr=sample_rate, n_mfcc=64, dct_type=1)
        #mfccs_mean = np.mean(mfccs, axis=0)

        mfccs = librosa.power_to_db(mfccs)
        mfccs = mfccs.flatten()
        audio_labels_df = audio_labels_df.append(Series([emo_label, gen_label]), ignore_index=True)
        audio_features_df = audio_features_df.append(Series(mfccs), ignore_index=True)

    print("Time to extract Mel features:  %s seconds." % (time.time() - start_time))
    full_audio_data_df = pd.concat([audio_features_df, audio_labels_df], ignore_index=True, axis=1)
    return full_audio_data_df


def parse_fname_to_emo_label(file_name):

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
    elif file_name[:1] == 'n':
        label = 'neutral'
    elif file_name[:2] == 'sa':
        label = 'male_sad'
    elif file_name[:2] == 'su':
        label = 'male_surprised'

    return label


def parse_fname_to_only_emo_label(file_name):

    label = None
    # RAVDESS file names
    if file_name[6:-16] == '01' and int(file_name[18:-4]) % 2 == 1:
        label = 'neutral'
    elif file_name[6:-16] == '01' and int(file_name[18:-4]) % 2 == 0:
        label = 'neutral'
    # elif file_name[6:-16] == '02' and int(file_name[18:-4]) % 2 == 0:
    #     label = 'calm'
    # elif file_name[6:-16] == '02' and int(file_name[18:-4]) % 2 == 1:
    #     label = 'calm'
    elif file_name[6:-16] == '03' and int(file_name[18:-4]) % 2 == 0:
        label = 'happy'
    elif file_name[6:-16] == '03' and int(file_name[18:-4]) % 2 == 1:
        label = 'happy'
    elif file_name[6:-16] == '04' and int(file_name[18:-4]) % 2 == 0:
        label = 'sad'
    elif file_name[6:-16] == '04' and int(file_name[18:-4]) % 2 == 1:
        label = 'sad'
    elif file_name[6:-16] == '05' and int(file_name[18:-4]) % 2 == 0:
        label = 'angry'
    elif file_name[6:-16] == '05' and int(file_name[18:-4]) % 2 == 1:
        label = 'angry'
    elif file_name[6:-16] == '06' and int(file_name[18:-4]) % 2 == 0:
        label = 'fearful'
    elif file_name[6:-16] == '06' and int(file_name[18:-4]) % 2 == 1:
        label = 'fearful'
    elif file_name[6:-16] == '07' and int(file_name[18:-4]) % 2 == 0:
        label = 'disgust'
    elif file_name[6:-16] == '07' and int(file_name[18:-4]) % 2 == 1:
        label = 'disgust'
    elif file_name[6:-16] == '08' and int(file_name[18:-4]) % 2 == 0:
        label = 'surprised'
    elif file_name[6:-16] == '08' and int(file_name[18:-4]) % 2 == 1:
        label = 'surprised'

    #EMOVO file names
    elif file_name[:3] == 'dis':
        label = 'disgust'
    elif file_name[:3] == 'gio':
        label = 'happy'
    elif file_name[:3] == 'neu':
        label = 'neutral'
    elif file_name[:3] == 'pau':
        label = 'fearful'
    elif file_name[:3] == 'rab':
        label = 'angry'
    elif file_name[:3] == 'sor':
        label = 'surprised'
    elif file_name[:3] == 'tri':
        label = 'sad'

    #SAVEE file names
    elif file_name[:1] == 'a':
        label = 'angry'
    elif file_name[:1] == 'f':
        label = 'fearful'
    elif file_name[:1] == 'h':
        label = 'happy'
    elif file_name[:1] == 'n':
        label = 'neutral'
    elif file_name[:1] == 'd':
        label = 'disgust'
    elif file_name[:2] == 'sa':
        label = 'sad'
    elif file_name[:2] == 'su':
        label = 'surprised'

    #EMODB
    elif file_name[5] == 'W':
        label = 'angry'
    # Unique to emodb.
    # elif file_name[5] == 'L':
    #     label = 'boredom'
    elif file_name[5] == 'E':
        label = 'disgust'
    elif file_name[5] == 'A':
        label = 'fearful'
    elif file_name[5] == 'F':
        label = 'happy'
    elif file_name[5] == 'T':
        label = 'sad'
    elif file_name[5] == 'N':
        label = 'neutral'

    return label


def parse_fname_to_gender_label(file_name):

    f_name_len = len(file_name)
    label = None
    # Emodb parse
    if f_name_len == 11:
        if int(file_name[:2]) == 3:
            label = 'male'
        elif int(file_name[:2]) == 8:
            label = 'female'
        elif int(file_name[:2]) == 9:
            label = 'female'
        elif int(file_name[:2]) == 10:
            label = 'male'
        elif int(file_name[:2]) == 11:
            label = 'male'
        elif int(file_name[:2]) == 12:
            label = 'male'
        elif int(file_name[:2]) == 13:
            label = 'female'
        elif int(file_name[:2]) == 14:
            label = 'female'
        elif int(file_name[:2]) == 15:
            label = 'male'
        elif int(file_name[:2]) == 16:
            label = 'female'
    # Emovo parse
    elif f_name_len == 13:
        if file_name[4] == 'm':
            label = 'male'
        else:
            label = 'female'

    # Savee all male
    elif f_name_len <= 8:
        label = 'male'

    # Ravdess
    elif int(file_name[18:-4]) % 2 == 1:
        label = 'male'
    else:
        label = 'female'

    if label is None:
        raise Exception('No gender set')

    return label

