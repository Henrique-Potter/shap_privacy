import glob

import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import cv2
from pandas import Series

from tqdm import tqdm

def pre_process_data(audio_files_path, get_emotion_label=True, augment_data=False):

    audio_files = glob.glob("{}/**/*.wav".format(audio_files_path), recursive=True)

    lst = []
    for full_fname in tqdm(audio_files):
        lst.append(full_fname)

    lst_np = np.array(lst)
    uniques = np.unique(lst_np)

    audio_raw_df_path = 'data/audio_data_raw_df.pkl'

    if not Path(audio_raw_df_path).exists():
        print("Loading files and raw audio data.")
        audio_raw_df = extract_raw_audio(audio_files)
        print("Loading files and raw audio data successful.")
        audio_raw_df.to_pickle(audio_raw_df_path)

    else:
        print("Loading pre extracted raw audio from pkl file.")
        audio_raw_df = pd.read_pickle(audio_raw_df_path)
        print("Loading pre extracted raw audio from pkl file successful.")

    # # # -1 and 1 scaling
    # for column in tqdm(audiol_features_df.iloc[:, :-2].columns):
    #     audiol_features_df[column] = audiol_features_df[column] / audiol_features_df[column].abs().max()

    if get_emotion_label:
        X_train, X_test, y_train, y_test = train_test_split(audio_raw_df.iloc[:, :-2],
                                                            audio_raw_df.iloc[:, -2:-1],
                                                            test_size=0.2, random_state=6)
    else:
        X_train, X_test, y_train, y_test = train_test_split(audio_raw_df.iloc[:, :-2],
                                                            audio_raw_df.iloc[:, -1:],
                                                            test_size=0.2, random_state=6)

    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder

    X_train_np = np.array(X_train)
    y_train_np = np.ravel(np.array(y_train))
    X_test_np = np.array(X_test)
    y_test_np = np.ravel(np.array(y_test))

    lb = LabelEncoder()
    y_train_encoded = np_utils.to_categorical(lb.fit_transform(y_train_np))
    y_test_encoded = np_utils.to_categorical(lb.fit_transform(y_test_np))

    train_data_mfcc_aug_path = 'data/audio_train_data_mfcc_aug_np.npy'
    train_data_mfcc_path = 'data/audio_train_data_mfcc_np.npy'

    n_mfcc = 40
    print("Extracting mfccs from raw nd train audio data split.")
    if augment_data:
        if not Path(train_data_mfcc_aug_path).exists():
            print("Augmenting data!")
            X_train1, X_train2, X_train3 = extract_mfcc_from_raw_ndarray_aug_shift(X_train_np, n_mfcc, 5000)
            X_train_mfcc = np.concatenate((X_train1, X_train2, X_train3), axis=0)

            np.save(train_data_mfcc_aug_path, X_train_mfcc)
            y_train_encoded = np.concatenate((y_train_encoded, y_train_encoded, y_train_encoded), axis=0)
        else:
            print("Augmented train data found. Loading from npy file.")
            X_train_mfcc = np.load(train_data_mfcc_aug_path)

        print("Loading augmented train data successful.")
    else:
        if not Path(train_data_mfcc_path).exists():
            X_train_mfcc = extract_mfcc_from_raw_ndarray(X_train_np, n_mfcc)
            np.save(train_data_mfcc_path, X_train_mfcc)
        else:
            print("Train data found. Loading from npy file.")
            X_train_mfcc = np.load(train_data_mfcc_path)
        print("Loading train successful.")

    test_data_mfcc_path = 'data/audio_test_data_mfcc_np.npy'
    print("Extracting mfccs from raw nd test audio data split.")
    if not Path(test_data_mfcc_path).exists():
        X_test_mfcc = extract_mfcc_from_raw_ndarray(X_test_np, n_mfcc)
        np.save(test_data_mfcc_path, X_test_mfcc)
    else:
        print("Test data found. Loading from npy file.")
        X_test_mfcc = np.load(test_data_mfcc_path)
    print("Loading test data successful.")

    x_traincnn = np.expand_dims(X_train_mfcc, axis=2)
    x_testcnn = np.expand_dims(X_test_mfcc, axis=2)

    return x_traincnn, y_train_encoded, x_testcnn, y_test_encoded


def extract_mfcc_from_raw_ndarray(X_train, n_mfcc):
    x_index = 0
    x_mfcc_train = np.ndarray((X_train.shape[0], n_mfcc))
    for x_row in tqdm(X_train):
        mfccs = librosa.feature.mfcc(y=x_row, sr=22050 * 2, n_mfcc=n_mfcc).T
        mfccs_mean = np.mean(mfccs, axis=0)

        x_mfcc_train[x_index, :] = mfccs_mean
        x_index += 1

    return x_mfcc_train


def extract_mfcc_from_raw_ndarray_aug_shift(X_train, n_mfcc, shift_array):
    from scipy.ndimage.interpolation import shift
    x_index = 0
    x_mfcc_train = np.ndarray((X_train.shape[0], n_mfcc))
    x_mfcc_train_pos_shift = np.ndarray((X_train.shape[0], n_mfcc))
    x_mfcc_train_neg_shift = np.ndarray((X_train.shape[0], n_mfcc))
    for x_row in tqdm(X_train):

        mfccs1 = librosa.feature.mfcc(y=x_row, sr=22050 * 2, n_mfcc=n_mfcc).T
        mfccs_mean1 = np.mean(mfccs1, axis=0)
        x_mfcc_train[x_index, :] = mfccs_mean1

        x_row_pos = shift(x_row, shift_array, cval=0)
        mfccs2 = librosa.feature.mfcc(y=x_row_pos, sr=22050 * 2, n_mfcc=n_mfcc).T
        mfccs_mean2 = np.mean(mfccs2, axis=0)
        x_mfcc_train_pos_shift[x_index, :] = mfccs_mean2

        x_row_neg = shift(x_row, shift_array * -1, cval=0)
        mfccs3 = librosa.feature.mfcc(y=x_row_neg, sr=22050 * 2, n_mfcc=n_mfcc).T
        mfccs_mean3 = np.mean(mfccs3, axis=0)
        x_mfcc_train_neg_shift[x_index, :] = mfccs_mean3

        x_index += 1

    return x_mfcc_train, x_mfcc_train_pos_shift, x_mfcc_train_neg_shift


def pre_process_data3(audio_files_path, get_emotion_label=True):

    audio_files = glob.glob("{}/**/*.wav".format(audio_files_path), recursive=True)

    lst = []
    for full_fname in tqdm(audio_files):
        lst.append(full_fname)

    lst_np = np.array(lst)
    uniques = np.unique(lst_np)

    audio_mfcc_x = './data/audio_data_x.npy'
    audio_mfcc_y = './data/audio_data_y.npy'

    if not Path(audio_mfcc_x).exists():
        print("Loading files and extracting features.")
        audiol_features_df = extract_mel_features2(audio_files)

        X, y = zip(*audiol_features_df)

        X = np.asarray(X)
        Y = np.asarray(y)

        np.save(audio_mfcc_x, X)
        np.save(audio_mfcc_y, Y)

    else:
        print("Loading pre_extracted features from file.")
        # audiol_features_df = pd.read_pickle(audio_mel_df_path)
        X = np.load(audio_mfcc_x)
        Y = np.load(audio_mfcc_y)

    # # # -1 and 1 scaling
    # for column in tqdm(audiol_features_df.iloc[:, :-2].columns):
    #     audiol_features_df[column] = audiol_features_df[column] / audiol_features_df[column].abs().max()

    if get_emotion_label:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(audiol_features_df.iloc[:, :-2], audiol_features_df.iloc[:, -1:],
                                                            test_size=0.2, random_state=5)

    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder

    pX = np.sum(np.concatenate((X_train, X_test), axis=0) != X)
    pY = np.sum(np.concatenate((y_train, y_test), axis=0) != Y)

    lb = LabelEncoder()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)

    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    return x_traincnn, y_train, x_testcnn, y_test


def pre_process_fseer_data(audio_files_path, get_emotion_label):

    audio_files = glob.glob("{}/**/*.wav".format(audio_files_path), recursive=True)

    audio_mel_df_path = 'data/fser_audio_data_df.pkl'

    if not Path(audio_mel_df_path).exists():
        print("Loading files and extracting features.")

        audiol_features_df = extract_2d_mel_features(audio_files)
        audiol_features_df.fillna(0, inplace=True)

        audiol_features_df.to_pickle(audio_mel_df_path)
    else:
        print("Loading pre_extracted features from file.")
        audiol_features_df = pd.read_pickle(audio_mel_df_path)

    print(audiol_features_df)

    # # # -1 and 1 scaling
    # for column in tqdm(audiol_features_df.iloc[:, :-2].columns):
    #     audiol_features_df[column] = audiol_features_df[column] / audiol_features_df[column].abs().max()

    if get_emotion_label:
        X_train, X_test, y_train, y_test = train_test_split(audiol_features_df.iloc[:, :-2],
                                                            audiol_features_df.iloc[:, -2:-1],
                                                            test_size=0.2, random_state=6)
    else:
        X_train, X_test, y_train, y_test = train_test_split(audiol_features_df.iloc[:, :-2],
                                                            audiol_features_df.iloc[:, -1:],
                                                            test_size=0.33, random_state=6)

    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder

    X_train = np.array(X_train)
    y_train = np.ravel(np.array(y_train))
    X_test = np.array(X_test)
    y_test = np.ravel(np.array(y_test))

    lb = LabelEncoder()
    y_train_encoded = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test_encoded = np_utils.to_categorical(lb.fit_transform(y_test))
    # y_train_encoded = lb.fit_transform(y_train)
    # y_test_encoded = lb.fit_transform(y_test)
    print(y_train)
    print(y_train_encoded)
    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    return x_traincnn, y_train_encoded, x_testcnn, y_test_encoded


def extract_raw_audio(audio_files):

    audio_raw_df = pd.DataFrame()
    audio_labels_df = pd.DataFrame()
    start_time = time.time()

    lst = []

    for full_fname in tqdm(audio_files):
        file_name = Path(full_fname).name

        emo_label = parse_fname_to_only_emo_label(file_name)
        gen_label = parse_fname_to_gender_label(file_name)

        if emo_label is None or emo_label == '':
            continue

        audio_bin, sample_rate = librosa.load(full_fname, res_type='kaiser_fast')

        audio_labels_df = audio_labels_df.append(Series([emo_label, gen_label]), ignore_index=True)
        audio_raw_df = audio_raw_df.append(Series(audio_bin), ignore_index=True)

    print("Time to extract Mel features:  %s seconds." % (time.time() - start_time))
    full_raw_audio_data_df = pd.concat([audio_raw_df, audio_labels_df], ignore_index=True, axis=1)

    # Setting NAs to 0 since audio waves have different sizes
    full_raw_audio_data_df.fillna(0, inplace=True)
    return full_raw_audio_data_df


def extract_mfcc_feature(audio_bin, audio_features_df, audio_labels_df, emo_label, gen_label, sample_rate):
    sample_rate = np.array(sample_rate)
    mfccs = librosa.feature.mfcc(y=audio_bin, sr=sample_rate, n_mfcc=40).T
    mfccs_mean = np.mean(mfccs, axis=0)
    # plot_mfccs(full_fname)
    audio_labels_df = audio_labels_df.append(Series([emo_label, gen_label]), ignore_index=True)
    audio_features_df = audio_features_df.append(Series(mfccs_mean), ignore_index=True)
    return audio_features_df, audio_labels_df


def extract_mel_features2(audio_files):
    start_time = time.time()

    all_data = []

    for full_fname in tqdm(audio_files):
        file_name = Path(full_fname).name

        emo_label = parse_fname_to_only_emo_label(file_name)
        gen_label = parse_fname_to_gender_label(file_name)

        if emo_label is None or emo_label == '':
            continue

        audio_bin, sample_rate = librosa.load(full_fname, res_type='kaiser_fast')

        sample_rate = np.array(sample_rate)
        mfccs = librosa.feature.mfcc(y=audio_bin, sr=sample_rate, n_mfcc=40).T
        mfccs_mean = np.mean(mfccs, axis=0)

        # file_name = file_name[7:8]
        # arr = mfccs_mean, file_name

        # emo_label = int(file_name[7:8])-1
        arr = mfccs_mean, emo_label

        all_data.append(arr)

    print("Time to extract Mel features:  %s seconds." % (time.time() - start_time))
    return all_data


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
    mfcc_size = 0
    for full_fname in tqdm(audio_files):
        file_name = Path(full_fname).name
        # if file_name[6:-16] != '01' and file_name[6:-16] != '08' \
        #         and file_name[:2] != 'su' and file_name[:1] != 'n' and file_name[:1] != 'd' and file_name[6:-16] != '07':

        emo_label = parse_fname_to_only_emo_label(file_name)
        gen_label = parse_fname_to_gender_label(file_name)

        if emo_label is None or emo_label == '':
            continue

        if len(file_name) >= 13:
            audio_bin, sample_rate = librosa.load(full_fname, duration=3, sr=22050 * 2, res_type='kaiser_fast', offset=1.1)
        else:
            audio_bin, sample_rate = librosa.load(full_fname, duration=3, sr=22050 * 2, res_type='kaiser_fast')

        sample_rate = np.array(sample_rate)

        mfccs = librosa.feature.melspectrogram(y=audio_bin, sr=sample_rate, n_mels=128, n_fft=512, hop_length=512,
                                               win_length=512)


        mfccs = cv2.resize(mfccs, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

        #mfccs = librosa.feature.mfcc(y=audio_bin, sr=sample_rate, n_mfcc=64, dct_type=1)
        #mfccs_mean = np.mean(mfccs, axis=0)

        if mfccs.shape[1] > mfcc_size:
            mfcc_size = mfccs.shape[1]
            print(mfcc_size)

        #mfccs = librosa.power_to_db(mfccs, ref=100)
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
    if len(file_name) >= 20:
        # RAVDESS file names
        if file_name[7:8] == '1':
            label = 'neutral'
        # elif file_name[7:8] == '2':
        #     label = 'calm'
        elif file_name[7:8] == '3':
            label = 'happy'
        elif file_name[7:8] == '4':
            label = 'sad'
        elif file_name[7:8] == '5':
            label = 'angry'
        elif file_name[7:8] == '6':
            label = 'fearful'
        elif file_name[7:8] == '7':
            label = 'disgust'
        elif file_name[7:8] == '8':
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

