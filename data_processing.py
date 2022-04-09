import glob
import time
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def pre_process_data(audio_files_path, db_name, n_mfcc=40):

    train_x_mfcc_path = 'data/audio_train_data_mfcc{}_{}_np.npy'.format(n_mfcc, db_name)
    test_x_mfcc_path = 'data/audio_test_data_mfcc{}_{}_np.npy'.format(n_mfcc, db_name)

    train_y_emo_mfcc_path = 'data/audio_emo_train_y_data_{}_{}_np.npy'.format(n_mfcc, db_name)
    test_y_emo_mfcc_path = 'data/audio_emo_test_y_data_{}_{}_np.npy'.format(n_mfcc, db_name)

    train_y_gen_mfcc_path = 'data/audio_gen_train_y_data_{}_{}_np.npy'.format(n_mfcc, db_name)
    test_y_gen_mfcc_path = 'data/audio_gen_test_y_data_{}_{}_np.npy'.format(n_mfcc, db_name)

    train_y_sv_mfcc_path = 'data/audio_sv_train_y_data_{}_{}_np.npy'.format(n_mfcc, db_name)
    test_y_sv_mfcc_path = 'data/audio_sv_test_y_data_{}_{}_np.npy'.format(n_mfcc, db_name)

    # Assuming that y has to exist if x exists
    if not Path(train_x_mfcc_path).exists() and not Path(test_x_mfcc_path).exists():

        audio_data_full_path = Path(audio_files_path).joinpath(db_name)
        audio_files = glob.glob("{}/**/*.wav".format(audio_data_full_path), recursive=True)

        lst = []
        for full_fname in tqdm(audio_files):
            lst.append(full_fname)

        audio_raw_df_path = 'data/audio_preprocess/audio_{}_raw_df.pkl'.format(db_name)

        if not Path(audio_raw_df_path).exists():

            print("Loading files and raw audio data.")

            audio_raw_df = extract_raw_audio(audio_files)
            print("Loading files and raw audio data successful.")
            audio_raw_df.to_pickle(audio_raw_df_path)

        else:
            print("Loading pre extracted raw audio from pkl file.")
            audio_raw_df = pd.read_pickle(audio_raw_df_path)
            print("Loading pre extracted raw audio from pkl file successful.")

        X_train, X_test, y_labels_train, y_labels_test = train_test_split(audio_raw_df.iloc[:, :-3],
                                                                          audio_raw_df.iloc[:, -3:], test_size=0.2,
                                                                          random_state=6)

        y_emo_train = y_labels_train.iloc[:, :1]
        y_gen_train = y_labels_train.iloc[:, 1:2]
        y_id_train = y_labels_train.iloc[:, 2:3]

        y_emo_test = y_labels_test.iloc[:, :1]
        y_gen_test = y_labels_test.iloc[:, 1:2]
        y_id_test = y_labels_test.iloc[:, 2:3]

        from keras.utils import np_utils
        from sklearn.preprocessing import LabelEncoder

        X_train_np = np.array(X_train)
        y_emo_train_np = np.ravel(np.array(y_emo_train))
        y_gen_train_np = np.ravel(np.array(y_gen_train))
        y_id_train_np = np.ravel(np.array(y_id_train))

        X_test_np = np.array(X_test)
        y_emo_test_np = np.ravel(np.array(y_emo_test))
        y_gen_test_np = np.ravel(np.array(y_gen_test))
        y_id_test_np = np.ravel(np.array(y_id_test))

        lb = LabelEncoder()
        y_emo_tr = np_utils.to_categorical(lb.fit_transform(y_emo_train_np))
        y_gen_tr = np_utils.to_categorical(lb.fit_transform(y_gen_train_np))
        y_id_tr = np_utils.to_categorical(lb.fit_transform(y_id_train_np))

        y_emo_te = np_utils.to_categorical(lb.fit_transform(y_emo_test_np))
        y_gen_te = np_utils.to_categorical(lb.fit_transform(y_gen_test_np))
        y_id_te = np_utils.to_categorical(lb.fit_transform(y_id_test_np))

        print("Augmenting data and extracting train MFCCs!")
        all_aug_data = extract_mfcc_from_raw_ndarray_aug_shift(X_train_np, n_mfcc, 5000)
        x_tr_mfcc = np.concatenate(all_aug_data, axis=0)

        y_multi = x_tr_mfcc.shape[0] / y_emo_tr.shape[0]
        y_emo_tr = np.concatenate((y_emo_tr,) * int(y_multi), axis=0)
        y_gen_tr = np.concatenate((y_gen_tr,) * int(y_multi), axis=0)
        y_id_tr = np.concatenate((y_id_tr,) * int(y_multi), axis=0)

        np.save(train_x_mfcc_path.format(n_mfcc), x_tr_mfcc)
        np.save(train_y_emo_mfcc_path.format(n_mfcc), y_emo_tr)
        np.save(train_y_gen_mfcc_path.format(n_mfcc), y_gen_tr)
        np.save(train_y_sv_mfcc_path.format(n_mfcc), y_id_tr)

        print("Extracting test MFCCs from raw audio.")
        x_te_mfcc = extract_mean_mfcc_from_raw_ndarray(X_test_np, n_mfcc)
        np.save(test_x_mfcc_path, x_te_mfcc)
        np.save(test_y_emo_mfcc_path, y_emo_te)
        np.save(test_y_gen_mfcc_path, y_gen_te)
        np.save(test_y_sv_mfcc_path, y_id_te)

        print("Loading data from raw audio successful.")

    else:

        x_tr_mfcc = np.load(train_x_mfcc_path)
        x_te_mfcc = np.load(test_x_mfcc_path)

        y_emo_tr = np.load(train_y_emo_mfcc_path)
        y_emo_te = np.load(test_y_emo_mfcc_path)
        y_gen_tr = np.load(train_y_gen_mfcc_path)
        y_gen_te = np.load(test_y_gen_mfcc_path)
        y_id_tr = np.load(train_y_sv_mfcc_path)
        y_id_te = np.load(test_y_sv_mfcc_path)

        print("Loading augmented data successful.")

    return x_tr_mfcc, x_te_mfcc, y_emo_tr, y_emo_te, y_gen_tr, y_gen_te, y_id_tr, y_id_te


def extract_mean_mfcc_from_raw_ndarray(X_train, n_mfcc):
    x_index = 0

    # feature_sz = n_mfcc * 3 + 1
    feature_sz = n_mfcc

    x_mfcc_train = np.ndarray((X_train.shape[0], feature_sz))
    for x_row in tqdm(X_train):
        extract_mfcc(x_mfcc_train, n_mfcc, x_index, x_row)

        x_index += 1

    return x_mfcc_train


def extract_mfcc_matrix_from_raw_ndarray(X_train, n_mfcc, matrix_size):
    x_mfcc_train = pd.DataFrame()
    for x_row in tqdm(X_train):
        mfccs1 = librosa.feature.mfcc(y=x_row, sr=22050 * 2, n_mfcc=n_mfcc)
        mfccs1 = cv2.resize(mfccs1, dsize=(matrix_size, matrix_size), interpolation=cv2.INTER_CUBIC)
        mfccs1 = mfccs1.flatten()
        x_mfcc_train = x_mfcc_train.append(Series(mfccs1), ignore_index=True)

    return x_mfcc_train.values


def extract_mfcc_from_raw_ndarray_aug_shift(x_data, n_mfcc, shift_array):
    from scipy.ndimage.interpolation import shift
    x_index = 0

    # feature_sz = n_mfcc * 3 + 1
    feature_sz = n_mfcc

    x_mfcc_train = np.ndarray((x_data.shape[0], feature_sz))
    x_mfcc_train_pos_shift = np.ndarray((x_data.shape[0], feature_sz))
    x_mfcc_train_neg_shift = np.ndarray((x_data.shape[0], feature_sz))

    # audio_features_np_rnd_1 = np.zeros((x_data.shape[0], feature_sz))
    # audio_features_np_rnd_2 = np.zeros((x_data.shape[0], feature_sz))
    # audio_features_np_rnd_3 = np.zeros((x_data.shape[0], feature_sz))

    for x_row in tqdm(x_data):
        extract_mfcc(x_mfcc_train, n_mfcc, x_index, x_row)

        x_row_pos = shift(x_row, shift_array, cval=0)
        extract_mfcc(x_mfcc_train_pos_shift, n_mfcc, x_index, x_row_pos)

        x_row_neg = shift(x_row, shift_array * -1, cval=0)
        extract_mfcc(x_mfcc_train_neg_shift, n_mfcc, x_index, x_row_neg)

        # increment_percent = 0.03
        # x_row_rnd = add_random_noise(increment_percent, x_row)
        # extract_mfcc(audio_features_np_rnd_1, n_mfcc, x_index, x_row_rnd)
        #
        # increment_percent = 0.03
        # x_row_rnd2 = add_random_noise(increment_percent, x_row)
        # extract_mfcc(audio_features_np_rnd_2, n_mfcc, x_index, x_row_rnd2)

        x_index += 1

    return x_mfcc_train, x_mfcc_train_pos_shift, x_mfcc_train_neg_shift,


def extract_mfcc(x_mfcc_train, n_mfcc, x_index, x_row):
    non_zero_idxs = np.argwhere(x_row != 0)[:, 0]
    x_row = x_row[non_zero_idxs[0]:non_zero_idxs[-1]]

    mfccs1 = librosa.feature.mfcc(y=x_row, sr=22050 * 2, n_mfcc=n_mfcc).T
    mfccs_mean1 = np.mean(mfccs1, axis=0)

    # energy = np.sum(np.power(mfccs_mean1, 2))
    # mfcc_delta1 = librosa.feature.delta(mfccs_mean1, width=3, axis=0, order=1)
    # mfcc_delta2 = librosa.feature.delta(mfccs_mean1, width=3, axis=0, order=2)

    # features = np.append(np.concatenate((mfccs_mean1, mfcc_delta1, mfcc_delta2)), energy)

    x_mfcc_train[x_index, :] = mfccs_mean1


def extract_mfcc_matrix_from_raw_ndarray_aug_shift(X_train, n_mfcc, shift_array, matrix_size):
    from scipy.ndimage.interpolation import shift
    x_index = 0

    audio_features_df = pd.DataFrame()
    audio_features_df_pos_shift = pd.DataFrame()
    audio_features_df_neg_shift = pd.DataFrame()

    for x_row in tqdm(X_train):
        mfccs1 = librosa.feature.mfcc(y=x_row, sr=22050 * 2, n_mfcc=n_mfcc)
        mfccs1 = cv2.resize(mfccs1, dsize=(matrix_size, matrix_size), interpolation=cv2.INTER_CUBIC)
        mfccs1 = mfccs1.flatten()
        audio_features_df = audio_features_df.append(Series([mfccs1]), ignore_index=True)

        x_row_pos = shift(x_row, shift_array, cval=0)
        mfccs2 = librosa.feature.mfcc(y=x_row_pos, sr=22050 * 2, n_mfcc=n_mfcc)
        mfccs2 = cv2.resize(mfccs2, dsize=(matrix_size, matrix_size), interpolation=cv2.INTER_CUBIC)
        mfccs2 = mfccs2.flatten()
        audio_features_df_pos_shift = audio_features_df.append(Series([mfccs2]), ignore_index=True)

        x_row_neg = shift(x_row, shift_array * -1, cval=0)
        mfccs3 = librosa.feature.mfcc(y=x_row_neg, sr=22050 * 2, n_mfcc=n_mfcc)
        mfccs3 = cv2.resize(mfccs3, dsize=(matrix_size, matrix_size), interpolation=cv2.INTER_CUBIC)
        mfccs3 = mfccs3.flatten()
        audio_features_df_neg_shift = audio_features_df.append(Series([mfccs3]), ignore_index=True)

        x_index += 1

    return audio_features_df.values, audio_features_df_pos_shift.values, audio_features_df_neg_shift.values


def extract_mel_matrix_from_raw_ndarray_aug_shift(x_data, shift_positions, n_mels, n_fft=2048):
    from scipy.ndimage.interpolation import shift

    x_index = 0
    sample_rate = 22050 * 2

    audio_features_np = np.zeros((x_data.shape[0], n_mels * n_mels))
    audio_features_np_pos_shift = np.zeros((x_data.shape[0], n_mels * n_mels))
    audio_features_np_neg_shift = np.zeros((x_data.shape[0], n_mels * n_mels))
    audio_features_np_rnd_1 = np.zeros((x_data.shape[0], n_mels * n_mels))
    audio_features_np_rnd_2 = np.zeros((x_data.shape[0], n_mels * n_mels))
    audio_features_np_rnd_3 = np.zeros((x_data.shape[0], n_mels * n_mels))

    output_list = []
    output_list.append(audio_features_np)
    output_list.append(audio_features_np_pos_shift)
    output_list.append(audio_features_np_neg_shift)
    output_list.append(audio_features_np_rnd_1)
    output_list.append(audio_features_np_rnd_2)
    output_list.append(audio_features_np_rnd_3)

    for x_row in tqdm(x_data):
        extract_mel_to_nd_array(audio_features_np, n_fft, n_mels, sample_rate, x_index, x_row)

        x_row_pos = shift(x_row, shift_positions, cval=0)
        extract_mel_to_nd_array(audio_features_np_pos_shift, n_fft, n_mels, sample_rate, x_index, x_row_pos)

        x_row_neg = shift(x_row, shift_positions * -1, cval=0)
        extract_mel_to_nd_array(audio_features_np_neg_shift, n_fft, n_mels, sample_rate, x_index, x_row_neg)

        increment_percent = 1.01
        x_row_rnd = add_random_noise(increment_percent, x_row)
        extract_mel_to_nd_array(audio_features_np_rnd_1, n_fft, n_mels, sample_rate, x_index, x_row_rnd)

        increment_percent = 1.01
        x_row_rnd2 = add_random_noise(increment_percent, x_row)
        extract_mel_to_nd_array(audio_features_np_rnd_2, n_fft, n_mels, sample_rate, x_index, x_row_rnd2)

        increment_percent = 1.02
        x_row_rnd3 = add_random_noise(increment_percent, x_row)
        extract_mel_to_nd_array(audio_features_np_rnd_3, n_fft, n_mels, sample_rate, x_index, x_row_rnd3)

        x_index += 1

    return output_list


def add_random_noise(increment_percent, x_row):
    rnd_direction = np.random.randint(0, 2, size=x_row.shape[0])
    rnd_direction[rnd_direction == 0] = -1
    x_row_rnd_noise = x_row * increment_percent * rnd_direction
    x_row_rnd = x_row + x_row_rnd_noise
    return x_row_rnd


def extract_mel_to_nd_array(audio_features_np, n_fft, n_mels, sample_rate, x_index, x_row):
    mel1 = librosa.feature.melspectrogram(y=x_row, sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    non_zero_idxs = np.argwhere(np.mean(mel1, axis=0) != 0)[:, 0]
    mel1 = mel1[:, non_zero_idxs[0]:non_zero_idxs[-1]]
    mel1 = librosa.power_to_db(mel1, ref=np.max)
    mel1 = cv2.resize(mel1, dsize=(n_mels, n_mels), interpolation=cv2.INTER_CUBIC)
    mel1 = mel1.flatten()
    audio_features_np[x_index, :] = mel1


def pre_process_fseer_data(audio_files_path, n_mfcc=40, get_emotion_label=True, augment_data=False):
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

    train_data_mfcc_aug_path = 'data/audio_train_data_mfcc_matrix{}_aug_np.npy'.format(n_mfcc)
    train_data_mfcc_path = 'data/audio_train_data_mfcc_matrix{}_np.npy'.format(n_mfcc)

    print("Extracting mfccs from raw nd train audio data split.")
    if augment_data:
        if not Path(train_data_mfcc_aug_path).exists():
            print("Augmenting data!")
            X_train1, X_train2, X_train3 = extract_mfcc_matrix_from_raw_ndarray_aug_shift(X_train_np, n_mfcc, 5000, 64)
            X_train_mfcc = np.concatenate((X_train1, X_train2, X_train3), axis=0)

            np.save(train_data_mfcc_aug_path.format(n_mfcc), X_train_mfcc)
            y_train_encoded = np.concatenate((y_train_encoded, y_train_encoded, y_train_encoded), axis=0)
        else:
            print("Augmented train data found. Loading from npy file.")
            X_train_mfcc = np.load(train_data_mfcc_aug_path)

        print("Loading augmented train data successful.")
    else:
        if not Path(train_data_mfcc_path).exists():
            X_train_mfcc = extract_mfcc_matrix_from_raw_ndarray(X_train_np, n_mfcc, 64)
            np.save(train_data_mfcc_path, X_train_mfcc)
        else:
            print("Train data found. Loading from npy file.")
            X_train_mfcc = np.load(train_data_mfcc_path)
        print("Loading train successful.")

    test_data_mfcc_path = 'data/audio_test_data_mfcc_matrix{}_np.npy'.format(n_mfcc)
    print("Extracting mfccs from raw nd test audio data split.")
    if not Path(test_data_mfcc_path).exists():
        X_test_mfcc = extract_mfcc_matrix_from_raw_ndarray(X_test_np, n_mfcc, 64)
        np.save(test_data_mfcc_path, X_test_mfcc)
    else:
        print("Test data found. Loading from npy file.")
        X_test_mfcc = np.load(test_data_mfcc_path)
    print("Loading test data successful.")

    # x_traincnn = np.expand_dims(X_train_mfcc, axis=2)
    # x_testcnn = np.expand_dims(X_test_mfcc, axis=2)

    return X_train_mfcc, y_train_encoded, X_test_mfcc, y_test_encoded


def to_batchdataset(x_train_input, x_test_input, batch_size):
    import tensorflow as tf

    mdl_train_dataset = tf.data.Dataset.from_tensor_slices(x_train_input)

    if x_test_input is not None:
        mdl_test_dataset = tf.data.Dataset.from_tensor_slices(x_test_input)
        te_batchdt = mdl_test_dataset.padded_batch(batch_size)
    else:
        te_batchdt = None

    tr_batchdt = mdl_train_dataset.padded_batch(batch_size, drop_remainder=True)

    return tr_batchdt, te_batchdt


def pre_process_audio_to_mel_data(audio_files_path, n_mels=128, get_emotion_label=True, augment_data=False):
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

    train_data_mel_aug_path = 'data/audio_train_data_mel_matrix{}_aug_np.npy'.format(n_mels)
    train_data_mel_path = 'data/audio_train_data_mel_matrix{}_np.npy'.format(n_mels)

    print("Extracting mels from raw nd train audio data split.")
    if augment_data:
        if not Path(train_data_mel_aug_path).exists():
            print("Augmenting data!")
            all_aug_data = extract_mel_matrix_from_raw_ndarray_aug_shift(X_train_np, 5000, n_mels)
            X_train_mel_matrix = np.concatenate(all_aug_data, axis=0)

            np.save(train_data_mel_aug_path.format(n_mels), X_train_mel_matrix)
            y_multi = X_train_mel_matrix.shape[0] / y_train_encoded.shape[0]
            y_train_encoded = np.concatenate((y_train_encoded,) * int(y_multi), axis=0)
        else:
            print("Augmented train data found. Loading from npy file.")
            X_train_mel_matrix = np.load(train_data_mel_aug_path)
            y_multi = X_train_mel_matrix.shape[0] / y_train_encoded.shape[0]
            y_train_encoded = np.concatenate((y_train_encoded,) * int(y_multi), axis=0)

        print("Loading augmented train data successful.")
    else:
        if not Path(train_data_mel_path).exists():
            # Ignoring the augmentation [0]
            X_train_mel_matrix = extract_mel_matrix_from_raw_ndarray_aug_shift(X_train_np, 5000, n_mels)[0]
            np.save(train_data_mel_path, X_train_mel_matrix)
        else:
            print("Train data found. Loading from npy file.")
            X_train_mel_matrix = np.load(train_data_mel_path, allow_pickle=True)
        print("Loading train successful.")

    test_data_mfcc_path = 'data/audio_test_data_mel_matrix{}_np.npy'.format(n_mels)
    print("Extracting mels from raw nd test audio data split.")
    if not Path(test_data_mfcc_path).exists():
        # Ignoring the augmentation [0]
        X_test_mel = extract_mel_matrix_from_raw_ndarray_aug_shift(X_test_np, 5000, n_mels)[0]
        np.save(test_data_mfcc_path, X_test_mel)
    else:
        print("Test data found. Loading from npy file.")
        X_test_mel = np.load(test_data_mfcc_path, allow_pickle=True)
    print("Loading test data successful.")

    # x_traincnn = np.expand_dims(X_train_mfcc, axis=2)
    # x_testcnn = np.expand_dims(X_test_mfcc, axis=2)

    return X_train_mel_matrix, y_train_encoded, X_test_mel, y_test_encoded


def extract_raw_audio(audio_files):
    audio_raw_df = pd.DataFrame()
    audio_labels_df = pd.DataFrame()
    start_time = time.time()
    audio_labels_list = []
    audio_raw_list = []
    for full_fname in tqdm(audio_files):
        file_name = Path(full_fname).name

        emo_label = parse_fname_to_only_emo_label(file_name)
        gen_label = parse_fname_to_gender_label(file_name)
        id_label = parse_fname_to_id_label(file_name)

        if emo_label is None or emo_label == '':
            continue

        audio_bin, sample_rate = librosa.load(full_fname, res_type='kaiser_fast')

        audio_labels_list.append([emo_label, gen_label, id_label])
        audio_raw_list.append(audio_bin)

    # audio_labels_df = audio_labels_df.append(Series([emo_label, gen_label]), ignore_index=True)
    # audio_raw_df = audio_raw_df.append(Series(audio_bin), ignore_index=True)
    # TODO maybe just using numpy array of arrays is faster
    print('Converting list to dataframes. This can take a while.')
    audio_raw_df = pd.DataFrame(audio_raw_list, )
    audio_labels_df = pd.DataFrame(audio_labels_list, )

    print("Time to extract Mel features:  %s seconds." % (time.time() - start_time))
    full_raw_audio_data_df = pd.concat([audio_raw_df, audio_labels_df], ignore_index=True, axis=1)

    # Setting NAs to 0 since audio waves have different sizes
    full_raw_audio_data_df.fillna(0, inplace=True)
    return full_raw_audio_data_df


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


def plot_mel_frequencies_from_bin(audio_bin, sample_rate=22050 * 2):
    import matplotlib.pyplot as plt
    import librosa.display

    fig, ax = plt.subplots()
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
            audio_bin, sample_rate = librosa.load(full_fname, duration=3, sr=22050 * 2, res_type='kaiser_fast',
                                                  offset=1.1)
        else:
            audio_bin, sample_rate = librosa.load(full_fname, duration=3, sr=22050 * 2, res_type='kaiser_fast')

        sample_rate = np.array(sample_rate)

        mfccs = librosa.feature.melspectrogram(y=audio_bin, sr=sample_rate, n_mels=128, n_fft=512, hop_length=512,
                                               win_length=512)

        mfccs = cv2.resize(mfccs, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

        # mfccs = librosa.feature.mfcc(y=audio_bin, sr=sample_rate, n_mfcc=64, dct_type=1)
        # mfccs_mean = np.mean(mfccs, axis=0)

        if mfccs.shape[1] > mfcc_size:
            mfcc_size = mfccs.shape[1]
            print(mfcc_size)

        # mfccs = librosa.power_to_db(mfccs, ref=100)
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

    # EMOVO file names
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

    # SAVEE file names
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

    # EMODB
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

    # if label is None:
    #     raise Exception('No emotion set')

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


def parse_fname_to_id_label(file_name):
    f_name_len = len(file_name)
    label = None
    # Emodb parse
    if f_name_len == 11:
        pass

    # Emovo parse
    elif f_name_len == 13:
        pass

    # Savee all male
    elif f_name_len <= 8:
        pass

    # Ravdess
    elif f_name_len == 24:
        label = file_name[18:-4]

    if label is None:
        raise Exception('No gender set')

    return label
