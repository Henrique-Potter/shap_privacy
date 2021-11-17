import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import time


def pre_process_data(audio_files, get_emotion_label):

    audio_mel_df_path = 'mel_df.pkl'

    if get_emotion_label:
        audio_label_df_path = 'emo_label_df.pkl'
    else:
        audio_label_df_path = 'gender_label_df.pkl'

    if not Path(audio_mel_df_path).exists() or not Path(audio_label_df_path).exists():

        mel_features_df, labels_df = extract_mel_features(audio_files, get_emotion_label)
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


def extract_mel_features(audio_files, get_emotion_label):
    from pandas import Series

    mel_feature_df = pd.DataFrame()
    label_list = []
    start_time = time.time()

    for index, full_fname in enumerate(audio_files):
        file_name = Path(full_fname).name
        if file_name[6:-16] != '01' and file_name[6:-16] != '08' \
                and file_name[:2] != 'su' and file_name[:1] != 'n' and file_name[:1] != 'd' and file_name[6:-16] != '07':

            if get_emotion_label:
                label_list.append(parse_fname_to_emo_label(file_name))
            else:
                label_list.append(parse_fname_to_gender_label(file_name))

            audio_bin, sample_rate = librosa.load(full_fname, duration=3, sr=22050 * 2, offset=1.1)
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
    return mel_feature_df, pd.DataFrame(label_list)


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
    # elif file_name[:1]=='n':
    # label = 'neutral')
    elif file_name[:2] == 'sa':
        label = 'male_sad'

    return label


def parse_fname_to_gender_label(file_name):

    if int(file_name[18:-4]) % 2 == 1:
        label = 'male'
    else:
        label = 'female'

    return label

