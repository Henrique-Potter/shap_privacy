import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import sys
import os
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.models import Sequential

import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model_id = 0

project_folder = os.path.join(os.path.dirname(__file__), "shap_privacy")
print(f"Project found at {project_folder}")

sys.path.insert(1, os.path.join(os.path.dirname(__file__), "shap_privacy"))


def main():

    audio_files_path = "../inputs/1/0"
    db_name = 'ravdess'

    print("Pre-processing audio files!")
    x_train_mfcc, x_test_mfcc, y_emo_train_encoded, y_emo_test_encoded, _, _, _, _ = pre_process_data(audio_files_path,
                                                                                                      db_name)
    print("Pre-processing audio files Complete!")

    # Scaling and setting type to float32
    sc = StandardScaler()
    x_train_cnn_scaled = sc.fit_transform(x_train_mfcc)
    x_test_cnn_scaled = sc.transform(x_test_mfcc)

    print("Building Neural Net")
    model = build_emo_model_swish(x_train_cnn_scaled)

    y_train = y_emo_train_encoded
    y_test = y_emo_test_encoded

    epochs = 10
    batch_size = 64

    print("Starting model training!")
    train_model(model, batch_size, epochs, x_train_cnn_scaled, y_train, x_test_cnn_scaled, y_test)

    test_acc = model.evaluate(x_test_cnn_scaled, y_test, batch_size=16)
    train_acc = model.evaluate(x_train_cnn_scaled, y_train, batch_size=16)
    print("Emo Model Train perf is:{}, Test perf is:{}".format(train_acc, test_acc))


def build_emo_model_swish(input_sample):
    input_shape_width = input_sample.shape[1]
    # input_shape_channels = input_sample.shape[2]

    model = Sequential()

    model.add(Dense(192, input_shape=(input_shape_width,), ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(160, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(128, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(96, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(64, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(32, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(7))
    model.add(Activation('softmax'))
    opt = optimizers.Adam(learning_rate=0.0001)

    loss_fn_emo = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fn_emo, optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def pre_process_data(audio_files_path, db_name, n_mfcc=40):

    train_x_mfcc_path = '{}/audio_train_data_mfcc{}_{}_np.npy'.format(audio_files_path, n_mfcc, db_name)
    test_x_mfcc_path = '{}/audio_test_data_mfcc{}_{}_np.npy'.format(audio_files_path, n_mfcc, db_name)

    train_y_emo_mfcc_path = '{}/audio_emo_train_y_data_{}_{}_np.npy'.format(audio_files_path, n_mfcc, db_name)
    test_y_emo_mfcc_path = '{}/audio_emo_test_y_data_{}_{}_np.npy'.format(audio_files_path, n_mfcc, db_name)

    train_y_gen_mfcc_path = '{}/audio_gen_train_y_data_{}_{}_np.npy'.format(audio_files_path, n_mfcc, db_name)
    test_y_gen_mfcc_path = '{}/audio_gen_test_y_data_{}_{}_np.npy'.format(audio_files_path, n_mfcc, db_name)

    train_y_sv_mfcc_path = '{}/audio_sv_train_y_data_{}_{}_np.npy'.format(audio_files_path, n_mfcc, db_name)
    test_y_sv_mfcc_path = '{}/audio_sv_test_y_data_{}_{}_np.npy'.format(audio_files_path, n_mfcc, db_name)


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


def train_model(model, batch, epoch, x_traincnn, y_train, x_testcnn, y_test):
    model_performance_data = model.fit(x_traincnn, y_train, batch_size=batch, epochs=epoch, validation_data=(x_testcnn, y_test))

    figure, axis = plt.subplots(2)
    axis[0].plot(model_performance_data.history['loss'])
    axis[0].plot(model_performance_data.history['val_loss'])
    axis[0].set_title('Loss')
    axis[0].set_ylabel('loss')
    axis[0].set_xlabel('epoch')
    axis[0].legend(['train', 'test'], loc='upper left')
    axis[1].plot(model_performance_data.history['accuracy'])
    axis[1].plot(model_performance_data.history['val_accuracy'])
    axis[1].set_title('Accuracy')
    axis[1].set_ylabel('accuracy')
    axis[1].set_xlabel('epoch')
    axis[1].legend(['train', 'test'], loc='upper left')
    plt.subplots_adjust(hspace=0.7)


if __name__ == '__main__':
    main()
