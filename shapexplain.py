import time
import os

import pandas as pd
import shap
import numpy as np
import tensorflow as tf
from pathlib import Path

from blume.table import table
from scipy.cluster.vq import whiten
from tqdm import tqdm
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from data_processing import pre_process_data, to_batchdataset
from experiment_neural_nets import get_obfuscation_model, get_obfuscation_model_swish, get_obfuscation_model_relu, \
    get_obfuscation_model_selu
from obfuscation_functions import *
from util.custom_functions import replace_outliers_by_std, mean_std_analysis, replace_outliers_by_quartile, \
    calc_confusion_matrix, priv_util_plot_perf_data, plot_obf_loss

from shap_experiment import extract_shap, extract_shap_values, parse_shap_values_by_class

from keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout, Activation
from sklearn.preprocessing import StandardScaler

from util.training_engine import train_step

tf.compat.v1.enable_v2_behavior()


def train_obfuscation_model(model, x_train_input, x_test_input, y_train_mdl1, y_test_mdl1,
                            y_train_mdl2, y_test_mdl2, obf_gender=True):

    # Convert to tensor batch iterators
    emo_tr_batchdt, emo_te_batchdt = to_batchdataset(x_train_input, x_test_input, y_train_mdl1, y_test_mdl1, batch_size)
    gen_tr_batchdt, gen_te_batchdt = to_batchdataset(x_train_input, x_test_input, y_train_mdl2, y_test_mdl2, batch_size)

    optimizer = tf.keras.optimizers.Adam()
    #optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.01)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0, decay=6e-8)

    loss_fn_emo = tf.keras.losses.CategoricalCrossentropy()
    loss_fn_gen = tf.keras.losses.BinaryCrossentropy()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    true_wron_train_loss = tf.keras.metrics.Mean(name="train_loss")
    estimated_train_loss = tf.keras.metrics.Mean(name="train_loss")

    emo_perf = []
    gen_perf = []
    final_loss_perf = []

    with tf.device('gpu:0'):
        for e in tqdm(range(epochs)):
            for (emo_train_x, emo_train_y), (gen_train_x, gen_train_y) in zip(emo_tr_batchdt, gen_tr_batchdt):
                loss, true_wrong_loss, estimated_wrong_loss = train_step(model,
                                                                          gender_model,
                                                                          emo_model,
                                                                          emo_train_x,
                                                                          gen_train_y,
                                                                          emo_train_y,
                                                                          optimizer,
                                                                          loss_fn_gen,
                                                                          loss_fn_emo,
                                                                          lambd,
                                                                          mdl_target)

                train_loss(loss)
                true_wron_train_loss(true_wrong_loss)
                estimated_train_loss(estimated_wrong_loss)

            final_loss_perf.append(loss.numpy())
            tf.print(train_loss.result())
            tf.print(true_wron_train_loss.result())
            tf.print(estimated_train_loss.result())

            train_loss.reset_states()
            true_wron_train_loss.reset_states()
            estimated_train_loss.reset_states()

            obf_input = obfuscate_input(model, x_test_input)
            mdl1_perf = emo_model.evaluate(obf_input, y_test_mdl1, verbose=0)
            mdl2_perf = gender_model.evaluate(obf_input, y_test_mdl2, verbose=0)
            emo_perf.append(mdl1_perf[1])
            gen_perf.append(mdl2_perf[1])

            # Plotting results.
            if e % 50 == 10:
                priv_util_plot_perf_data(gen_perf, emo_perf, "NN Obfuscator Performance")
                plot_obf_loss(final_loss_perf)
            if e % 100 == 90:
                calc_confusion_matrix(emo_model, obf_input, y_test_mdl1)
                calc_confusion_matrix(gender_model, obf_input, y_test_mdl2)
                # model.save(obf_model_path)

    return model


def obfuscate_input(model, x_test_input):
    # Generating the mask
    obf_masks = model.predict(x_test_input)
    nr_features = x_test_input[0].shape[0]
    mask_size = obf_masks[0].shape[0]
    padded_masks = np.pad(obf_masks, [(0, 0), (0, nr_features - mask_size)], mode='constant',
                          constant_values=0)
    # Adding the mask to the input
    obf_input = x_test_input + padded_masks
    return obf_input


def main():

    # get dataset
    print("Pre-processing audio files!")
    x_train_emo_cnn, y_train_emo_encoded, x_test_emo_cnn, y_test_emo_encoded = pre_process_data(audio_files_path, get_emotion_label=True)
    x_train_gen_cnn, y_train_gen_encoded, x_test_gen_cnn, y_test_gen_encoded = pre_process_data(audio_files_path, get_emotion_label=False)
    print("Pre-processing audio files Complete!")

    # Sanity check. These summations should be 0.
    train_equal_sum = np.sum(x_train_emo_cnn != x_train_gen_cnn)
    test_equal_sum = np.sum(x_test_emo_cnn != x_test_gen_cnn)

    if train_equal_sum + test_equal_sum != 0:
        raise Exception('Train and Test sets for different models do not match')

    # Squeeze extra dimension
    x_train_emo_cnn = np.squeeze(x_train_emo_cnn)
    x_test_emo_cnn = np.squeeze(x_test_emo_cnn)

    # Scaling and setting type to float32
    sc = StandardScaler()
    x_train_emo_cnn_scaled = sc.fit_transform(x_train_emo_cnn).astype(np.float32)
    x_test_emo_cnn_scaled = sc.transform(x_test_emo_cnn).astype(np.float32)

    model = get_obfuscation_model_selu()

    if not Path(obf_model_path).exists():
        model = train_obfuscation_model(model,
                                        x_train_emo_cnn_scaled,
                                        x_test_emo_cnn_scaled,
                                        y_train_emo_encoded,
                                        y_test_emo_encoded,
                                        y_train_gen_encoded,
                                        y_test_gen_encoded)
    else:
        model = tf.keras.models.load_model(obf_model_path)


if __name__ == "__main__":
    #emo_model_path = './emo_checkpoint/emodel_m2_all_aug_5k_16.h5'
    emo_model_path = "emo_checkpoint/emo_model_simple.h5"
    gender_model_path = "gmodel_checkpoint/gender_model_simple.h5"

    genderPrivacy = True
    # gender_model_path = './gmodel_checkpoint/gmodel_m2_all_aug_5k_16.h5'
    if genderPrivacy:
        model_path = 'emo_checkpoint/model_gender_simple.h5'
    else:
        model_path = 'emo_checkpoint/model_emo_simple.h5'

    obf_model_path = 'obf_checkpoint/model_obf.h5'

    # datasets
    audio_files_path = "./NNDatasets/audio"
    gen_shap_df_path = './data/gen_shap_df.npy'
    emo_shap_df_path = './data/emo_shap_df.npy'

    print("Loading trained Neural Nets")
    gender_model = load_model(gender_model_path)
    emo_model = load_model(emo_model_path)

    batch_size = 16
    epochs = 500
    max_iter = 1
    number_features = 40
    #metrics
    lambd = .8
    mdl_target = 1

    main()
