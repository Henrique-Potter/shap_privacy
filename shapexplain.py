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
    get_obfuscation_model_selu, get_obfuscation_model_tanh2
from obfuscation_functions import *
from util.custom_functions import replace_outliers_by_std, mean_std_analysis, replace_outliers_by_quartile, \
    calc_confusion_matrix, priv_util_plot_perf_data, plot_obf_loss

from shap_experiment import extract_shap, extract_shap_values, parse_shap_values_by_class, export_shap_to_csv

from keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout, Activation
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from mpl_toolkits.mplot3d import Axes3D

from util.training_engine import train_step

# tf.compat.v1.enable_v2_behavior()
# tf.config.run_functions_eagerly(True)

def train_obfuscation_model(model, x_train_input, x_test_input, x_train_priv, x_test_priv, y_train_mdl1, y_test_mdl1,
                            y_train_mdl2, y_test_mdl2, optimizer, obf_gender=True):

    # Convert to tensor batch iterators
    emo_tr_batchdt, emo_te_batchdt = to_batchdataset(x_train_input, x_test_input, batch_size, y_train_mdl1, y_test_mdl1)
    gen_tr_batchdt, gen_te_batchdt = to_batchdataset(x_train_input, x_test_input, batch_size, y_train_mdl2, y_test_mdl2)

    # Convert to tensor batch iterators
    emo_tr_batchdt_slice, emo_te_batchdt_slice = to_batchdataset(x_train_priv, x_test_priv, batch_size)

    loss_fn_emo = tf.keras.losses.CategoricalCrossentropy()
    loss_fn_gen = tf.keras.losses.BinaryCrossentropy()

    train_loss = tf.keras.metrics.Mean(name="train_loss")

    emo_perf = []
    gen_perf = []
    final_loss_perf = []

    with tf.device('gpu:0'):
        for e in tqdm(range(epochs)):
            for (emo_train_x, emo_train_y), (_, gen_train_y), emo_train_x_slice in zip(emo_tr_batchdt, gen_tr_batchdt, emo_tr_batchdt_slice):

                loss, gradients = train_step(model,
                                  gender_model,
                                  emo_model,
                                  emo_train_x,
                                  emo_train_x_slice,
                                  gen_train_y,
                                  emo_train_y,
                                  optimizer,
                                  loss_fn_gen,
                                  loss_fn_emo,
                                  lambd,
                                  mdl_target)

                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss(loss)

            final_loss_perf.append(loss.numpy())
            tf.print(train_loss.result())

            train_loss.reset_states()

            obf_input = obfuscate_input(model, x_test_priv, x_test_input)
            mdl1_perf = emo_model.evaluate(obf_input, y_test_mdl1, verbose=0)
            # mdl1_perf2 = emo_model.evaluate(x_test_input, y_test_mdl1, verbose=1)
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

    model.reset_states()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    return model, np.array(emo_perf), np.array(gen_perf)


def obfuscate_input(model, obfuscator_x_input, clean_x_innput):
    # Generating the mask
    obf_masks = model.predict(obfuscator_x_input)
    nr_features = clean_x_innput[0].shape[0]
    mask_size = obf_masks[0].shape[0]
    padded_masks = np.pad(obf_masks, [(0, 0), (0, nr_features - mask_size)], mode='constant',
                          constant_values=0)
    # Adding the mask to the input
    obf_input = clean_x_innput + padded_masks
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

    gender_nr_classes = len(y_train_gen_encoded[0])
    emo_nr_classes = len(y_train_emo_encoded[0])
    print("Loading shap values")
    # When using ranked outputs, the shapeley values are also sorted by rank (e.g., index 0 always has the shapeley of the model prediction)
    gen_shap_values = extract_shap_values(gen_shap_df_path, gender_model, x_test_emo_cnn_scaled, x_train_emo_cnn_scaled, gender_nr_classes)
    emo_shap_values = extract_shap_values(emo_shap_df_path, emo_model, x_test_emo_cnn_scaled, x_train_emo_cnn_scaled, emo_nr_classes)

    # Isolating shap values by class.
    gen_gt_shap_list, gen_corr_shap_list = parse_shap_values_by_class(gen_shap_values, y_test_gen_encoded)
    emo_gt_shap_list, emo_corr_shap_list = parse_shap_values_by_class(emo_shap_values, y_test_emo_encoded)

    # model_name = 'gender_model_gt'
    # export_shap_to_csv(gen_gt_shap_list, model_name)
    # model_name = 'gender_model_cr'
    # export_shap_to_csv(gen_corr_shap_list, model_name)
    #
    # model_name = 'emo_mo del_gt'
    # export_shap_to_csv(emo_gt_shap_list, model_name)
    # model_name = 'emo_model_cr'
    # export_shap_to_csv(emo_corr_shap_list, model_name)
    #
    # mean_std_analysis(gen_gt_shap_list)
    # mean_std_analysis(gen_corr_shap_list)
    # mean_std_analysis(emo_gt_shap_list)
    # mean_std_analysis(emo_corr_shap_list)

    # pclass_shap_list0 = gen_gt_shap_list[0]
    # pclass_shap_list1 = gen_gt_shap_list[1]

    pclass_shap_mean0 = np.mean(np.concatenate(gen_gt_shap_list), axis=0)
    pclass_shap_mean_abs = np.mean(np.abs(np.concatenate(gen_gt_shap_list)), axis=0)
    p_shap_mean_sorted_idxs = np.argsort(pclass_shap_mean0)


    topk_size = 2
    topk_step = 1
    nr_topk_experiments = 40/topk_step
    topk_emo_perf = []
    topk_gen_perf = []

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # optimizers = [tf.keras.optimizers.Adam(learning_rate=0.0001, name='opt{}'.format(x)) for x in range(40)]

    for top_k_experiment in range(0, 40, topk_step):
        # optimizer = optimizers[top_k_experiment]

        # optimizer_state = [optimizer.iterations, optimizer.lr, optimizer.beta_1,
        #                    optimizer.beta_2, optimizer.decay]
        # optimizer_reset = tf.variables_initializer(optimizer_state)

        # Later when you want to reset the optimizer
        # K.get_session().run(optimizer_reset)

        emo_perf, gen_perf = train_obfuscator_top_k_features(p_shap_mean_sorted_idxs,
                                                             top_k_experiment,
                                                             x_test_emo_cnn_scaled,
                                                             x_train_emo_cnn_scaled,
                                                             y_test_emo_encoded,
                                                             y_test_gen_encoded,
                                                             y_train_emo_encoded,
                                                             y_train_gen_encoded)
        topk_emo_perf.append(emo_perf)
        topk_gen_perf.append(gen_perf)

    fig = plt.figure()
    ax = Axes3D(fig)

    X = np.arange(0, epochs)
    Y = np.arange(0, nr_topk_experiments)
    X, Y = np.meshgrid(X, Y)
    Z = np.vstack(topk_gen_perf)

    ax.plot_surface(X, Y, Z, )

    plt.show()



def train_obfuscator_top_k_features(p_shap_mean_sorted_idxs, topk_size, x_test_emo_cnn_scaled, x_train_emo_cnn_scaled,
                                    y_test_emo_encoded, y_test_gen_encoded, y_train_emo_encoded, y_train_gen_encoded):
    # Get the Top K is positive
    priv_feature_mask = p_shap_mean_sorted_idxs[-topk_size:]
    x_train_emo_cnn_priv = np.zeros(x_train_emo_cnn_scaled.shape)
    x_test_emo_cnn_priv = np.zeros(x_test_emo_cnn_scaled.shape)
    x_train_emo_cnn_priv[:, priv_feature_mask] = x_train_emo_cnn_scaled[:, priv_feature_mask]
    x_test_emo_cnn_priv[:, priv_feature_mask] = x_test_emo_cnn_scaled[:, priv_feature_mask]
    nn_input_sz = x_train_emo_cnn_priv.shape

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.01)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0, decay=6e-8)

    model = get_obfuscation_model_swish(nn_input_sz[1])

    if not Path(obf_model_path).exists():
        model, emo_perf, gen_perf = train_obfuscation_model(model,
                                                            x_train_emo_cnn_scaled,
                                                            x_test_emo_cnn_scaled,
                                                            x_train_emo_cnn_priv,
                                                            x_test_emo_cnn_priv,
                                                            y_train_emo_encoded,
                                                            y_test_emo_encoded,
                                                            y_train_gen_encoded,
                                                            y_test_gen_encoded, optimizer)
    else:
        model = tf.keras.models.load_model(obf_model_path)

    return emo_perf, gen_perf


if __name__ == "__main__":
    # emo_model_path = "emo_checkpoint/emodel_scalarized_data.h5"
    # gender_model_path = "gmodel_checkpoint/gmodel_scaled_16.h5"

    emo_model_path = "emo_checkpoint/emo_model_simple.h5"
    gender_model_path = "gmodel_checkpoint/gender_model_simple.h5"

    genderPrivacy = True
    if genderPrivacy:
        model_path = 'emo_checkpoint/model_gender_simple.h5'
    else:
        model_path = 'emo_checkpoint/model_emo_simple.h5'

    obf_model_path = 'obf_checkpoint/model_obf.h5'

    # datasets
    audio_files_path = "./NNDatasets/audio"
    gen_shap_df_path = './data/shapeley/gen_shap_df.npy'
    emo_shap_df_path = './data/shapeley/emo_shap_df.npy'

    print("Loading trained Neural Nets")
    gender_model = load_model(gender_model_path)
    emo_model = load_model(emo_model_path)

    batch_size = 64
    epochs = 2
    max_iter = 1
    number_features = 40
    #metrics
    lambd = .8
    mdl_target = 1

    main()
