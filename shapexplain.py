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
import matplotlib as mpl

from util.training_engine import train_step

# tf.compat.v1.enable_v2_behavior()
# tf.config.run_functions_eagerly(True)


def train_obfuscation_model(model, x_train_input, x_test_input, x_train_priv, x_test_priv, y_train_mdl1, y_test_mdl1,
                            y_train_mdl2, y_test_mdl2, optimizer, lambd, obf_gender=True):

    # Convert to tensor batch iterators
    emo_tr_batchdt, emo_te_batchdt = to_batchdataset(x_train_input, x_test_input, batch_size, y_train_mdl1, y_test_mdl1)
    gen_tr_batchdt, gen_te_batchdt = to_batchdataset(x_train_input, x_test_input, batch_size, y_train_mdl2, y_test_mdl2)

    # Convert to tensor batch iterators
    emo_tr_batchdt_slice, emo_te_batchdt_slice = to_batchdataset(x_train_priv, x_test_priv, batch_size)

    loss_fn_emo = tf.keras.losses.CategoricalCrossentropy()
    loss_fn_gen = tf.keras.losses.BinaryCrossentropy()
    # loss_fn_gen = tf.keras.losses.KLDivergence()

    total_tr_loss = tf.keras.metrics.Mean(name="total_train_loss")
    priv_loss = tf.keras.metrics.Mean(name="priv_train_loss")
    util_loss = tf.keras.metrics.Mean(name="util_train_loss")

    emo_perf = []
    gen_perf = []
    final_loss_perf = [[], [], [], [], []]

    with tf.device('gpu:0'):
        for e in tqdm(range(epochs)):
            for (emo_train_x, emo_train_y), (_, gen_train_y), emo_train_x_slice in zip(emo_tr_batchdt,
                                                                                       gen_tr_batchdt,
                                                                                       emo_tr_batchdt_slice):

                tloss, ploss, uloss, gradients, _ = train_step(model,
                                                                             gender_model,
                                                                             emo_model,
                                                                             emo_train_x,
                                                                             emo_train_x_slice,
                                                                             gen_train_y,
                                                                             emo_train_y,
                                                                             loss_fn_gen,
                                                                             loss_fn_emo,
                                                                             lambd,
                                                                             mdl_target)

                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                total_tr_loss(tloss)
                priv_loss(ploss)
                util_loss(uloss)

            final_loss_perf[0].append(total_tr_loss.result().numpy())
            final_loss_perf[1].append(priv_loss.result().numpy())
            final_loss_perf[2].append(util_loss.result().numpy())

            tf.print(total_tr_loss.result())

            # print_gen_logits(gen_train_y, priv_mdl_logits)

            total_tr_loss.reset_states()
            priv_loss.reset_states()
            util_loss.reset_states()

            obf_input = obfuscate_input(model, x_test_priv, x_test_input)
            mdl1_perf = emo_model.evaluate(obf_input, y_test_mdl1, verbose=0)
            # mdl1_perf2 = emo_model.evaluate(x_test_input, y_test_mdl1, verbose=1)
            mdl2_perf = gender_model.evaluate(obf_input, y_test_mdl2, verbose=0)
            emo_perf.append(mdl1_perf[1])
            gen_perf.append(mdl2_perf[1])
            final_loss_perf[3].append(mdl1_perf[0])
            final_loss_perf[4].append(mdl2_perf[0])

            # # Plotting results.
            # if (e+1) % plot_at_epoch == 0:
            #     priv_util_plot_perf_data(gen_perf, emo_perf, "NN Obfuscator Performance")
            #     plot_obf_loss(final_loss_perf)
            # if (e+1) % plot_at_epoch == 0:
            #     calc_confusion_matrix(emo_model, obf_input, y_test_mdl1)
            #     calc_confusion_matrix(gender_model, obf_input, y_test_mdl2)
            #     # model.save(obf_model_path)

    model.reset_states()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    opt_reset = tf.group([v.initializer for v in optimizer.variables()])
    return model, np.array(emo_perf), np.array(gen_perf)


def print_gen_logits(gen_train_y, priv_mdl_logits):
    true_labels = np.argmax(gen_train_y.numpy(), axis=1)
    male = true_labels == 0
    female = true_labels == 1
    male_logits = priv_mdl_logits.numpy()[male]
    female_logits = priv_mdl_logits.numpy()[female]
    male_logit_avg = np.mean(male_logits, axis=0)
    female_logit_avg = np.mean(female_logits, axis=0)
    print(male_logit_avg)
    print(female_logit_avg)


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
    p_shap_mean_sorted_abs_idxs = np.argsort(pclass_shap_mean_abs)
    temp = np.vstack((p_shap_mean_sorted_idxs, p_shap_mean_sorted_abs_idxs))

    lambds = [.2, .5, .7, .9]

    for lambd in lambds:

        emo_perf_path = './data/nn_obfuscator_perf/gender_privacy/emo_data_l'+str(lambd)+'_40ft_{}.npy'
        gen_perf_path = './data/nn_obfuscator_perf/gender_privacy/gen_data_l'+str(lambd)+'_40ft_{}.npy'

        for top_k_size in range(1, 41, 2):
            # top_k_experiment = -top_k_experiment
            emo_perf_path_full = emo_perf_path.format(top_k_size)
            gen_perf_path_full = gen_perf_path.format(top_k_size)

            if not Path(emo_perf_path_full).exists() or not Path(gen_perf_path_full).exists():
                emo_perf, gen_perf = train_obfuscator_top_k_features(p_shap_mean_sorted_abs_idxs,
                                                                     top_k_size,
                                                                     x_test_emo_cnn_scaled,
                                                                     x_train_emo_cnn_scaled,
                                                                     y_test_emo_encoded,
                                                                     y_test_gen_encoded,
                                                                     y_train_emo_encoded,
                                                                     y_train_gen_encoded, lambd)

                np.save(emo_perf_path_full, emo_perf,)
                np.save(gen_perf_path_full, gen_perf,)


def train_obfuscator_top_k_features(p_shap_idxs, topk_size, x_test_emo_scaled, x_train_emo_scaled,
                                    y_test_emo_encoded, y_test_gen_encoded, y_train_emo_encoded, y_train_gen_encoded, lambd):
    # Get the Top K is positive
    if topk_size > 0:
        priv_feature_mask = p_shap_idxs[-topk_size:]
    else:
        priv_feature_mask = p_shap_idxs[:-topk_size]

    # Populating only top k/bot k features to be used for obfuscator trainning
    x_train_emo_priv = np.zeros(x_train_emo_scaled.shape)
    x_test_emo_priv = np.zeros(x_test_emo_scaled.shape)
    x_train_emo_priv[:, priv_feature_mask] = x_train_emo_scaled[:, priv_feature_mask]
    x_test_emo_priv[:, priv_feature_mask] = x_test_emo_scaled[:, priv_feature_mask]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    # optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.01)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0, decay=6e-8)

    nn_input_sz = x_train_emo_priv.shape
    model = get_obfuscation_model_swish(nn_input_sz[1])

    if not Path(obf_model_path).exists():
        model, emo_perf, gen_perf = train_obfuscation_model(model,
                                                            x_train_emo_scaled,
                                                            x_test_emo_scaled,
                                                            x_train_emo_priv,
                                                            x_test_emo_priv,
                                                            y_train_emo_encoded,
                                                            y_test_emo_encoded,
                                                            y_train_gen_encoded,
                                                            y_test_gen_encoded, optimizer, lambd)
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

    batch_size = 128
    epochs = 50
    max_iter = 1
    number_features = 40

    #metrics
    mdl_target = 1
    plot_at_epoch = 49

    main()
