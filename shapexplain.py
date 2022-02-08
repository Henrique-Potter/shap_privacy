import time
import os

import pandas as pd
import shap
import numpy as np
import tensorflow as tf
from pathlib import Path

from tqdm import tqdm
from tensorflow.keras.models import load_model
from data_processing import pre_process_data, to_batchdataset
from experiment_neural_nets import get_obfuscation_model, get_obfuscation_model_swish, get_obfuscation_model_relu, \
    get_obfuscation_model_selu, get_obfuscation_model_tanh2

from shap_experiment import extract_shap_values, parse_shap_values_by_class
from sklearn.preprocessing import StandardScaler

from util.custom_functions import priv_util_plot_perf_data, plot_obf_loss, calc_confusion_matrix
from util.training_engine import train_step


def train_obfuscation_model(obf_model, x_train_input, x_test_input, masked_input, util_labels, priv_labels, optimizer, lambd):

    # Convert to tensor batch iterators
    # Both models will always share the same X train and X test.
    util_xy_tr_batch_dt = tf.data.Dataset.from_tensor_slices((x_train_input, util_labels[0])).batch(batch_size)
    priv_y_tr_batch_dt = tf.data.Dataset.from_tensor_slices(priv_labels[0]).batch(batch_size)

    # Converting the feature specific input to tensors
    masked_x_tr_batch_dt, masked_x_te_batchdt = to_batchdataset(masked_input[0], masked_input[1], batch_size)

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
            # emo_train_x = gen_train_x
            for (x_tr_batch, util_tr_y_batch), priv_tr_y_batch, masked_x_tr_batch in zip(util_xy_tr_batch_dt,
                                                                                         priv_y_tr_batch_dt,
                                                                                         masked_x_tr_batch_dt):

                tloss, ploss, uloss, gradients, _ = train_step(obf_model,
                                                               priv_model,
                                                               util_model,
                                                               x_tr_batch,
                                                               masked_x_tr_batch,
                                                               priv_tr_y_batch,
                                                               util_tr_y_batch,
                                                               loss_fn_gen,
                                                               loss_fn_emo,
                                                               lambd)

                optimizer.apply_gradients(zip(gradients, obf_model.trainable_variables))
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

            obf_input = obfuscate_input(obf_model, masked_x_te_batchdt, x_test_input)
            util_perf = util_model.evaluate(obf_input, util_labels[1], verbose=0)
            # mdl1_perf2 = emo_model.evaluate(x_test_input, y_test_mdl1, verbose=1)
            priv_perf = priv_model.evaluate(obf_input, priv_labels[1], verbose=0)
            emo_perf.append(util_perf[1])
            gen_perf.append(priv_perf[1])
            final_loss_perf[3].append(util_perf[0])
            final_loss_perf[4].append(priv_perf[0])

            # Plotting results.
            if (e+1) % plot_at_epoch == 0:
                priv_util_plot_perf_data(gen_perf, emo_perf, "NN Obfuscator Performance")
                plot_obf_loss(final_loss_perf)
            if (e+1) % plot_at_epoch == 0:
                calc_confusion_matrix(priv_model, obf_input, priv_labels[1])
                calc_confusion_matrix(util_model, obf_input, util_labels[1])
                # model.save(obf_model_path)

    obf_model.reset_states()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    opt_reset = tf.group([v.initializer for v in optimizer.variables()])
    return obf_model, np.array(emo_perf), np.array(gen_perf)


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
    x_train, x_test, y_train_emo, y_test_emo, y_train_gen, y_test_gen = pre_process_data(audio_files_path,)
    print("Pre-processing audio files Complete!")

    # Scaling and setting type to float32
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train).astype(np.float32)
    x_test_scaled = sc.transform(x_test).astype(np.float32)

    gender_nr_classes = len(y_train_gen[0])
    emo_nr_classes = len(y_train_emo[0])
    print("Loading shap values")
    # When using ranked outputs, the shapeley values are also sorted by rank (e.g., index 0 always has the shapeley of the model prediction)
    gen_shap_values = extract_shap_values(gen_shap_df_path, priv_model, x_test_scaled, x_train_scaled, gender_nr_classes)
    emo_shap_values = extract_shap_values(emo_shap_df_path, util_model, x_test_scaled, x_train_scaled, emo_nr_classes)

    # Isolating shap values by class.
    gen_gt_shap_list, gen_corr_shap_list = parse_shap_values_by_class(gen_shap_values, y_test_gen)
    emo_gt_shap_list, emo_corr_shap_list = parse_shap_values_by_class(emo_shap_values, y_test_emo)

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

    g_pclass_shap_mean0 = np.mean(np.concatenate(emo_gt_shap_list), axis=0)
    g_pclass_shap_mean_abs = np.mean(np.abs(np.concatenate(emo_gt_shap_list)), axis=0)
    g_p_shap_mean_sorted_idxs = np.argsort(g_pclass_shap_mean0)
    g_p_shap_mean_sorted_abs_idxs = np.argsort(g_pclass_shap_mean_abs)
    gtemp = np.vstack((g_p_shap_mean_sorted_idxs, g_p_shap_mean_sorted_abs_idxs))

    e_pclass_shap_mean0 = np.mean(np.concatenate(gen_gt_shap_list), axis=0)
    e_pclass_shap_mean_abs = np.mean(np.abs(np.concatenate(gen_gt_shap_list)), axis=0)
    e_p_shap_mean_sorted_idxs = np.argsort(e_pclass_shap_mean0)
    e_p_shap_mean_sorted_abs_idxs = np.argsort(e_pclass_shap_mean_abs)
    temp = np.vstack((e_p_shap_mean_sorted_idxs, e_p_shap_mean_sorted_abs_idxs))

    lambds = [.7]

    for lambd in lambds:

        emo_perf_path = './data/nn_obfuscator_perf/gender_privacy/emo_data_l'+str(lambd)+'_40ft_{}.npy'
        gen_perf_path = './data/nn_obfuscator_perf/gender_privacy/gen_data_l'+str(lambd)+'_40ft_{}.npy'

        # top_k_sizes = [x for x in range(1, 41, 2)]
        top_k_sizes = [40]

        for top_k_size in top_k_sizes:
            # top_k_experiment = -top_k_experiment
            emo_perf_path_full = emo_perf_path.format(top_k_size)
            gen_perf_path_full = gen_perf_path.format(top_k_size)

            util_labels = (y_train_emo, y_test_emo)
            priv_labels = (y_train_gen, y_test_gen)

            if not Path(emo_perf_path_full).exists() or not Path(gen_perf_path_full).exists():
                emo_perf, gen_perf = train_obfuscator_top_k_features(e_p_shap_mean_sorted_abs_idxs,
                                                                     top_k_size,
                                                                     x_train_scaled,
                                                                     x_test_scaled,
                                                                     util_labels,
                                                                     priv_labels,
                                                                     lambd)

                np.save(emo_perf_path_full, emo_perf,)
                np.save(gen_perf_path_full, gen_perf,)


def train_obfuscator_top_k_features(p_shap_idxs, topk_size, x_train, x_test,
                                    util_labels, priv_labels, lambd):
    # Get the Top K is positive
    if topk_size > 0:
        priv_feature_mask = p_shap_idxs[-topk_size:]
    else:
        priv_feature_mask = p_shap_idxs[:-topk_size]

    # Populating only top k/bot k features to be used for obfuscator training
    masked_x_train = np.zeros(x_train.shape)
    masked_x_test = np.zeros(x_test.shape)
    masked_x_train[:, priv_feature_mask] = x_train[:, priv_feature_mask]
    masked_x_test[:, priv_feature_mask] = x_test[:, priv_feature_mask]

    masked_input = (masked_x_train, masked_x_test)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    # optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.01)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0, decay=6e-8)

    nn_input_sz = masked_x_train.shape
    model = get_obfuscation_model_swish(nn_input_sz[1])

    if not Path(obf_model_path).exists():
        model, emo_perf, gen_perf = train_obfuscation_model(model,
                                                            x_train,
                                                            x_test,
                                                            masked_input,
                                                            util_labels,
                                                            priv_labels,
                                                            optimizer,
                                                            lambd)
    else:
        model = tf.keras.models.load_model(obf_model_path)

    return emo_perf, gen_perf


if __name__ == "__main__":

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
    # priv_model = load_model(gender_model_path)
    priv_model = load_model(emo_model_path)
    # util_model = load_model(emo_model_path)
    util_model = load_model(gender_model_path)

    batch_size = 128
    epochs = 500

    #metrics
    plot_at_epoch = 49

    main()
