from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tqdm import tqdm

from data_processing import pre_process_data, to_batchdataset
from experiment_neural_nets import get_obfuscation_model_swish
from shap_experiment import extract_shap_values, parse_shap_values_by_class
from util.custom_functions import plot_obf_loss_from_list, calc_confusion_matrix, priv_util_plot_acc_data, priv_util_plot_f1_data, \
    collect_perf_metrics, obfuscate_input
from util.training_engine import train_step


def main():
    # get dataset
    print("Pre-processing audio files!")
    db_name = 'ravdess'

    x_train, x_test, y_tr_emo, y_te_emo, y_tr_gen, y_te_gen, y_tr_sv, y_te_sv = pre_process_data(audio_files_path, db_name)
    print("Pre-processing audio files Complete!")

    # Scaling and setting type to float32
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train).astype(np.float32)
    x_test_scaled = sc.transform(x_test).astype(np.float32)

    print("Loading shap values")
    # When using ranked outputs, the shapeley values are also sorted by rank (e.g., index 0 always has the shapeley of the model prediction)
    gen_shap_values = extract_shap_values(gen_shap_df_path, priv_gen_model, x_test_scaled, x_train_scaled, y_tr_gen)
    emo_shap_values = extract_shap_values(emo_shap_df_path, priv_emo_model, x_test_scaled, x_train_scaled, y_tr_emo)
    sv_shap_values = extract_shap_values(sv_shap_df_path, util_sv_model, x_test_scaled, x_train_scaled, y_tr_sv)

    # Isolating shap values by class.
    gen_gt_shap_list, gen_corr_shap_list = parse_shap_values_by_class(gen_shap_values, y_te_gen)
    emo_gt_shap_list, emo_corr_shap_list = parse_shap_values_by_class(emo_shap_values, y_te_emo)
    sv_gt_shap_list, sv_corr_shap_list = parse_shap_values_by_class(sv_shap_values, y_te_sv)

    # model_name = 'gender_model_cr'
    # export_shap_to_csv(gen_corr_shap_list, model_name)
    # model_name = 'emo_model_cr'
    # export_shap_to_csv(emo_corr_shap_list, model_name)

    # mean_std_analysis(gen_corr_shap_list)
    # mean_std_analysis(emo_corr_shap_list)
    # pclass_shap_list0 = gen_gt_shap_list[0]
    # pclass_shap_list1 = gen_gt_shap_list[1]

    shap_imp_order_emotion = summarize_shap_scores(emo_gt_shap_list)
    shap_imp_order_gen = summarize_shap_scores(gen_gt_shap_list)
    shap_imp_order_sv = summarize_shap_scores(sv_gt_shap_list)

    shap_data_dict = {"emo": shap_imp_order_emotion, "gen": shap_imp_order_gen, "sv": shap_imp_order_sv}

    # ------------ Util/Priv Definitions ----------------
    priv_g_labels = (y_tr_gen, y_te_gen)
    priv_e_labels = (y_tr_emo, y_te_emo)
    util_labels = (y_tr_sv, y_te_sv)

    # ------------ Util/Priv performance paths ----------------
    util_sv_perf_path = './data/nn_obfuscator_perf/sv_privacy/util_sv_data_cls{}_{}fts_e{}_util5.npy'
    priv_emo_perf_path = './data/nn_obfuscator_perf/sv_privacy/priv_emo_data_cls{}_{}fts_e{}_util5.npy'
    priv_gen_perf_path = './data/nn_obfuscator_perf/sv_privacy/priv_gen_data_cls{}_{}fts_e{}_util5.npy'

    for index, top_k_size in enumerate(top_k_sizes):
        global current_top_k
        current_top_k = top_k_size

        util_perf_path_full = set_file_name(top_k_size, util_sv_perf_path)
        priv_emo_perf_path_full = set_file_name(top_k_size, priv_emo_perf_path)
        priv_gen_perf_path_full = set_file_name(top_k_size, priv_gen_perf_path)

        if not Path(util_perf_path_full).exists() or not Path(priv_emo_perf_path_full).exists():
            # ------------ Util/Priv Definitions ----------------
            util_sv_perf_list, priv_e_perf_list, priv_g_perf_list = train_obfuscator_top_k_features(shap_data_dict,
                                                                                                    top_k_size,
                                                                                                    x_train_scaled,
                                                                                                    x_test_scaled,
                                                                                                    util_labels,
                                                                                                    priv_e_labels,
                                                                                                    priv_g_labels)

            if len(util_sv_perf_list) == 0:
                print("All features have been removed, stopping the experiment.")
                return
            elif util_sv_perf_list[0] == 9999:
                print("Private features already evaluated k={} skipping to k={}.".format(top_k_size,
                                                                                         top_k_sizes[index + 1]))
                continue

            np.save(util_perf_path_full, util_sv_perf_list, )
            np.save(priv_emo_perf_path_full, priv_e_perf_list, )
            np.save(priv_gen_perf_path_full, priv_g_perf_list, )
        else:
            print("Experiments with this configuration have been performed.")


def train_obfuscator_top_k_features(shap_data_dict, topk_size, x_train, x_test, util_labels, priv_e_labels,
                                    priv_g_labels):

    gen_shap_idxs = shap_data_dict["gen"]
    emo_shap_idxs = shap_data_dict["emo"]
    util_shap_idxs = shap_data_dict["sv"]

    if topk_size > 0:
        priv1_feature_mask = gen_shap_idxs[-topk_size:]
        priv2_feature_mask = emo_shap_idxs[-topk_size:]
        util_feature_mask = util_shap_idxs[-topk_size:]
    elif topk_size < 0:
        priv1_feature_mask = gen_shap_idxs[:-topk_size]
        priv2_feature_mask = emo_shap_idxs[:-topk_size]
        util_feature_mask = util_shap_idxs[:-topk_size]
    else:
        priv1_feature_mask = []
        priv2_feature_mask = []

    # Selecting top k features.
    features = [x for x in range(40)]
    priv_features = np.union1d(priv1_feature_mask, priv2_feature_mask)
    # priv_util_features = np.union1d(priv_features, util_feature_mask)

    # model_features = np.setdiff1d(features, priv_util_features)

    model_features = priv_features.astype(np.int32)

    model_features.flags.writeable = False
    features_map_hash = hash(model_features.data.tobytes())
    # Addressing when all features are removed or its a repeated combination
    # Using empty lists or lists with 999 different error types. TODO add proper exception handling.
    if len(model_features) == 0:
        return [], [], []
    elif features_map_hash in feature_map_hash:
        return [9999], [9999], [9999]
    else:
        feature_map_hash[hash(model_features.data.tobytes())] = None

    if mask_input:
        # Creating masked input for training the model.
        masked_x_train = x_train[:, model_features]
        masked_x_test = x_test[:, model_features]
        masked_input = (masked_x_train, masked_x_test)
    else:
        masked_input = (x_train, x_test)

    # Removing priv features from input
    x_train[:, model_features] = 0
    x_test[:, model_features] = 0

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

    nn_input_sz = masked_input[0].shape[1]
    nn_output_sz = model_features.shape[0]
    model = get_obfuscation_model_swish(nn_input_sz, nn_output_sz)

    save_model_meta_data(model_features, topk_size)

    # if not Path(obf_model_path).exists():
    obf_model, util_sv_perf_list, priv_e_perf_list, priv_g_perf_list = train_obfuscation_model(model,
                                                                                               model_features,
                                                                                               x_train,
                                                                                               x_test,
                                                                                               masked_input,
                                                                                               util_labels,
                                                                                               priv_e_labels,
                                                                                               priv_g_labels,
                                                                                               optimizer)
    # else:
    #     model = tf.keras.models.load_model(obf_model_path)

    return util_sv_perf_list, priv_e_perf_list, priv_g_perf_list


def train_obfuscation_model(obf_model, model_features, x_train_input, x_test_input, masked_input, util_sv_labels,
                            priv_e_labels, priv_g_labels, optimizer):

    nr_e_classes = priv_e_labels[0][0].shape[0]
    nr_g_classes = priv_g_labels[0][0].shape[0]
    nr_sv_classes = util_sv_labels[0][0].shape[0]

    # Creating labels for private models as (batch size X nr features).
    # Target values should match no better than random guess.
    priv_e_label = tf.constant(np.tile(np.ones(nr_e_classes) * 1 / nr_e_classes, (batch_size, 1)))
    priv_g_label = tf.constant(np.tile(np.ones(nr_g_classes) * 1 / nr_g_classes, (batch_size, 1)))

    model_features_tf = tf.constant(model_features)

    # Convert to tensor batch iterators
    # Both models will always share the same X train and X test.
    util_xy_tr_batch_dt = tf.data.Dataset.from_tensor_slices((x_train_input, util_sv_labels[0])).padded_batch(batch_size, drop_remainder=True)

    # Converting the feature specific input to tensors
    masked_x_tr_batch_dt, masked_x_te_batchdt = to_batchdataset(masked_input[0], masked_input[1], batch_size)

    util_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    priv_e_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    priv_g_loss_fn = tf.keras.losses.BinaryCrossentropy()

    total_tr_loss = tf.keras.metrics.Mean(name="total_train_loss")
    priv_e_loss = tf.keras.metrics.Mean(name="epriv_train_loss")
    priv_g_loss = tf.keras.metrics.Mean(name="gpriv_train_loss")
    priv_sv_loss = tf.keras.metrics.Mean(name="util_train_loss")

    util_sv_perf_list = [[], []]
    priv_e_perf_list = [[], []]
    priv_g_perf_list = [[], []]
    final_loss_perf = [[], [], [], [], [], [], []]

    with tf.device('cpu:0'):
        for e in tqdm(range(epochs)):
            for (x_tr_batch, util_tr_y_batch), masked_x_tr_batch in zip(util_xy_tr_batch_dt, masked_x_tr_batch_dt):
                tloss, peloss, pgloss, uloss, gradients, _ = train_step(obf_model,
                                                                        model_features_tf,
                                                                        priv_emo_model,
                                                                        priv_gen_model,
                                                                        util_sv_model,
                                                                        x_tr_batch,
                                                                        masked_x_tr_batch,
                                                                        util_tr_y_batch,
                                                                        priv_e_label,
                                                                        priv_g_label,
                                                                        util_loss_fn,
                                                                        priv_e_loss_fn,
                                                                        priv_g_loss_fn)

                optimizer.apply_gradients(zip(gradients, obf_model.trainable_variables))
                total_tr_loss(tloss)
                priv_e_loss(peloss)
                priv_g_loss(pgloss)
                priv_sv_loss(uloss)

            get_loss_per_epoch_info(final_loss_perf, priv_e_loss, priv_g_loss, priv_sv_loss, total_tr_loss)

            if (e + 1) % sample_results_rate == 0:

                obf_input = obfuscate_input(obf_model, masked_x_te_batchdt, x_test_input, model_features)

                sv_y_pred = collect_perf_metrics(util_sv_model, obf_input, util_sv_labels[1], util_sv_perf_list)
                emo_y_pred = collect_perf_metrics(priv_emo_model, obf_input, priv_e_labels[1], priv_e_perf_list)
                gen_y_pred = collect_perf_metrics(priv_gen_model, obf_input, priv_g_labels[1], priv_g_perf_list)

                # Plotting results.
                if (e + 1) % plot_epoch_rate == 0:
                    x_label = "Number of epochs"
                    priv_util_plot_acc_data(priv_e_perf_list[0], priv_g_perf_list[0], util_sv_perf_list[0],
                                            f'NN Obfuscator ACC Performance for {current_top_k}', x_label)
                    priv_util_plot_f1_data(priv_e_perf_list[1], priv_g_perf_list[1], util_sv_perf_list[1],
                                           f"NN Obfuscator F1 Performance {current_top_k}", x_label)

                    plot_obf_loss_from_list(final_loss_perf)

                if (e + 1) % plot_epoch_rate == 0:
                    calc_confusion_matrix(sv_y_pred, util_sv_labels[1], current_top_k)
                    calc_confusion_matrix(gen_y_pred, priv_g_labels[1], current_top_k)
                    calc_confusion_matrix(emo_y_pred, priv_e_labels[1], current_top_k)

                    save_model_data(obf_model, current_top_k)

    obf_model.reset_states()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    return obf_model, util_sv_perf_list, priv_e_perf_list, priv_g_perf_list


def get_loss_per_epoch_info(final_loss_perf, priv_e_loss, priv_g_loss, priv_sv_loss, total_tr_loss):

    final_loss_perf[0].append(total_tr_loss.result().numpy())
    final_loss_perf[1].append(priv_e_loss.result().numpy())
    final_loss_perf[2].append(priv_g_loss.result().numpy())
    final_loss_perf[3].append(priv_sv_loss.result().numpy())
    if print_loss:
        print("\n")
        print(f"Total Loss: {total_tr_loss.result().numpy()}")
        print(f"Emotion Loss: {priv_e_loss.result().numpy()}")
        print(f"Gender Loss: {priv_g_loss.result().numpy()}")
        print(f"SV Loss: {priv_sv_loss.result().numpy()}")
    total_tr_loss.reset_states()
    priv_e_loss.reset_states()
    priv_g_loss.reset_states()
    priv_sv_loss.reset_states()


def save_model_meta_data(model_features, topk_size):
    if topk_size >= 0:
        if topk_size <= 9:
            k_id = "+0" + str(topk_size)
        else:
            k_id = "+" + str(topk_size)
    else:
        if topk_size >= -9:
            k_id = "-0" + str(abs(topk_size))
        else:
            k_id = "-" + str(abs(topk_size))

    np.save(obf_model_meta_data.format(k_id), model_features)


def save_model_data(obf_model, topk_size):

    if topk_size >= 0:
        if topk_size <= 9:
            k_id = "+0" + str(topk_size)
        else:
            k_id = "+" + str(topk_size)
    else:
        if topk_size >= -9:
            k_id = "-0" + str(abs(topk_size))
        else:
            k_id = "-" + str(abs(topk_size))

    tf.saved_model.save(obf_model, obf_model_lite_path.format(k_id))
    obf_model.save(obf_model_keras_path.format(k_id))


def summarize_shap_scores(shap_scores_list):
    shap_all_classes_mean = np.mean(np.concatenate(shap_scores_list), axis=0)
    shap_all_classes_mean_abs = np.mean(np.abs(np.concatenate(shap_scores_list)), axis=0)
    shap_importance_sorted_by_mean = np.argsort(shap_all_classes_mean)
    shap_importance_sorted_by_mean_abs = np.argsort(shap_all_classes_mean_abs)

    # Debug
    temp = np.vstack((shap_importance_sorted_by_mean, shap_importance_sorted_by_mean_abs))

    return shap_importance_sorted_by_mean_abs


def create_label_mask(priv_classes, priv_labels):
    tr_priv_labels = priv_labels[0].copy()
    nr_classes = tr_priv_labels.shape[1]

    priv_label_output = np.ones(nr_classes) * 1 / nr_classes

    # If nr of classes is two or empty we should target all classes!
    if nr_classes <= 2:
        tr_priv_labels = np.ones(tr_priv_labels.shape) * priv_label_output
    else:
        class_mask_index = np.argmax(tr_priv_labels, axis=1)
        class_mask_row = np.zeros(tr_priv_labels.shape[0], dtype=bool)

        for priv_class in priv_classes:
            temp = class_mask_index == priv_class
            class_mask_row = np.logical_or(class_mask_row, temp)

        tr_priv_labels[class_mask_row] = priv_label_output

    return tr_priv_labels


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


def set_file_name(top_k_size, util_perf_path):
    util_perf_path_full = util_perf_path.format(str(priv_classes), top_k_size, epochs)
    return util_perf_path_full


def GetTopKFeatures(shap_data_dict, topk_size, x_test, x_train):
    # Get the Top K if positive
    p_shap_idxs = shap_data_dict["gen"]
    if topk_size > 0:
        priv_feature_mask = p_shap_idxs[-topk_size:]
    else:
        priv_feature_mask = p_shap_idxs[:topk_size]
    # Populating only top k/bot k features to be used by the obfuscator training
    masked_x_train = np.zeros(x_train.shape)
    masked_x_test = np.zeros(x_test.shape)
    masked_x_train[:, priv_feature_mask] = x_train[:, priv_feature_mask]
    masked_x_test[:, priv_feature_mask] = x_test[:, priv_feature_mask]
    return masked_x_test, masked_x_train


if __name__ == "__main__":
    emo_model_path = "emo_checkpoint/emodel_scalarized_ravdess.h5"
    id_model_path = "sv_model_checkpoint/sver_model_scalarized_data.h5"
    gender_model_path = "gmodel_checkpoint/gmodel_scaled_ravdess.h5"

    obf_model_lite_path = 'obf_checkpoint/lite/model_obf_k_{}_util5'
    obf_model_keras_path = 'obf_checkpoint/model_obf_k_{}_util5'
    obf_model_meta_data = 'obf_checkpoint/model_obf_meta_k_{}_util5'

    # datasets
    audio_files_path = "./NNDatasets/audio"
    gen_shap_df_path = './data/shapeley/gen_shap_df.npy'
    emo_shap_df_path = './data/shapeley/emo_shap_df.npy'
    sv_shap_df_path = './data/shapeley/id_shap_df.npy'

    print("Loading trained Neural Nets")

    util_sv_model = load_model(id_model_path)
    priv_emo_model = load_model(emo_model_path)
    priv_gen_model = load_model(gender_model_path)

    # ------------- Hyperparameters -------------
    lambds = [.95]
    # lambds = [x/10 for x in range(1, 10)]

    priv_classes = []
    top_k_sizes = [-x for x in range(0, 40, 1)]
    current_top_k = 0
    top_k_sizes = [5, 10]

    # Reduce input features to match top K restrictions
    mask_input = False

    batch_size = 32
    epochs = 300
    sample_results_rate = 10
    print_loss = False
    # metrics
    plot_epoch_rate = 50

    feature_map_hash = {}

    main()
