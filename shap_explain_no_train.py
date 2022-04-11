from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tqdm import tqdm
from data_processing import pre_process_data

from shap_experiment import extract_shap_values, parse_shap_values_by_class
from util.custom_functions import collect_perf_metrics


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

    util_sv_perf_list = []
    priv_e_perf_list =[]
    priv_g_perf_list =[]
    model_features_list = []
    top_k_list = []

    # ------------ Util/Priv performance paths ----------------
    util_sv_perf_path = f'./data/nn_obfuscator_perf/sv_privacy/util_sv_tec_{tech}.csv'
    priv_emo_perf_path = f'./data/nn_obfuscator_perf/sv_privacy/priv_emo_tec_{tech}.csv'
    priv_gen_perf_path = f'./data/nn_obfuscator_perf/sv_privacy/priv_gen_tec_{tech}.csv'

    for index, top_k_size in tqdm(enumerate(top_k_sizes)):
        global current_top_k
        current_top_k = top_k_size

        if not Path(util_sv_perf_path).exists() or not Path(priv_emo_perf_path).exists():
            # ------------ Util/Priv Definitions ----------------
            util_sv_perf, priv_e_perf, priv_g_perf, model_features, top_k = train_obfuscator_top_k_features(shap_data_dict,
                                                                                                            top_k_size,
                                                                                                            x_train_scaled,
                                                                                                            x_test_scaled,
                                                                                                            util_labels,
                                                                                                            priv_e_labels,
                                                                                                            priv_g_labels)
            if len(util_sv_perf) == 0:
                print("All features have been removed, stopping the experiment.")
                continue
            elif util_sv_perf[0] == 9999:
                print("Private features already evaluated k={} skipping to the next.".format(top_k_size))
                continue

            # Collecting F1 scores
            util_sv_perf_list.append(util_sv_perf[1][0])
            priv_e_perf_list.append(priv_e_perf[1][0])
            priv_g_perf_list.append(priv_g_perf[1][0])
            model_features_list.append(str(model_features))
            top_k_list.append(top_k)

        else:
            print("Experiments with this configuration have been performed.")

    import pandas as pd
    data_df = pd.concat([pd.Series(util_sv_perf_list), pd.Series(priv_e_perf_list), pd.Series(priv_g_perf_list), pd.Series(model_features_list), pd.Series(top_k_list)], axis=1)
    data_df.columns = ['SV F1', 'Emotion F1', 'Gender F1', 'Features', 'Top K']
    data_df.to_csv(util_sv_perf_path)


def train_obfuscator_top_k_features(shap_data_dict, topk_size, x_train, x_test, util_labels, priv_e_labels,
                                    priv_g_labels):

    gen_shap_idxs = shap_data_dict["gen"]
    emo_shap_idxs = shap_data_dict["emo"]
    util_shap_idxs = shap_data_dict["sv"]

    if topk_size > 0:
        priv1_feature_mask = gen_shap_idxs[-topk_size:]
        priv2_feature_mask = emo_shap_idxs[-topk_size:]
        util_feature_mask = util_shap_idxs[-5:]
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
    model_features = np.setdiff1d(priv_features, util_feature_mask)
    # model_features = priv_features

    model_features.flags.writeable = False
    features_map_hash = hash(model_features.data.tobytes())
    # Addressing when all features are removed or its a repeated combination
    # Using empty lists or lists with 999 different error types. TODO add proper exception handling.
    if len(model_features) == 0:
        return [], [], [], [], []
    elif features_map_hash in feature_map_hash:
        return [9999], [9999], [9999], [9999], [9999]
    else:
        feature_map_hash[hash(model_features.data.tobytes())] = None

    # Removing priv features from input
    x_train[:, priv_features] = 0
    x_test[:, priv_features] = 0

    # save_model_meta_data(model_features, topk_size)

    # if not Path(obf_model_path).exists():
    util_sv_perf_list, priv_e_perf_list, priv_g_perf_list = evaluate_technique(x_train,
                                                                               x_test,
                                                                               util_labels,
                                                                               priv_e_labels,
                                                                               priv_g_labels)

    return util_sv_perf_list, priv_e_perf_list, priv_g_perf_list, priv_features, topk_size


def evaluate_technique(x_train, x_test_input, util_sv_labels, priv_e_labels, priv_g_labels):

    util_sv_perf_list = [[], []]
    priv_e_perf_list = [[], []]
    priv_g_perf_list = [[], []]

    collect_perf_metrics(util_sv_model, x_test_input, util_sv_labels[1], util_sv_perf_list)
    collect_perf_metrics(priv_emo_model, x_test_input, priv_e_labels[1], priv_e_perf_list)
    collect_perf_metrics(priv_gen_model, x_test_input, priv_g_labels[1], priv_g_perf_list)

    return util_sv_perf_list, priv_e_perf_list, priv_g_perf_list


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


def set_file_name(lambd, top_k_size, util_perf_path):
    util_perf_path_full = util_perf_path.format(str(priv_classes), str(lambd), top_k_size, epochs)
    return util_perf_path_full


if __name__ == "__main__":
    emo_model_path = "emo_checkpoint/emodel_scalarized_ravdess.h5"
    id_model_path = "sv_model_checkpoint/sver_model_scalarized_data.h5"
    gender_model_path = "gmodel_checkpoint/gmodel_scaled_ravdess.h5"

    obf_model_lite_path = 'obf_checkpoint/lite/model_obf_k_{}_util5'
    obf_model_keras_path = 'obf_checkpoint/model_obf_k_{}_util5'
    obf_model_meta_data = 'obf_checkpoint/model_obf_meta_k_removal_{}_util5'

    tech = "removal_top5_util"

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
    top_k_sizes = [x for x in range(1, 40, 1)]
    current_top_k = 0
    # top_k_sizes = [5, 10]

    batch_size = 32
    epochs = 300
    sample_results_rate = 10

    # metrics
    plot_epoch_rate = 50

    feature_map_hash = {}

    main()
