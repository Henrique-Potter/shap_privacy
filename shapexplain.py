import time

import numpy as np
import tensorflow as tf
from pathlib import Path

from tqdm import tqdm
from tensorflow.keras.models import load_model
from data_processing import pre_process_data, to_batchdataset
from experiment_neural_nets import get_obfuscation_model_swish, get_obfuscation_model_tanh2, get_obfuscation_model, \
    get_obfuscation_model_selu, get_obfuscation_model_gelu

from shap_experiment import extract_shap_values, parse_shap_values_by_class, evaluate_by_class, export_shap_to_csv
from sklearn.preprocessing import StandardScaler

from util.custom_functions import plot_obf_loss, calc_confusion_matrix, \
    priv_plot_perf_data_by_class, mean_std_analysis, priv_util_plot_acc_data, priv_util_plot_f1_data
from util.training_engine import train_step


def main():

    # get dataset
    print("Pre-processing audio files!")
    db_name = 'ravdess'

    x_train, x_test, y_tr_emo, y_te_emo, y_tr_gen, y_te_gen, y_tr_id, y_te_id = pre_process_data(audio_files_path, db_name)
    print("Pre-processing audio files Complete!")

    # Scaling and setting type to float32
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train).astype(np.float32)
    x_test_scaled = sc.transform(x_test).astype(np.float32)

    gender_nr_classes = len(y_tr_gen[0])
    emo_nr_classes = len(y_tr_emo[0])
    id_nr_classes = len(y_tr_id[0])

    print("Loading shap values")
    # When using ranked outputs, the shapeley values are also sorted by rank (e.g., index 0 always has the shapeley of the model prediction)
    gen_shap_values = extract_shap_values(gen_shap_df_path, priv_gen_model, x_test_scaled, x_train_scaled, gender_nr_classes)
    emo_shap_values = extract_shap_values(emo_shap_df_path, priv_emo_model, x_test_scaled, x_train_scaled, emo_nr_classes)
    id_shap_values = extract_shap_values(id_shap_df_path, util_sv_model, x_test_scaled, x_train_scaled, id_nr_classes)

    # Isolating shap values by class.
    gen_gt_shap_list, gen_corr_shap_list = parse_shap_values_by_class(gen_shap_values, y_te_gen)
    emo_gt_shap_list, emo_corr_shap_list = parse_shap_values_by_class(emo_shap_values, y_te_emo)
    id_gt_shap_list, id_corr_shap_list = parse_shap_values_by_class(id_shap_values, y_te_id)

    # model_name = 'gender_model_cr'
    # export_shap_to_csv(gen_corr_shap_list, model_name)
    #
    # model_name = 'emo_model_cr'
    # export_shap_to_csv(emo_corr_shap_list, model_name)

    # mean_std_analysis(gen_corr_shap_list)
    # mean_std_analysis(emo_corr_shap_list)
    #
    # pclass_shap_list0 = gen_gt_shap_list[0]
    # pclass_shap_list1 = gen_gt_shap_list[1]

    shap_imp_order_emotion = summarize_shap_scores(emo_gt_shap_list)
    shap_imp_order_gen = summarize_shap_scores(gen_gt_shap_list)
    shap_imp_order_id = summarize_shap_scores(id_gt_shap_list)

    shap_data_dict = {"emo": shap_imp_order_emotion, "gen": shap_imp_order_gen, "id": shap_imp_order_id}

    # ------------ Util/Priv Definitions ----------------
    priv_g_labels = (y_tr_gen, y_te_gen)
    priv_e_labels = (y_tr_emo, y_te_emo)
    util_labels = (y_tr_id, y_te_id)

    for lambd in lambds:

        # ------------ Util/Priv perf paths ----------------
        util_sv_perf_path = './data/nn_obfuscator_perf/sv_privacy/util_sv_data_cls{}_l{}_{}fts_e{}.npy'
        priv_emo_perf_path = './data/nn_obfuscator_perf/sv_privacy/priv_emo_data_cls{}_l{}_{}fts_e{}.npy'
        priv_gen_perf_path = './data/nn_obfuscator_perf/sv_privacy/priv_gen_data_cls{}_l{}_{}fts_e{}.npy'

        for top_k_size in top_k_sizes:
            # top_k_experiment = -top_k_experiment
            global current_top_k
            current_top_k = top_k_size

            util_perf_path_full = set_file_name(lambd, top_k_size, util_sv_perf_path)
            priv_emo_perf_path_full = set_file_name(lambd, top_k_size, priv_emo_perf_path)
            priv_gen_perf_path_full = set_file_name(lambd, top_k_size, priv_gen_perf_path)

            if Path(util_perf_path_full).exists() or not Path(priv_emo_perf_path_full).exists():
                # ------------ Util/Priv Definitions ----------------
                util_sv_perf_list, priv_e_perf_list, priv_g_perf_list = train_obfuscator_top_k_features(shap_data_dict,
                                                                                                        top_k_size,
                                                                                                        x_train_scaled,
                                                                                                        x_test_scaled,
                                                                                                        util_labels,
                                                                                                        priv_e_labels,
                                                                                                        priv_g_labels,
                                                                                                        lambd)

                np.save(util_perf_path_full, util_sv_perf_list,)
                np.save(priv_emo_perf_path_full, priv_e_perf_list,)
                np.save(priv_gen_perf_path_full, priv_g_perf_list,)
            else:
                print("Experiments with this configuration was already performed.")


def train_obfuscator_top_k_features(shap_data_dict, topk_size, x_train, x_test, util_labels, priv_e_labels, priv_g_labels, lambd):

    gen_shap_idxs = shap_data_dict["gen"]
    emo_shap_idxs = shap_data_dict["emo"]
    util_shap_idxs = shap_data_dict["id"]

    if topk_size > 0:
        priv1_feature_mask = gen_shap_idxs[-topk_size:]
        priv2_feature_mask = emo_shap_idxs[-topk_size:]
    elif topk_size < 0:
        priv1_feature_mask = gen_shap_idxs[:-topk_size]
        priv2_feature_mask = emo_shap_idxs[:-topk_size]
    else:
        priv1_feature_mask = []
        priv2_feature_mask = []

    # Guarantees some location information since its sorted and diff will only remove
    features = [x for x in range(40)]
    priv_features = np.union1d(priv1_feature_mask, priv2_feature_mask)
    model_features = np.setdiff1d(features, priv_features)

    # masked_x_train = np.zeros(x_train.shape)
    # masked_x_test = np.zeros(x_test.shape)

    masked_x_train = x_train[:, model_features]
    masked_x_test = x_test[:, model_features]

    # masked_x_test, masked_x_train = GetTopKFeatures(shap_data_dict, topk_size, x_test, x_train)

    masked_input = (masked_x_train, masked_x_test)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    # optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.01)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0, decay=6e-8)

    nn_input_sz = masked_x_train.shape
    # model = get_obfuscation_model_swish(nn_input_sz[1])
    model = get_obfuscation_model_swish(nn_input_sz[1])

    save_model_meta_data(model_features, topk_size)

    # if not Path(obf_model_path).exists():
    obf_model, util_sv_perf_list, priv_e_perf_list, priv_g_perf_list = train_obfuscation_model(model,
                                                                                               x_train,
                                                                                               x_test,
                                                                                               masked_input,
                                                                                               util_labels,
                                                                                               priv_e_labels,
                                                                                               priv_g_labels,
                                                                                               optimizer,
                                                                                               lambd)
    # else:
    #     model = tf.keras.models.load_model(obf_model_path)

    return util_sv_perf_list, priv_e_perf_list, priv_g_perf_list


def train_obfuscation_model(obf_model, x_train_input, x_test_input, masked_input, util_sv_labels, priv_e_labels,
                            priv_g_labels, optimizer, lambd):

    nr_e_classes = priv_e_labels[0][0].shape[0]
    nr_g_classes = priv_g_labels[0][0].shape[0]
    nr_sv_classes = util_sv_labels[0][0].shape[0]

    # Creating labels for private models as (batch size X nr features).
    # Target values should match no better than random guess.
    priv_e_label = tf.constant(np.tile(np.ones(nr_e_classes) * 1 / nr_e_classes, (batch_size, 1)))
    priv_g_label = tf.constant(np.tile(np.ones(nr_g_classes) * 1 / nr_g_classes, (batch_size, 1)))

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

    with tf.device('gpu:0'):
        for e in tqdm(range(epochs)):
            for (x_tr_batch, util_tr_y_batch), masked_x_tr_batch in zip(util_xy_tr_batch_dt, masked_x_tr_batch_dt):

                tloss, peloss, pgloss, uloss, gradients, _ = train_step(obf_model,
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
                                                                        priv_g_loss_fn,
                                                                        lambd)

                optimizer.apply_gradients(zip(gradients, obf_model.trainable_variables))
                total_tr_loss(tloss)
                priv_e_loss(peloss)
                priv_g_loss(pgloss)
                priv_sv_loss(uloss)

            final_loss_perf[0].append(total_tr_loss.result().numpy())
            final_loss_perf[1].append(priv_e_loss.result().numpy())
            final_loss_perf[2].append(priv_g_loss.result().numpy())
            final_loss_perf[3].append(priv_sv_loss.result().numpy())

            tf.print(total_tr_loss.result())
            tf.print(priv_e_loss.result())
            tf.print(priv_g_loss.result())
            tf.print(priv_sv_loss.result())

            total_tr_loss.reset_states()
            priv_e_loss.reset_states()
            priv_g_loss.reset_states()
            priv_sv_loss.reset_states()

            if (e+1) % sample_results_rate == 0:

                obf_input = obfuscate_input(obf_model, masked_x_te_batchdt, x_test_input)

                sv_y_pred = collect_perf_metrics(util_sv_model, obf_input, util_sv_labels, util_sv_perf_list)
                emo_y_pred = collect_perf_metrics(priv_emo_model, obf_input, priv_e_labels, priv_e_perf_list)
                gen_y_pred = collect_perf_metrics(priv_gen_model, obf_input, priv_g_labels, priv_g_perf_list)

                # util_perf = util_sv_model.evaluate(obf_input, util_sv_labels[1], verbose=0)
                # priv_e_perf = priv_emo_model.evaluate(obf_input, priv_e_labels[1], verbose=0)
                # priv_g_perf = priv_gen_model.evaluate(obf_input, priv_g_labels[1], verbose=0)

                # --- By class evaluation ---
                # evaluate_by_class(priv_emo_model, obf_input, priv_e_labels[1], e_by_class_perf)
                # evaluate_by_class(priv_gen_model, obf_input, priv_g_labels[1], g_by_class_perf)
                # evaluate_by_class(util_sv_model, obf_input, util_sv_labels[1], sv_by_class_perf)

                # util_sv_list.append(util_perf[1])
                # priv_e_perf_list.append(priv_e_perf[1])
                # priv_g_perf_list.append(priv_g_perf[1])
                #
                # final_loss_perf[4].append(util_perf[0])
                # final_loss_perf[5].append(priv_e_perf[0])
                # final_loss_perf[6].append(priv_g_perf[0])

            # Plotting results.
                if (e+1) % plot_epoch_rate == 0:
                    priv_util_plot_acc_data(priv_e_perf_list[0], priv_g_perf_list[0], util_sv_perf_list[0], "NN Obfuscator ACC Performance")
                    priv_util_plot_f1_data(priv_e_perf_list[1], priv_g_perf_list[1], util_sv_perf_list[1], "NN Obfuscator F1 Performance")

                    plot_obf_loss(final_loss_perf)

                if (e+1) % plot_epoch_rate == 0:

                    calc_confusion_matrix(sv_y_pred, util_sv_labels[1], current_top_k)
                    calc_confusion_matrix(gen_y_pred, priv_g_labels[1], current_top_k)
                    calc_confusion_matrix(emo_y_pred, priv_e_labels[1], current_top_k)

                    obf_model.save(obf_model_path.format(current_top_k))

    obf_model.reset_states()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    return obf_model, util_sv_perf_list, priv_e_perf_list, priv_g_perf_list


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


def collect_perf_metrics(model, obf_input, util_sv_labels, perf_list):
    from sklearn.metrics import classification_report

    y_predictions = model.predict(obf_input)
    y_labels_pred = np.argmax(y_predictions, axis=1)
    y_labels_true = np.argmax(util_sv_labels[1], axis=1)
    class_report = classification_report(y_labels_true, y_labels_pred, zero_division=0, output_dict=True)
    perf_list[0].append(class_report['accuracy'])
    perf_list[1].append(class_report['macro avg']['f1-score'])

    return y_predictions


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


def obfuscate_input(model, obfuscator_x_input, clean_x_innput):

    nr_features = clean_x_innput[0].shape[0]

    # Generating the mask
    start = time.time()
    obf_masks = model.predict(obfuscator_x_input)
    end = time.time()

    print("Time to create mask with {} features: {}".format(nr_features, end-start))

    mask_size = obf_masks[0].shape[0]
    padded_masks = np.pad(obf_masks, [(0, 0), (0, nr_features - mask_size)], mode='constant', constant_values=0)
    # Adding the mask to the input
    obf_input = clean_x_innput + padded_masks
    return obf_input


def set_file_name(lambd, top_k_size, util_perf_path):
    util_perf_path_full = util_perf_path.format(str(priv_classes), str(lambd), top_k_size, epochs)
    return util_perf_path_full


def GetTopKFeatures(shap_data_dict, topk_size, x_test, x_train):
    # Get the Top K if positive
    p_shap_idxs = shap_data_dict["gen"]
    if topk_size > 0:
        priv_feature_mask = p_shap_idxs[-topk_size:]
    else:
        priv_feature_mask = p_shap_idxs[:-topk_size]
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

    obf_model_path = 'obf_checkpoint/model_obf_k_{}.h5'
    obf_model_meta_data = 'obf_checkpoint/model_obf_meta_k_{}.h5'

    # datasets
    audio_files_path = "./NNDatasets/audio"
    gen_shap_df_path = './data/shapeley/gen_shap_df.npy'
    emo_shap_df_path = './data/shapeley/emo_shap_df.npy'
    id_shap_df_path = './data/shapeley/id_shap_df.npy'

    print("Loading trained Neural Nets")

    util_sv_model = load_model(id_model_path)
    priv_emo_model = load_model(emo_model_path)
    priv_gen_model = load_model(gender_model_path)

    # ------------- Hyperparameters -------------
    lambds = [.95]
    # lambds = [x/10 for x in range(1, 10)]

    priv_classes = []
    top_k_sizes = [-x for x in range(0, 41, 1)]
    current_top_k = 0
    # top_k_sizes = [0]

    batch_size = 32
    epochs = 300
    sample_results_rate = 10

    #metrics
    plot_epoch_rate = 300

    main()
