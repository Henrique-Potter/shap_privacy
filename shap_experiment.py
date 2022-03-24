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
from data_processing import pre_process_data
from experiment_config import set_experiment_config
from obfuscation_functions import general_by_class_mask
from util.custom_functions import replace_outliers_by_std, replace_outliers_by_quartile, mean_std_analysis
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def main():
    emo_model_path = './emo_checkpoint/emodel_m2_all_aug_5k_16.h5'
    gender_model_path = './gmodel_checkpoint/gmodel_m2_all_aug_5k_16.h5'

    audio_files_path = "./NNDatasets/audio"
    gen_shap_df_path = './data/gen_shap_df.npy'
    emo_shap_df_path = './data/emo_shap_df.npy'

    print("Pre-processing audio files!")
    x_train_emo_cnn, y_train_emo_encoded, x_test_emo_cnn, y_test_emo_encoded = pre_process_data(audio_files_path, get_emotion_label=True)
    x_train_gen_cnn, y_train_gen_encoded, x_test_gen_cnn, y_test_gen_encoded = pre_process_data(audio_files_path, get_emotion_label=False)
    print("Pre-processing audio files Complete!")

    # Sanity check. These summations should be 0.
    train_equal_sum = np.sum(x_train_emo_cnn != x_train_gen_cnn)
    test_equal_sum = np.sum(x_test_emo_cnn != x_test_gen_cnn)

    if train_equal_sum and test_equal_sum:
        print("Train and test are NOT the same")

    print("Loading trained Neural Nets")
    gender_model = load_model(gender_model_path)
    emo_model = load_model(emo_model_path)

    gender_nr_classes = len(y_train_gen_encoded[0])
    emo_nr_classes = len(y_train_emo_encoded[0])
    print("Loading shap values")

    # When using ranked outputs, the shapeley values are also sorted by rank (e.g., index 0 always has the shapeley of the model prediction)
    gen_shap_values = extract_shap_values(gen_shap_df_path, gender_model, x_test_emo_cnn, x_train_emo_cnn, gender_nr_classes)
    emo_shap_values = extract_shap_values(emo_shap_df_path, emo_model, x_test_emo_cnn, x_train_emo_cnn, emo_nr_classes)

    # Isolating shap values by class.
    gen_gt_shap_list, gen_corr_shap_list = parse_shap_values_by_class(gen_shap_values, y_test_gen_encoded)
    emo_gt_shap_list, emo_corr_shap_list = parse_shap_values_by_class(emo_shap_values, y_test_emo_encoded)

    # Exporting SHAP to excel
    model_name = 'gender_model_gt'
    export_shap_to_csv(gen_gt_shap_list, model_name)
    model_name = 'gender_model_cr'
    export_shap_to_csv(gen_corr_shap_list, model_name)

    model_name = 'emo_mo del_gt'
    export_shap_to_csv(emo_gt_shap_list, model_name)
    model_name = 'emo_model_cr'
    export_shap_to_csv(emo_corr_shap_list, model_name)
    #
    # # ------------------------ Analyzing Shap values ------------------------
    mean_std_analysis(gen_gt_shap_list)
    mean_std_analysis(gen_corr_shap_list)
    mean_std_analysis(emo_gt_shap_list)
    mean_std_analysis(emo_corr_shap_list)

    # mean_std_analysis(gen_corr_shap_list)
    # mean_std_analysis(emo_corr_shap_list)

    # shap_np_scaled_sorted, shap_sorted_scaled_avg, shap_sorted_indexes = analyse_shap_values(m_shap_list)
    # shap_np_scaled_sorted, shap_sorted_scaled_avg, shap_sorted_indexes = analyse_shap_values(f_shap_list)

    # Building obfuscation experiment data
    model_list, obfuscation_f_list = set_experiment_config(emo_model,
                                                           gender_model,
                                                           emo_gt_shap_list,
                                                           gen_gt_shap_list,
                                                           y_test_emo_encoded,
                                                           y_test_gen_encoded)

    # # Sanity check model performance check
    # evaluate_model(model_list, x_test_emo_cnn)
    # # Sanity check model performance check
    # evaluate_model(model_list, x_test_gen_cnn)

    # general_mask_evaluation(model_list, x_test_gen_cnn)

    # Evaluating obfuscation functions
    perf_list = evaluate_obfuscation_function(model_list,
                                              obfuscation_f_list,
                                              x_test_gen_cnn)

    # Plotting results
    plot_obs_f_performance(perf_list)


def export_shap_to_csv(gen_ground_truth_list, model_name):
    class_id = 0
    for class_data in gen_ground_truth_list:
        numpy_matrix_df = pd.DataFrame(class_data)
        numpy_matrix_df.to_excel("./data/csv/{}_class_{}_shap_values.xlsx".format(model_name, class_id))
        class_id += 1


def extract_shap_values(shap_df_path, model, x_target_data, x_background_data, nr_classes):
    # Extracting SHAP values
    if not Path(shap_df_path).exists():
        print("Calculating Shap values")
        # Generating Shap Values
        shap_vals, e = extract_shap(model, x_target_data, x_background_data, 1000, nr_classes)
        np.save(shap_df_path, shap_vals, allow_pickle=True)
    else:
        shap_vals = np.load(shap_df_path, allow_pickle=True)

    return shap_vals


def evaluate_obfuscation_function(model_list, obf_f_list, x_model_input):
    model_perf_list = []

    priv_target_mdl = None
    util_target_mdl = None

    for model_dict in model_list:
        if model_dict['privacy_target']:
            priv_target_mdl = model_dict
        if model_dict['utility_target']:
            util_target_mdl = model_dict

    for model_dict in model_list:
        model_name = model_dict['model_name']

        obf_f_perf_list = []
        for obf_f_dict in obf_f_list:

            obf_f_str_list = obf_f_dict['intensities']

            # Function Name
            obf_f_name = obf_f_dict['label']

            # Return list of noise str parameters
            obfuscated_model_perf_loss = []
            obfuscated_model_perf_acc = []
            obfuscated_model_by_cls_perf = []

            metrics_perf_list = []

            for obf_intensity in tqdm(obf_f_str_list):

                overall_perf, by_class_perf = obfuscate_and_evaluate(model_dict,
                                                                     obf_f_dict,
                                                                     obf_intensity,
                                                                     priv_target_mdl,
                                                                     util_target_mdl,
                                                                     x_model_input)

                # --- Collecting Metrics ----
                obfuscated_model_perf_loss.append(overall_perf[0])
                obfuscated_model_perf_acc.append(overall_perf[1])
                obfuscated_model_by_cls_perf.append(by_class_perf)

            metrics_perf_list.append(("loss", obfuscated_model_perf_loss))
            metrics_perf_list.append(("acc", obfuscated_model_perf_acc))
            metrics_perf_list.append(("by_class", obfuscated_model_by_cls_perf))

            obf_f_perf_list.append((obf_f_name, metrics_perf_list))

        model_perf_list.append((model_dict, obf_f_perf_list))

        tf.keras.backend.clear_session()

    return model_perf_list


def obfuscate_and_evaluate(model_dict,
                           obf_f_dict,
                           obf_intensity,
                           priv_target_mdl,
                           util_target_mdl,
                           x_model_input):

    model = model_dict['model']
    curr_y_labels = model_dict['ground_truth']
    curr_model_name = model_dict['model_name']

    obf_f = obf_f_dict['obf_f_handler']
    kwargs = obf_f_dict['kwargs']

    model_perf_list = []
    per_class_model_perf_list = []
    obf_function_name = obf_f_dict['label']
    avg_reps = kwargs['avg_reps']

    priv_shap_data = priv_target_mdl['shap_values']
    util_shap_data = util_target_mdl['shap_values']

    for i in range(avg_reps):

        # local_priv_shap_data = copy_numpy_matrix_list(priv_shap_data)
        # local_util_shap_data = copy_numpy_matrix_list(util_shap_data)

        local_x_data = x_model_input.copy()
        local_y_data = curr_y_labels.copy()

        # Applying Obfuscation Function
        # y_match returns x->y labels in case x was changed
        obfuscated_x, y_match = obf_f(local_x_data,
                                      priv_target_mdl,
                                      util_target_mdl,
                                      local_y_data,
                                      obf_intensity,
                                      **kwargs)

        # Evaluating models performance with obfuscated data
        obfuscated_perf = model.evaluate(obfuscated_x, y_match, verbose=0)
        # --- By class evaluation ---
        by_class_perf = evaluate_by_class(model, obfuscated_x, y_match)

        model_perf_list.append(obfuscated_perf)
        per_class_model_perf_list.append(by_class_perf)

    overall_perf_avg = avg_model_perf(model_perf_list, avg_reps)
    by_class_perf_avg = avg_by_class_perf(per_class_model_perf_list, avg_reps)

    status_text = "Overall performance for model {} with obfuscation function {} and noise level at {}:"
    print("\n")
    print(status_text.format(curr_model_name, obf_function_name, obf_intensity))
    print(overall_perf_avg)
    print("\n")

    return overall_perf_avg, by_class_perf_avg


def avg_by_class_perf(per_class_model_perf_list, perf_reps_nr):

    nr_classes = len(per_class_model_perf_list[0])
    final_by_class_perf = []
    for class_index in range(nr_classes):

        total_loss = 0
        total_acc = 0
        for by_class_item in per_class_model_perf_list:
            total_loss += by_class_item[class_index][0]
            total_acc += by_class_item[class_index][1]

        final_by_class_perf.append((class_index, [total_loss / perf_reps_nr, total_acc / perf_reps_nr]))
    return final_by_class_perf


def avg_model_perf(model_perf_list, perf_range):
    total_loss = 0
    total_acc = 0
    for perf in model_perf_list:
        ploss = perf[0]
        pAcc = perf[1]
        total_loss += ploss
        total_acc += pAcc
    final_model_perf = [total_loss / perf_range, total_acc / perf_range]
    return final_model_perf


def copy_numpy_matrix_list(shap_data):
    local_list = []
    for shap_class_data in shap_data:
        local_shap_class = shap_class_data.copy()
        local_list.append(local_shap_class)
    return local_list


def parse_per_class_perf_data(by_class_perf_data):
    perf_data_by_class = []
    nr_obf_intsties = len(by_class_perf_data)
    nr_classes = len(by_class_perf_data[0])

    # Restructuring data from by intensity->classes to class->intensities
    for class_index in range(nr_classes):
        class_perf_acc = []
        class_perf_loss = []
        for intsty_index in range(nr_obf_intsties):
            # Getting both acc and loss
            class_perf_loss.append(by_class_perf_data[intsty_index][class_index][1][0])
            class_perf_acc.append(by_class_perf_data[intsty_index][class_index][1][1])

        perf_data_by_class.append((class_index, class_perf_loss, class_perf_acc))

    return perf_data_by_class


def evaluate_model(model_list, x_test):
    for model_dict in model_list:
        model_name = model_dict['model_name']
        model = model_dict['model']
        y_model_input = model_dict['ground_truth']
        test_perf = model.evaluate(x_test, y_model_input)
        print("{} Model Test perf is:{}".format(model_name, test_perf))


def general_mask_evaluation(model_list, x_test):

    modulation_levels = [x/10 for x in range(7, 13, 1)]

    p_model_id = 0
    p_class_id = 5
    u_model_id = 0
    u_class_id = 1
    priv_topk_size = 10
    util_topk_size = 10

    priv_feature_mask, util_feature_mask, features_removed, origi_pmask = general_by_class_mask(p_model_id,
                                                                                                u_model_id,
                                                                                                p_class_id,
                                                                                                u_class_id,
                                                                                                model_list,
                                                                                                priv_topk_size,
                                                                                                util_topk_size)

    models_perf = [[] for _ in model_list]
    by_class_models_perf = [[] for _ in model_list]
    print("------------------ Simple evaluation Start------------------")
    for mdl_lvl in modulation_levels:

        mask_array = np.ones(40, dtype=float)
        mask_array[priv_feature_mask] = mdl_lvl
        x_input = x_test.copy()
        # mask_array = mask_array.astype(bool)
        # Setting only the non top k to 0. Creating a Top k shap where all other values are 0.
        x_input[:, :, 0] = x_input[:, :, 0] * mask_array
        index = 0

        for model_list_idx in range(len(model_list)):
            model_name = model_list[model_list_idx]['model_name']
            model = model_list[model_list_idx]['model']
            y_model_input = model_list[model_list_idx]['ground_truth']

            test_perf = model.evaluate(x_input, y_model_input, verbose=0)
            by_class_perf = evaluate_by_class(model, x_input, y_model_input)

            models_perf[model_list_idx].append(test_perf)
            by_class_models_perf[model_list_idx].append(by_class_perf)

            print("{} Model Test perf is:{}".format(model_name, test_perf))

    fig = plt.figure()
    fig.set_size_inches(17, 10)
    fig.set_dpi(100)

    eacc = np.vstack(models_perf[0])[:, 1]
    gacc = np.vstack(models_perf[1])[:, 1]

    x_list = [x for x in range(len(modulation_levels))]

    plt.plot(x_list, eacc, label='Emo model ACC')
    plt.plot(x_list, gacc, label='Gen model ACC')
    tlt = "(Overall View) General mask for private model {} with class {}. The Top {}: {} Masked: {}\n".format(p_model_id, p_class_id, priv_topk_size, origi_pmask,priv_feature_mask)
    tlt2 = "Utility top {} are {}. Priv-Util Match {}".format(util_topk_size, util_feature_mask, features_removed)
    plt.title(tlt+tlt2)
    plt.ylabel('ACC')
    plt.xlabel('Modulation multiplier')
    plt.legend()
    plt.xticks(x_list, modulation_levels)
    plt.show()

    fig = plt.figure()
    fig.set_size_inches(17, 10)
    fig.set_dpi(100)
    ax1 = fig.add_subplot(111)
    number_of_plots = 9
    colormap = plt.cm.nipy_spectral
    colors = [colormap(i) for i in np.linspace(0, 1, number_of_plots)]
    ax1.set_prop_cycle('color', colors)

    for model_idx in range(len(by_class_models_perf)):
        class_perf_list = parse_perf_class(by_class_models_perf[model_idx])

        for class_data_idx in range(len(class_perf_list)):
            if class_data_idx == p_class_id and model_idx == p_model_id:
                clr = 'r'
                lbl = '{} ACC'.format('(Priv) Model {} class {}'.format(model_idx, class_data_idx))
                ax1.plot(x_list, class_perf_list[class_data_idx], color=clr, label=lbl)
            elif class_data_idx == u_class_id and model_idx == u_model_id:
                clr = 'b'
                lbl = '{} ACC'.format('(Util) Model {} class {}'.format(model_idx, class_data_idx))
                ax1.plot(x_list, class_perf_list[class_data_idx], color=clr, label=lbl)
            else:
                lbl = '{} ACC'.format('Model {} class {}'.format(model_idx, class_data_idx))
                plt.plot(x_list, class_perf_list[class_data_idx], label=lbl)

    tlt3 = "(By class View) General mask for private model {} with class {}. The Top {}: {} Masked: {}\n".format(p_model_id, p_class_id, priv_topk_size, origi_pmask, priv_feature_mask)
    tlt4 = "Utility model {} class {}. Top {}: {}. Priv-Util Match: {}".format(u_model_id, u_class_id, util_topk_size, util_feature_mask, features_removed)
    ax1.set_title(tlt3+tlt4)
    ax1.set_ylabel('ACC')
    ax1.set_xlabel('Modulation multiplier')
    ax1.set_xticks(x_list, modulation_levels)
    plt.legend()
    plt.show()

    print("------------------ Simple evaluation END ------------------")


def parse_perf_class(gen_model_by_class_perf):
    class_perf_list = [[] for x in range(len(gen_model_by_class_perf[0]))]
    for classes_perf in gen_model_by_class_perf:
        for class_perf_idx in range(len(classes_perf)):
            class_perf_list[class_perf_idx].append(classes_perf[class_perf_idx][1])
    return class_perf_list


# Plots performance data for N number of models with N number of obfuscation functions
def plot_obs_f_performance(perf_list):
    plt.clf()

    header = []
    collum_data = []

    # Iterating over models evals
    for model in perf_list:

        priv_class = None
        util_class = None
        if model[0]['privacy_target']:
            priv_target_mdl = model[0]
            priv_class = priv_target_mdl['priv_class']
        if model[0]['utility_target']:
            util_target_mdl = model[0]
            util_class = util_target_mdl['util_class']

        model_name = model[0]['model_name']
        obf_list = model[1]
        obf_f_index = 0
        # Iterating over obfuscation function's evals
        for obf_f in obf_list:
            obf_f_name = obf_f[0]
            obf_f_eval_metrics = obf_f[1]
            # Iterating over metrics
            for metric in obf_f_eval_metrics:
                metric_name = metric[0]
                metric_data = metric[1]

                title = "{} NN model {}".format(model_name, metric_name)
                lbl = "{} mdl w/ {}".format(model_name, obf_f_name)

                if metric_name == "loss":
                    continue
                    line_plot_metric_data(lbl, metric_data, obf_f_name, title)

                elif metric_name == "acc":
                    header.append(lbl)
                    first, half, last, one_quarter, three_quarters = get_data_samples(metric_data)
                    collum_data.append([first, one_quarter, half, three_quarters, last])
                    continue
                    line_plot_metric_data(lbl, metric_data, obf_f_name, title)

                elif metric_name == "by_class":
                    # Parsing by class data
                    parsed_perf_by_class = parse_per_class_perf_data(metric_data)
                    plot_obs_f_performance_by_class(model_name, obf_f_name, obf_f_index, parsed_perf_by_class, priv_class, util_class)
            obf_f_index += 1

    fig, ax = plt.subplots()
    fig.set_size_inches(17, 10)
    fig.set_dpi(150)
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    row_header = ['First Value', '25%', '50%', '75%', 'Last']
    models_acc_data = np.array(collum_data).transpose()
    tab = table(plt.gca(),
                cellText=models_acc_data,
                rowLabels=row_header,
                colLabels=header,
                loc='center',
                cellLoc='center')
    tab.auto_set_column_width(col=list(range(len(header))))
    tab.scale(1, 2)
    tab.set_fontsize(10)

    plt.title('Models ACC overview')
    plt.show()
    plt.clf()


# Plots performance data for N number of models with N number of obfuscation functions
def plot_obs_f_performance_by_class(model_name, obf_f_name, obf_f_index, parsed_perf_by_class, priv_class, util_class):
    title_loss = "NN models Loss"
    title_acc = "NN models Accuracy"
    if len(parsed_perf_by_class) == 0:
        return

    nr_intensities = len(parsed_perf_by_class[0][1])
    x_list = [x for x in range(0, nr_intensities)]

    fig, ax = plt.subplots()
    fig.set_size_inches(17, 10)
    fig.set_dpi(200)
    # gs = fig.add_gridspec(2)
    # ax1 = fig.add_subplot(gs[0])
    # ax2 = fig.add_subplot(gs[1])

    header = []
    row_data = []

    for class_nr, class_perf_loss, class_perf_acc in parsed_perf_by_class:
        if class_nr == priv_class:
            lbl = "Class {} (Private)".format(class_nr)
        elif class_nr == util_class:
            lbl = "Class {} (Utility)".format(class_nr)
        else:
            lbl = "Class {}".format(class_nr)

        ax.plot(x_list, class_perf_acc, label=lbl)
        header.append(lbl)
        first, half, last, one_quarter, three_quarters = get_data_samples(class_perf_acc)
        row_data.append([first, one_quarter, half, three_quarters, last])

    # ax1.set_title(title_loss)
    # ax1.set_ylabel('loss')
    # ax1.set_xlabel('Noise Intensity level')
    # ax1.legend()
    ax.set_ylim([0, 1])
    ax.set_title(title_acc)
    ax.set_ylabel('accuracy')
    ax.set_xlabel('Noise Intensity level')
    ax.legend()
    # plt.subplots_adjust(hspace=0.7)
    plt.title('Model Accuracy per obfuscation Intensity for {}'.format(obf_f_name))
    plt.show()
    plt.clf()

    fig, ax = plt.subplots()
    fig.set_size_inches(14, 10)
    fig.set_dpi(150)
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    row_header = ['First Value', '25%', '50%', '75%', 'Last']
    models_acc_data = np.array(row_data).transpose()
    tab = table(plt.gca(),
                cellText=models_acc_data,
                rowLabels=row_header,
                colLabels=header,
                loc='center',
                cellLoc='center')
    tab.auto_set_column_width(col=list(range(len(header))))
    tab.scale(1, 2)
    tab.set_fontsize(10)
    #
    # fig.tight_layout()
    plt.title('{} Accuracy Overview for {}'.format(model_name,obf_f_name))
    plt.show()
    plt.clf()


def get_data_samples(perf_data):
    first = round(perf_data[0], 2)
    perf_dt_sz = len(perf_data)
    half = round(perf_data[int(perf_dt_sz / 2)], 2)
    three_quarters = round(perf_data[int(3 * perf_dt_sz / 4)], 2)
    one_quarter = round(perf_data[int(perf_dt_sz / 4)], 2)
    last = round(perf_data[int(perf_dt_sz - 1)], 2)
    return first, half, last, one_quarter, three_quarters


def evaluate_by_class(model, obfuscated_x, y_model_input, by_class_perf):

    nr_classes = y_model_input.shape[1]

    for cls_index in range(nr_classes):
        single_class_map = y_model_input[:, cls_index] == 1

        obfuscated_x_single_class = obfuscated_x[single_class_map]
        y_model_input_single_class = y_model_input[single_class_map]
        obfuscated_single_class_perf = model.evaluate(obfuscated_x_single_class, y_model_input_single_class, verbose=0)
        by_class_perf[0][cls_index].append(obfuscated_single_class_perf[0])
        by_class_perf[1][cls_index].append(obfuscated_single_class_perf[1])


def simple_bar_plot(obfuscated_x, class_id):
    fig = plt.figure()
    fig.set_size_inches(17, 10)
    fig.set_dpi(100)
    x_list = [x for x in range(39)]

    random_idx = np.random.randint(0, obfuscated_x.shape[0])
    data_mean = np.mean(obfuscated_x[:, 1:, 0])

    plt.bar(x_list, data_mean, axis=0)
    plt.ylabel('MFCC')
    plt.xlabel('Features 1-39')
    plt.title('Class nr {}'.format(class_id))
    plt.legend()
    plt.show()


def parse_shap_values_by_class(shap_data, y_data):

    shap_values = shap_data[0]
    shap_predictions = shap_data[1]

    shap_predictions = shap_predictions[:, 0]
    y_data_int = np.argmax(y_data, axis=1)
    correct_shap_map = shap_predictions == y_data_int
    misses = np.sum(shap_predictions != y_data_int)

    #This will be equal to the number of classes
    gt_shap_list = [[] for x in range(len(shap_values))]
    correct_shap_list = [[] for x in range(len(shap_values))]

    for index in range(y_data.shape[0]):
        # Getting the true class number (equivalent to logit id)
        gt_class_nr = y_data_int[index]
        # Getting the index position of the ground truth shap values
        gt_class_shp_index = np.argwhere(shap_data[1][index] == gt_class_nr)[0][0]
        # Get the actual shapeley value at the gt_class_shp_index postion
        gt_shp_vals = shap_values[gt_class_shp_index][index]
        # Append in a list ordered by class number
        gt_shap_list[gt_class_nr].append(gt_shp_vals)

        # This maps the matches between model prediction and ground truth
        if correct_shap_map[index]:
            # The correct shap will always be at position 0 since its ranked by output probability.
            correct_shp_vals = shap_values[0][index]
            correct_shap_list[gt_class_nr].append(correct_shp_vals)

    processed_gt_matrix_list = clean_outliers(gt_shap_list)
    processed_cr_matrix_list = clean_outliers(correct_shap_list)

    return processed_gt_matrix_list, processed_cr_matrix_list


def clean_outliers(gt_shap_list):

    cleared_class_matrix_list = []
    for class_data in gt_shap_list:
        class_data_matrix = np.vstack(class_data)
        class_data_matrix = replace_outliers_by_quartile(class_data_matrix)
        cleared_class_matrix_list.append(class_data_matrix)

    return cleared_class_matrix_list


def add_noise(m_shap_sum, x_gen_train, y_gen_train, noise_str):
    male_only_x = None
    male_only_y = None
    import random as rd
    for sample_index in range(x_gen_train.shape[0]):
        true_label_index = np.where(y_gen_train[sample_index] == 1)[0][0]

        if true_label_index == 0:
            for feature_index in range(259):
                if m_shap_sum[feature_index] > 0:
                    rd_noise = np.random.randn(1, 1)[0, 0] * noise_str
                    current_value = x_gen_train[sample_index, feature_index, 0]
                    x_gen_train[sample_index, feature_index, 0] = current_value + rd_noise
        else:
            male_only_x = np.delete(x_gen_train, sample_index, 0)
            male_only_y = np.delete(y_gen_train, sample_index, 0)
    return male_only_x, male_only_y


def analyse_shap_values(m_shap_list):

    shap_np = np.array(m_shap_list)
    shap_np = replace_outliers_by_std(shap_np, 3)
    shap_np = whiten(shap_np)
    #shap_np_scaled = normalizeData_0_1(shap_np)
    shap_np_scaled = shap_np

    shap_nr_samples = shap_np.shape[0]
    shap_nr_features = shap_np.shape[1]
    x_list = [x for x in range(shap_nr_features)]

    shap_samples_sum = np.sum(shap_np_scaled, axis=0)
    # shap_samples_abssum = np.sum(np.abs(shap_np_scaled, axis=0))

    shap_sorted_indexes = np.argsort(shap_samples_sum)
    shap_samples_sum_sorted = shap_samples_sum[shap_sorted_indexes]
    shap_sorted_scaled_avg = shap_samples_sum_sorted/shap_nr_samples
    shap_np_scaled_sorted = shap_np_scaled[:, shap_sorted_indexes]

    for index in range(int(shap_nr_samples/10)):
        plt.plot(x_list, shap_np_scaled_sorted[index, :], c='b', alpha=0.2)
    plt.plot(x_list, shap_sorted_scaled_avg, c='r')
    # plt.xticks(x_list, shap_sorted_indexes)

    #plt.yscale('log')
    #plt.yticks((np.arange(-0.1, 0.1, step=0.001)))
    #plt.ylim((-0.1, 0.1))
    plt.show()

    plt.bar(x_list, shap_samples_sum_sorted)
    plt.show()

    return shap_np_scaled_sorted, shap_sorted_scaled_avg, shap_sorted_indexes


def analyse_timeseries_kmeans(m_shap_list):

    shap_np = np.array(m_shap_list)
    #shap_np = replace_outliers_by_std(shap_np, 2)
    #shap_np = whiten(shap_np)
    #shap_np_scaled = NormalizeData_0_1(shap_np)
    shap_np_scaled = shap_np

    shap_samples_sum = np.sum(shap_np_scaled, axis=0)
    shap_sorted_indexes = np.argsort(shap_samples_sum)

    shap_np_scaled = shap_np_scaled[:, shap_sorted_indexes]

    from tslearn.clustering import TimeSeriesKMeans
    n_clusters = 2
    model = TimeSeriesKMeans(n_clusters=n_clusters, n_jobs=4, metric="euclidean", max_iter_barycenter=5, max_iter=100)
    y_pred = model.fit_predict(shap_np_scaled)
    sz = shap_np_scaled.shape[1]

    plt.figure()
    for yi in range(n_clusters):
        plt.subplot(n_clusters, 1, yi + 1)
        for xx in shap_np_scaled[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.1)
        plt.plot(model.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.text(1.01, 0.85, 'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
        if yi == 0:
            plt.title("Euclidean $k$-means")

    plt.subplots_adjust(hspace=0.7)
    plt.show()

    return shap_np_scaled


def plot_scatter_first_vs_all(codebook, m_shap_np, m_shap_np_scaled_whitened):
    for index in range(m_shap_np.shape[1]):
        plt.scatter(m_shap_np_scaled_whitened[:, 0], m_shap_np_scaled_whitened[:, index], c='b')
    plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    plt.show()


def plot_scatter_first_feature(codebook, m_shap_np_scaled_whitened):
    plt.plot(m_shap_np_scaled_whitened[:, 0], m_shap_np_scaled_whitened[:, 1])
    plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    plt.show()


def normalizeData_0_1(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def normalizeData_minus1_1(data):
    return data / np.max(data)


def plot_shap_values(shap_values, shap_input_np):
    tilled_shap_list = []
    for shap_value in shap_values:
        temp = shap_value.reshape((shap_value.shape[0], 1, 259, 1))
        temp2 = np.tile(temp, (100, 1, 1))
        tilled_shap_list.append(temp2)

    # Forcing array as a image by tilling the array (# samples x width x height x channels),
    reshaped_input = shap_input_np.reshape((shap_input_np.shape[0], 1, 259, 1))
    tilling = np.tile(reshaped_input, (100, 1, 1))
    shap.image_plot(tilled_shap_list, tilling)


def extract_shap(model, shap_input, background_data, background_size, nr_classes):
    rank_order = 'max'
    if nr_classes is None:
        rank_order = None

    background = background_data[:background_size]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(shap_input, ranked_outputs=nr_classes, output_rank_order=rank_order, check_additivity=False)
    # shap_values_raw = e.shap_values(shap_input, check_additivity=False)

    return shap_values, e


if __name__ == '__main__':
    main()
