import time

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
from obfuscation_functions import *
from util.custom_functions import replace_outliers_by_std

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def main():
    emo_model_path = './emo_checkpoint/emodel_m2_all_aug_5k_16.h5'
    gender_model_path = './gmodel_checkpoint/gmodel_m2_all_aug_5k_16.h5'

    audio_files_path = "./NNDatasets/audio"
    gen_shap_df_path = './data/emo_shap_df.npy'

    print("Pre-processing audio files!")
    x_train_emo_cnn, y_train_emo_encoded, x_test_emo_cnn, y_test_emo_encoded = pre_process_data(audio_files_path, get_emotion_label=True)
    x_train_gen_cnn, y_train_gen_encoded, x_test_gen_cnn, y_test_gen_encoded = pre_process_data(audio_files_path, get_emotion_label=False)
    print("Pre-processing audio files Complete!")

    # Sanity check. These summations should be 0.
    train_equal_sum = np.sum(x_train_emo_cnn != x_train_gen_cnn)
    test_equal_sum = np.sum(x_test_emo_cnn != x_test_gen_cnn)

    print("Loading trained Neural Nets")
    emo_model = load_model(emo_model_path)
    gender_model = load_model(gender_model_path)

    # Extracting SHAP values
    if not Path(gen_shap_df_path).exists():
        print("Calculating Shap values")
        # Generating Shap Values
        gen_shap_values, e = extract_shap(gender_model, x_test_gen_cnn, x_train_gen_cnn, 150)
        np.save(gen_shap_df_path, gen_shap_values)
    else:
        gen_shap_values = np.load(gen_shap_df_path)

    # Isolating shap values by class.
    m_shap_list, f_shap_list = get_target_shap(gen_shap_values, x_test_gen_cnn, y_test_gen_encoded)

    # ------------------------ Analyzing Shap values ------------------------
    plot_shap_2(m_shap_list, f_shap_list)
    shap_np_scaled_sorted, shap_sorted_scaled_avg, shap_sorted_indexes = analyse_shap_values(m_shap_list)
    shap_np_scaled_sorted, shap_sorted_scaled_avg, shap_sorted_indexes = analyse_shap_values(f_shap_list)

    # Building obfuscation experiment data
    emo_model_dict = {'model_name': "emotion_model", 'model': emo_model, 'ground_truth': y_test_emo_encoded, 'privacy_target': False}
    gen_model_dict = {'model_name': "gen_model", 'model': gender_model, 'ground_truth': y_test_gen_encoded, 'privacy_target': True}

    model_list = [emo_model_dict, gen_model_dict]

    # Building Obfuscation list functions
    # Noise intensity List
    norm_noise_list = [x/10 for x in range(1, 500, 10)]
    obfuscation_f_list = []
    obf_by_male_gender = {'obf_f_handler': obfuscate_by_class, 'intensities': norm_noise_list, 'kwargs': {'class_index':0}, 'label':'obf_male'}
    obf_by_female_gender = {'obf_f_handler': obfuscate_by_class, 'intensities': norm_noise_list, 'kwargs': {'class_index':1}, 'label':'obf_female'}
    obfuscation_f_list.append(obf_by_male_gender)
    obfuscation_f_list.append(obf_by_female_gender)

    # Sanity check model performance check
    evaluate_model(model_list, x_test_emo_cnn)

    time.time()

    # Evaluating obfuscation functions
    perf_list = evaluate_obfuscation_function(gen_shap_values, model_list, obfuscation_f_list, x_test_gen_cnn)

    # Plotting results
    plot_obs_f_performance(perf_list)


def evaluate_obfuscation_function(shap_values, model_list, obf_f_list, x_model_input):
    model_perf_list = []

    target_mdl = None
    for model_dict in model_list:
        if model_dict['privacy_target']:
            target_mdl = model_dict

    for model_dict in model_list:

        model_name = model_dict['model_name']
        model = model_dict['model']
        y_model_input = model_dict['ground_truth']
        privacy_target = model_dict['privacy_target']

        obf_f_perf_list = []
        for obf_f_dict in obf_f_list:

            obf_f = obf_f_dict['obf_f_handler']
            obf_f_str_list = obf_f_dict['intensities']
            kwargs = obf_f_dict['kwargs']

            # Function Name
            obf_f_name = obf_f_dict['label']

            # Return list of noise str parameters
            obfuscated_model_perf_loss = []
            obfuscated_model_perf_acc = []
            obfuscated_model_by_cls_perf = []

            metrics_perf_list = []

            for obf_intensity in tqdm(obf_f_str_list):
                # Obfuscating only males index class is 0
                # Applying Obfuscation Function
                obfuscated_x = obf_f(shap_values, x_model_input, target_mdl['ground_truth'], obf_intensity, **kwargs)
                # --- Collecting Metrics ----
                obfuscated_perf = model.evaluate(obfuscated_x, y_model_input)
                obfuscated_model_perf_loss.append(obfuscated_perf[0])
                obfuscated_model_perf_acc.append(obfuscated_perf[1])
                # By class evaluation
                by_class_perf = evaluate_by_class(model, obfuscated_x, y_model_input)
                obfuscated_model_by_cls_perf.append(by_class_perf)

            metrics_perf_list.append(("loss", obfuscated_model_perf_loss))
            metrics_perf_list.append(("acc", obfuscated_model_perf_acc))
            metrics_perf_list.append(("by_class", obfuscated_model_by_cls_perf))

            obf_f_perf_list.append((obf_f_name, metrics_perf_list))

        model_perf_list.append((model_name, obf_f_perf_list))

        tf.keras.backend.clear_session()

    return model_perf_list


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


# Plots performance data for N number of models with N number of obfuscation functions
def plot_obs_f_performance(perf_list):
    plt.clf()

    header = []
    collum_data = []

    # Iterating over models evals
    for model in perf_list:
        model_name = model[0]
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
                    line_plot_metric_data(lbl, metric_data, obf_f_name, title)

                elif metric_name == "acc":
                    header.append(lbl)
                    first, half, last, one_quarter, three_quarters = get_data_samples(metric_data)
                    collum_data.append([first, one_quarter, half, three_quarters, last])

                    line_plot_metric_data(lbl, metric_data, obf_f_name, title)

                elif metric_name == "by_class":
                    # Parsing by class data
                    parsed_perf_by_class = parse_per_class_perf_data(metric_data)
                    plot_obs_f_performance_by_class(model_name, obf_f_name, obf_f_index, parsed_perf_by_class)
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
    #
    # fig.tight_layout()
    plt.title('Models ACC overview')
    plt.show()
    plt.clf()

    # # Plotting overview table
    # fig, ax = plt.subplots()
    # fig.set_dpi(100)
    # fig.set_size_inches(9, 10)
    # # hide axes
    # fig.patch.set_visible(False)
    # ax.axis('off')
    # ax.axis('tight')
    #
    # row_header = ['First Value', '25%', '50%', '75%', 'Last']
    # models_acc_data = np.array(collum_data).transpose()
    #
    # tab = table(plt.gca(),
    #             cellText=models_acc_data,
    #             rowLabels=row_header,
    #             colLabels=header,
    #             loc='center',
    #             cellLoc='center')
    #
    # tab.auto_set_column_width(col=list(range(len(header))))
    # fig.tight_layout()
    # plt.show()
    # plt.clf()


def line_plot_metric_data(lbl, metric_data, obf_f_name, title):
    nr_intensity_levels = len(metric_data)
    x_list = [x for x in range(0, nr_intensity_levels)]
    fig = plt.figure()
    fig.set_dpi(100)
    plt.plot(x_list, metric_data, label=lbl)
    plt.legend()
    plt.title(title)
    plt.xlabel('{} intensity level'.format(obf_f_name))
    # plt.title('{} by Obfuscation Intensity for {}'.format(title, obf_f_name))
    plt.show()
    plt.clf()
    return x_list


# Plots performance data for N number of models with N number of obfuscation functions
def plot_obs_f_performance_by_class(model_name, obf_f_name, obf_f_index, parsed_perf_by_class):
    title_loss = "NN models Loss"
    title_acc = "NN models Accuracy"
    nr_intensities = len(parsed_perf_by_class[0][1])
    x_list = [x for x in range(0, nr_intensities)]

    fig = plt.figure()
    fig.set_size_inches(17, 10)
    fig.set_dpi(200)
    gs = fig.add_gridspec(2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    header = []
    row_data = []
    for class_nr, class_perf_loss, class_perf_acc in parsed_perf_by_class:
        lbl = "Class {}".format(class_nr)

        ax1.plot(x_list, class_perf_loss, label=lbl)
        ax2.plot(x_list, class_perf_acc, label=lbl)
        header.append(lbl)
        first, half, last, one_quarter, three_quarters = get_data_samples(class_perf_acc)
        row_data.append([first, one_quarter, half, three_quarters, last])

    ax1.set_title(title_loss)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('Noise Intensity level')
    ax1.legend()
    ax2.set_title(title_acc)
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('Noise Intensity level')
    ax2.legend()
    plt.subplots_adjust(hspace=0.7)
    plt.title('Model Accuracy and Loss by Obfuscation Intensity for {}'.format(obf_f_name))
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


def evaluate_by_class(model, obfuscated_x, y_model_input):

    nr_classes = y_model_input.shape[1]
    by_class_perf = []

    for cls_index in range(nr_classes):
        single_class_map = y_model_input[:, cls_index] == 1

        obfuscated_x_single_class = obfuscated_x[single_class_map]
        y_model_input_single_class = y_model_input[single_class_map]
        obfuscated_single_class_perf = model.evaluate(obfuscated_x_single_class, y_model_input_single_class)
        by_class_perf.append((cls_index, obfuscated_single_class_perf))

    return by_class_perf


def get_target_shap(gen_shap_values, x_gen_test, y_gen_test):

    m_shap_list = []
    f_shap_list = []
    print("Parsing Shap values. ")
    for index in tqdm(range(x_gen_test.shape[0])):
        ismale = y_gen_test[index][0]

        # Male
        if ismale:
            shap_value = gen_shap_values[0][index]
            m_shap_list.append(np.squeeze(shap_value, axis=1))
        else:
            shap_value = gen_shap_values[1][index]
            f_shap_list.append(np.squeeze(shap_value, axis=1))
    return m_shap_list, f_shap_list


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


def plot_shap_2(m_shap_list, f_shap_list):
    m_shap_np = np.array(m_shap_list)
    f_shap_p = np.array(f_shap_list)
    # temp = replace_outliers_by_std(temp, 3)
    # m_shap_np = whiten(m_shap_np)
    summation = np.mean(f_shap_p, axis=0) - np.mean(m_shap_np, axis=0)

    std = np.std(m_shap_np) - np.std(f_shap_p)
    plot_title = "Shap Value Mean (Male mean - Female mean)"
    shap_nr_features = m_shap_np.shape[1]
    x_list = ['C{}'.format(x) for x in range(shap_nr_features)]
    plt.figure(figsize=(25, 10))
    plt.bar(x_list, summation, yerr=std, ecolor='black', capsize=10)
    plt.title(plot_title, fontsize=28)
    plt.xlabel('Coefficient Order (Higher Order captures higher frequencies)', fontsize=22)
    plt.ylabel('Mel Frequency Cepstrum Coefficient (Mean)', fontsize=22)
    plt.xticks(fontsize=16)
    plt.show()


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


def extract_shap(model, shap_input, background_data, background_size):

    background = background_data[:background_size]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(shap_input, check_additivity=False)
    return shap_values, e


if __name__ == '__main__':
    main()
