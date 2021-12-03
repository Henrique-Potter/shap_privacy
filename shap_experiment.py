import shap
import numpy as np
import tensorflow as tf
from pathlib import Path

from scipy.cluster.vq import whiten
from tqdm import tqdm
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from data_processing import pre_process_data
from util import replace_outliers_by_std

tf.random.set_seed(42)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def main():
    emo_model_path = './emo_checkpoint/emodel.h5'
    gender_model_path = './gmodel_checkpoint/gmodel.h5'

    audio_files_path = "./NNDatasets/audio"

    gen_shap_df_path = './data/emo_shap_df.npy'

    print("Pre-processing audio files!")
    x_emo_test, x_emo_train, y_emo_test, y_emo_train = pre_process_data(audio_files_path, get_emotion_label=True)
    x_gen_test, x_gen_train, y_gen_test, y_gen_train = pre_process_data(audio_files_path, get_emotion_label=False)
    print("Pre-processing audio files Complete!")

    print("Building Neural Net")
    emo_model = load_model(emo_model_path)

    test_emo_perf = emo_model.evaluate(x_emo_test, y_emo_test)
    train_emo_perf = emo_model.evaluate(x_emo_train, y_emo_train)
    print("Emo Model Train perf is:{}, Test perf is:{}".format(train_emo_perf, test_emo_perf))

    gender_model = load_model(gender_model_path)
    test_gen_perf = gender_model.evaluate(x_gen_test, y_gen_test)
    train_gen_perf = gender_model.evaluate(x_gen_train, y_gen_train)
    print("Gen Model Train perf is:{}, Test perf is:{}".format(train_gen_perf, test_gen_perf))

    if not Path(gen_shap_df_path).exists():
        print("Calculating Shap values")
        # Generating Shap Values
        gen_shap_values, e = extract_shap(gender_model, x_gen_test, x_gen_test, 150)
        np.save(gen_shap_df_path, gen_shap_values)
    else:
        gen_shap_values = np.load(gen_shap_df_path)

    f_shap_list, m_shap_list = get_taget_shap(gen_shap_values, x_gen_test, y_gen_test)

    # ------------------------ Analyzing Shap values ------------------------
    shap_np_scaled_sorted, shap_sorted_scaled_avg, shap_sorted_indexes = analyse_shap_values(m_shap_list)

    sigma_range = 100
    obfuscated_perf_loss = []
    obfuscated_perf_acc = []

    for sigma in range(1, sigma_range):
        obfuscated_x = add_specific_noise(gen_shap_values, x_gen_test, y_gen_test, sigma)
        obfuscated_gen_perf = gender_model.evaluate(obfuscated_x, y_gen_test)
        obfuscated_perf_loss.append(obfuscated_gen_perf[0])
        obfuscated_perf_acc.append(obfuscated_gen_perf[1])

    title_loss = "Gender NN model Loss"
    title_acc = "Gender NN model Accuracy"
    x_list = [x for x in range(1, sigma_range)]
    figure, axis = plt.subplots(2)
    axis[0].plot(x_list, obfuscated_perf_loss, label="Train loss")
    axis[0].set_title(title_loss)
    axis[0].set_ylabel('loss')
    axis[0].set_xlabel('Noise multiplier')
    axis[0].legend()

    axis[1].plot(x_list, obfuscated_perf_acc, label="Train Accuracy")
    axis[1].set_title(title_acc)
    axis[1].set_ylabel('accuracy')
    axis[1].set_xlabel('Noise multiplier')
    axis[1].legend()

    plt.subplots_adjust(hspace=0.7)
    plt.show()


def add_specific_noise(gen_shap_values, x_gen_test, y_gen_test, sigma):
    # gen_shap_values_norm = normalizeData_0_1(gen_shap_values)
    gen_shap_values[gen_shap_values < 0] = 0

    mu = 0
    print("Parsing Shap values. ")
    for index in tqdm(range(x_gen_test.shape[0])):
        ismale = y_gen_test[index][0]

        # Male
        if ismale:
            shap_value = np.squeeze(gen_shap_values[0][index], axis=1)
            random_noise = sigma * np.random.randn(x_gen_test.shape[1])
            shap_scaled_noise = shap_value * random_noise

            x_gen_test[index, :, 0] = x_gen_test[index, :, 0] + shap_scaled_noise

        else:
            shap_value = np.squeeze(gen_shap_values[1][index], axis=1)
            random_noise = sigma * np.random.randn(x_gen_test.shape[1])
            shap_scaled_noise = shap_value * random_noise

            x_gen_test[index, :, 0] = x_gen_test[index, :, 0] + shap_scaled_noise

    return x_gen_test.copy()


def get_taget_shap(gen_shap_values, x_gen_test, y_gen_test):

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
    return f_shap_list, m_shap_list


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

    for index in range(shap_nr_samples):
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


def extract_shap(model, shap_input, background_data, background_size):

    background = background_data[:background_size]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(shap_input, check_additivity=False)
    return shap_values, e


if __name__ == '__main__':
    main()
