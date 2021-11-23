import glob
import shap
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from data_processing import pre_process_data
from scipy.cluster.vq import vq, kmeans, whiten
tf.random.set_seed(42)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def main():
    gen_shap_df_path = './data/emo_label_df.npy'

    audio_files_path = "G:\\NNDatasets\\audio"
    audio_files = glob.glob("{}/**/*.wav".format(audio_files_path), recursive=True)

    print("Pre-processing audio files!")
    x_emo_test, x_emo_train, y_emo_test, y_emo_train = pre_process_data(audio_files, get_emotion_label=True)
    x_gen_test, x_gen_train, y_gen_test, y_gen_train = pre_process_data(audio_files, get_emotion_label=False)
    print("Pre-processing audio files Complete!")

    print("Building Neural Net")
    emo_model_path = './emo_checkpoint/emodel.h5'
    gender_model_path = './gmodel_checkpoint/gmodel.h5'
    emo_model = load_model(emo_model_path)

    test_emo_perf = emo_model.evaluate(x_emo_test, y_emo_test)
    train_emo_perf = emo_model.evaluate(x_emo_train, y_emo_train)
    print("Emo Model Train perf is:{}, Test perf is:{}".format(train_emo_perf, test_emo_perf))

    gender_model = load_model(gender_model_path)
    test_gen_perf = gender_model.evaluate(x_gen_test, y_gen_test)
    train_gen_perf = gender_model.evaluate(x_gen_train, y_gen_train)
    print("Gen Model Train perf is:{}, Test perf is:{}".format(train_gen_perf, test_gen_perf))

    one_class_emo = []
    one_class_gen = []

    # # One class only
    # print("Isolating single class")
    # for mfcc_idx in tqdm(range(x_emo_train.shape[0])):
    #     if y_emo_train[mfcc_idx][5] == 1:
    #         one_class_emo.append(x_emo_train[mfcc_idx])
    #     if y_gen_train[mfcc_idx][0] == 0:
    #         one_class_gen.append(x_gen_train[mfcc_idx])

    # shap_input_np = np.array(one_class_emo[:10])
    # # Model, Data to find Shap values, background, background size
    # emo_shap_values = extract_shap(emo_model, shap_input_np, x_emo_train, 100)
    # plot_shap_values(emo_shap_values, shap_input_np)
    #
    # shap_input_np = np.array(one_class_gen[:10])
    # # Model, Data to find Shap values, background, background size
    # gen_shap_values = extract_shap(gender_model, shap_input_np, x_gen_train, 100)
    # plot_shap_values(gen_shap_values, shap_input_np)

    if not Path(gen_shap_df_path).exists():
        # Generating Shap Values
        gen_shap_values = extract_shap(gender_model, x_gen_train, x_gen_train, 100)
        np.save(gen_shap_df_path, gen_shap_values)
    else:
        gen_shap_values = np.load(gen_shap_df_path)

    m_shap_list = []
    m_input_list = []

    f_shap_list = []

    print("Parsing Shap values. ")
    for index in tqdm(range(x_gen_train.shape[0])):
        true_label_index = np.where(y_gen_train[index] == 1)[0][0]

        # Male
        if true_label_index == 0:
            shap_value = gen_shap_values[true_label_index][index]
            m_shap_list.append(np.squeeze(shap_value, axis=1))
            x_gen_train_audio = x_gen_train[index]
            m_input_list.append(x_gen_train_audio)
        else:
            shap_value = gen_shap_values[true_label_index][index]
            f_shap_list.append(np.squeeze(shap_value, axis=1))

    x_list = [x for x in range(259)]

    # ------------------------ Analyzing Shap values ------------------------
    m_shap_np_scaled = analyse_shap_values(m_shap_list, x_list)
    f_shap_np_scaled = analyse_shap_values(f_shap_list, x_list)

    m_shap_sum = np.sum(m_shap_np_scaled, axis=0)
    f_shap_sum = np.sum(f_shap_np_scaled, axis=0)

    m_positive_shap_map = m_shap_sum > 0
    f_positive_shap_map = f_shap_sum > 0

    # intersection_map = np.where(m_positive_shap_map & f_positive_shap_map)
    intersection_map = m_positive_shap_map & f_positive_shap_map

    fig = plt.figure()
    m_shap_sum2 = np.copy(m_shap_sum)
    m_shap_sum[m_shap_sum < 0] = 0
    f_shap_sum[f_shap_sum < 0] = 0
    m_shap_sum2[~intersection_map] = 0

    plt.bar(x_list, m_shap_sum, color='b')
    plt.bar(x_list, f_shap_sum, color='r')
    plt.bar(x_list, m_shap_sum2, color='g')
    plt.show()


def analyse_shap_values(m_shap_list, x_list):

    shap_np = np.array(m_shap_list)
    shap_np_scaled = NormalizeData_minus1_1(shap_np)
    shap_np_scaled_whitened = whiten(shap_np_scaled)

    codebook, distortion = kmeans(shap_np_scaled_whitened, 8)
    plot_scatter_first_feature(codebook, shap_np_scaled_whitened)
    plot_scatter_first_vs_all(codebook, shap_np, shap_np_scaled_whitened)

    shap_sorted_indexes = np.argsort(shap_np_scaled[0, :])
    sorted_shap = shap_np_scaled[:, shap_sorted_indexes]
    codebook, distortion = kmeans(sorted_shap, 8)

    plt.scatter(sorted_shap[:, 0], sorted_shap[:, 1], c='b')
    plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    plt.show()

    fig = plt.figure()
    y_sum = np.sum(shap_np_scaled, axis=0)
    plt.bar(x_list, y_sum)
    plt.show()

    fig = plt.figure()
    y_sum = np.sum(whiten(shap_np_scaled_whitened), axis=0)
    plt.bar(x_list, y_sum)
    plt.show()

    return shap_np_scaled


def plot_scatter_first_vs_all(codebook, m_shap_np, m_shap_np_scaled_whitened):
    for index in range(m_shap_np.shape[1]):
        plt.scatter(m_shap_np_scaled_whitened[:, 0], m_shap_np_scaled_whitened[:, index], c='b')
    plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    plt.show()


def plot_scatter_first_feature(codebook, m_shap_np_scaled_whitened):
    plt.scatter(m_shap_np_scaled_whitened[:, 0], m_shap_np_scaled_whitened[:, 1])
    plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    plt.show()


def NormalizeData_0_1(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def NormalizeData_minus1_1(data):
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


def extract_shap(emo_model, shap_input, x_traincnn, background_size):

    background = x_traincnn[:background_size]
    e = shap.DeepExplainer(emo_model, background)
    shap_values = e.shap_values(shap_input)
    return shap_values


if __name__ == '__main__':
    main()
