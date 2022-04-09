import glob
import time

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from data_processing import pre_process_data
from util.custom_functions import collect_benchmark_metrics, plot_models_perf


def main():
    experiment_data_lite = "./obf_checkpoint/lite"
    experiment_data = "./obf_checkpoint"

    lite_model_paths = glob.glob("{}/model_obf_k_*".format(experiment_data_lite), recursive=True)
    full_model_paths = glob.glob("{}/model_obf_k_*".format(experiment_data), recursive=True)
    model_meta_paths = glob.glob("{}/model_obf_meta_k_-*.npy".format(experiment_data), recursive=True)

    # get dataset
    print("Pre-processing audio files!")
    db_name = 'ravdess'
    # datasets
    audio_files_path = "./NNDatasets/audio"

    x_train, x_test, y_tr_emo, y_te_emo, y_tr_gen, y_te_gen, y_tr_id, y_te_id = pre_process_data(audio_files_path,
                                                                                                 db_name)
    print("Pre-processing audio files Complete!")

    # Scaling and setting type to float32
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train).astype(np.float32)
    x_test_scaled = sc.transform(x_test).astype(np.float32)

    lite_model_execution_time_list = []
    full_model_execution_time_list = []
    model_size_list = []

    util_sv_perf_dict = {}
    priv_e_perf_dict = {}
    priv_g_perf_dict = {}

    for lite_mdl_path, full_mld_path, meta_paths in zip(lite_model_paths, full_model_paths, model_meta_paths):

        original_model = load_model(full_mld_path)

        print("Converting from saved model to lite model")
        # converter = tf.lite.TFLiteConverter.from_saved_model(lite_mdl_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(original_model)

        # converter.allow_custom_ops = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        tflite_model_quant = converter.convert()

        # lite_model_path = lite_mdl_path[:28] + 'lite_' + lite_mdl_path[28:]
        #
        # # Save the model.
        # if not Path(lite_model_path).exists():
        #     with open(lite_model_path, 'wb') as f:
        #         f.write(tflite_model_quant)

        print("Loading lite model")
        interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Loading input info
        model_features_map = np.load(meta_paths)
        model_features_size = model_features_map.shape[0]
        model_size_list.append(model_features_size)

        print("Sleeping for a second before next model evaluation.")
        print("Model input size: {}".format(model_features_size))
        time.sleep(1)
        total_lite_time = 0
        total_full_time = 0
        lite_predictions = np.zeros((x_test_scaled.shape[0], x_test_scaled.shape[1]), dtype=np.float32)
        full_predictions = np.zeros((x_test_scaled.shape[0], x_test_scaled.shape[1]), dtype=np.float32)

        for x_index, x_test in enumerate(x_test_scaled):
            # Getting only the features used to train the model
            model_input = x_test[model_features_map]
            model_input = model_input.reshape((1, model_features_size))

            time_lite_start = time.time()
            # mock_input = tf.constant(np.ones(model_features_size).reshape((1, model_features_size)), dtype=tf.float32)

            interpreter.set_tensor(input_details[0]['index'], model_input)
            interpreter.invoke()
            lite_prediction = interpreter.get_tensor(output_details[0]['index'])

            total_lite_time = total_lite_time + time.time() - time_lite_start

            time_full_start = time.time()
            full_prediction = original_model.predict(model_input)
            total_full_time = total_full_time + time.time() - time_full_start

            lite_predictions[x_index, :] = lite_prediction
            full_predictions[x_index, :] = full_prediction

        print("Time to execute Lite model {} was {}".format(lite_mdl_path, total_lite_time / x_test_scaled.shape[0]))
        print("Time to execute Full model {} was {}".format(full_mld_path, total_full_time / x_test_scaled.shape[0]))
        lite_model_execution_time_list.append(total_lite_time / x_test_scaled.shape[0])
        full_model_execution_time_list.append(total_full_time / x_test_scaled.shape[0])

        # Evaluating models performance with lite model
        util_sv_model = load_model(id_model_path)
        priv_emo_model = load_model(emo_model_path)
        priv_gen_model = load_model(gender_model_path)

        # Applying obfuscation masks
        obf_lite_input = lite_predictions + x_test_scaled
        obf_full_input = full_predictions + x_test_scaled

        collect_benchmark_metrics(util_sv_model, obf_lite_input, y_te_id, util_sv_perf_dict, lite_keys[0])
        collect_benchmark_metrics(priv_emo_model, obf_lite_input, y_te_emo, priv_e_perf_dict, lite_keys[0])
        collect_benchmark_metrics(priv_gen_model, obf_lite_input, y_te_gen, priv_g_perf_dict, lite_keys[0])

        collect_benchmark_metrics(util_sv_model, obf_full_input, y_te_id, util_sv_perf_dict, full_key)
        collect_benchmark_metrics(priv_emo_model, obf_full_input, y_te_emo, priv_e_perf_dict, full_key)
        collect_benchmark_metrics(priv_gen_model, obf_full_input, y_te_gen, priv_g_perf_dict, full_key)

        time.sleep(1)

    acc_plot_title = 'Quantized Obfuscator Model ACC Performance'
    plot_models_perf(priv_e_perf_dict, priv_g_perf_dict, util_sv_perf_dict, acc_plot_title, 'ACC', model_size_list)
    f1_plot_title = 'Quantized Obfuscator Model F1 Performance'
    plot_models_perf(priv_e_perf_dict, priv_g_perf_dict, util_sv_perf_dict, f1_plot_title, 'F1', model_size_list)

    np.save(f"{experiment_data}/model_lite_exec_results.npy", lite_model_execution_time_list)
    np.save(f"{experiment_data}/model_full_exec_results.npy", full_model_execution_time_list)

    plot_model_exec_time(lite_model_execution_time_list, model_size_list)
    plot_model_exec_time(full_model_execution_time_list, model_size_list)


def plot_acc_tradeoffs(e_data, gen_data, sv_date, plt_title, y_axis_label, x_ticks, key):
    import numpy as np
    import matplotlib.pyplot as plt
    # set width of bar
    barWidth = 0.25

    lite_key = ''

    fig = plt.subplots(figsize=(12, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(e_data))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    # Make the plot
    plt.bar(br1, e_data[key], color='r', width=barWidth, edgecolor='grey', label='Emotion ACC')
    plt.bar(br2, gen_data[key], color='g', width=barWidth, edgecolor='grey', label='Gender ACC')
    plt.bar(br3, sv_date[key], color='b', width=barWidth, edgecolor='grey', label='SV ACC')
    # Adding Xticks
    plt.xlabel('Private Bottom K removed', fontweight='bold', fontsize=15)
    plt.ylabel(y_axis_label, fontweight='bold', fontsize=15)

    x_plot_labels = x_ticks

    plt.xticks([r + barWidth for r in range(len(e_data))], x_plot_labels)
    plt.title = plt_title
    plt.legend()
    plt.show()


def plot_model_exec_time(plot_data, x_labels):
    import matplotlib.pyplot as plt
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(plot_data))
    # Make the plot
    plt.bar(br1, plot_data, color='r', width=barWidth, edgecolor='grey', label='Time to execute lite model')
    # Adding Xticks
    plt.xlabel('Model input size', fontweight='bold', fontsize=15)
    plt.ylabel('Time in seconds', fontweight='bold', fontsize=15)

    x_plot_labels = x_labels

    plt.xticks([r for r in range(len(plot_data))], x_plot_labels)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    lite_keys = ["Lite Model"]
    full_key = "Full Model"

    emo_model_path = "emo_checkpoint/emodel_scalarized_ravdess.h5"
    id_model_path = "sv_model_checkpoint/sver_model_scalarized_data.h5"
    gender_model_path = "gmodel_checkpoint/gmodel_scaled_ravdess.h5"

    main()
