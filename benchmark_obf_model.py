import glob
import time

import tensorflow as tf
import numpy as np


def main():
    experiment_data = "./obf_checkpoint"

    model_paths = glob.glob("{}/model_obf_k_-*.h5".format(experiment_data), recursive=True)
    model_meta_paths = glob.glob("{}/model_obf_meta_k_-*.npy".format(experiment_data), recursive=True)

    model_execution_time_list = []
    model_size_list = []

    for path, meta_paths in zip(model_paths, model_meta_paths):
        model = tf.keras.models.load_model(path)
        model.compile()
        model_features_map = np.load(meta_paths)
        model_features_size = model_features_map.shape[0]
        model_size_list.append(model_features_size)

        print("Sleeping for a second before next model mock test.")
        print("Model input size: {}".format(model_features_size))
        time.sleep(1)
        mock_input = tf.constant(np.ones(model_features_size).reshape((1, model_features_size)))

        # mdl_train_dataset = tf.data.Dataset.from_tensor_slices(mock_input)
        # tr_batchdt = mdl_train_dataset.padded_batch(1, drop_remainder=True)
        with tf.device('/cpu:0'):
            start = time.time()
            mock_output = model.predict(mock_input)
            end = time.time() - start

            print("Time to execute {} was {}".format(path, end))
            model_execution_time_list.append(end)

        time.sleep(1)
        # model.reset_states()
        # tf.keras.backend.clear_session()
        # tf.compat.v1.reset_default_graph()
        # print("TF session cleaned!")

    np.save("./model_exec_results.npy", model_execution_time_list)

    plot_model_exec_time(model_execution_time_list, model_size_list)


def plot_model_exec_time(plot_data, x_labels):

    import matplotlib.pyplot as plt
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(plot_data))
    # Make the plot
    plt.bar(br1, plot_data, color='r', width=barWidth, edgecolor='grey', label='Time to execute model')
    # Adding Xticks
    plt.xlabel('Model size', fontweight='bold', fontsize=15)
    plt.ylabel('Time in seconds', fontweight='bold', fontsize=15)

    x_plot_labels = x_labels

    plt.xticks([r for r in range(len(plot_data))], x_plot_labels)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
