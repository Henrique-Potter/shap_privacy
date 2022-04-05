import glob
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tqdm import tqdm
from keras.utils.layer_utils import count_params
import pandas as pd

from data_processing import pre_process_data, to_batchdataset
from util.custom_functions import collect_benchmark_metrics, plot_models_perf, add_list_in_dict, obfuscate_input, \
    collect_perf_metrics, priv_util_plot_acc_data, priv_util_plot_f1_data, get_pruning_configs, plot_obf_loss_from_dict
from util.trim_insig_weights import inspect_weigths


def main():
    # experiment_data = "./obf_checkpoint/large_swish"
    experiment_data = "./obf_checkpoint/small_swish"

    full_model_paths = glob.glob("{}/model_obf_k_*".format(experiment_data), recursive=True)
    model_meta_paths = glob.glob("{}/model_obf_meta_k_-*.npy".format(experiment_data), recursive=True)

    # get dataset
    print("Pre-processing audio files!")
    db_name = 'ravdess'
    # datasets
    audio_files_path = "./NNDatasets/audio"

    x_train, x_test, y_tr_emo, y_te_emo, y_tr_gen, y_te_gen, y_tr_id, y_te_id = pre_process_data(audio_files_path, db_name)
    print("Pre-processing audio files Complete!")

    # Scaling and setting type to float32
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train).astype(np.float32)
    x_test_scaled = sc.transform(x_test).astype(np.float32)

    execution_time_dict = {}
    model_size_list = []

    util_sv_perf_dict = {}
    priv_e_perf_dict = {}
    priv_g_perf_dict = {}

    for full_mld_path, meta_paths in zip(full_model_paths, model_meta_paths):

        original_model = load_model(full_mld_path)

        # Loading input info
        model_features_map = np.load(meta_paths)
        model_features_size = model_features_map.shape[0]
        model_size_list.append(model_features_size)

        attemp_configs = get_pruning_configs(end_step=None)

        # Iterating over pruning configurations
        for pruning_config in list(attemp_configs.items()):

            pruning_model_name = pruning_config[0]
            pre_tr_model_id = full_mld_path[29:]
            full_model_name = f"{pruning_model_name}_{pre_tr_model_id}"
            pruned_mdl_path = experiment_data + f"/{full_model_name}"

            if Path(pruned_mdl_path).exists():
                print("Loading pruned model")
                pruned_model = load_model(pruned_mdl_path)
                model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
            else:
                print("Pruning model")
                pruned_model = prune_model(pruning_model_name, original_model, model_features_map, x_train_scaled, x_test_scaled, y_tr_id, y_te_id, y_te_gen, y_te_emo, pruning_config[1])

                model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
                pruned_model.save(pruned_mdl_path, include_optimizer=False)

            print("Sleeping for a second before model evaluation.")
            print("Model input size: {}".format(model_features_size))
            time.sleep(1)

            total_full_time = 0
            full_predictions = np.zeros((x_test_scaled.shape[0], x_test_scaled.shape[1]), dtype=np.float32)

            for x_index, x_instance in enumerate(x_test_scaled):
                # Getting only the features used to train the model
                model_input = x_instance[model_features_map]
                model_input = model_input.reshape((1, model_features_size), )

                time_full_start = time.time()

                full_prediction = model_for_export.predict(model_input)
                total_full_time = total_full_time + time.time() - time_full_start

                full_predictions[x_index, :] = full_prediction

            print("Time to execute Full model {} was {}".format(full_mld_path, total_full_time / x_test_scaled.shape[0]))

            if pruning_model_name in execution_time_dict:
                execution_time_dict[pruning_model_name].append(total_full_time)
            else:
                execution_time_dict[pruning_model_name] = [total_full_time]

            # Applying obfuscation masks
            obf_full_input = full_predictions + x_test_scaled

            collect_benchmark_metrics(util_sv_model, obf_full_input, y_te_id, util_sv_perf_dict, pruning_model_name)
            collect_benchmark_metrics(priv_emo_model, obf_full_input, y_te_emo, priv_e_perf_dict, pruning_model_name)
            collect_benchmark_metrics(priv_gen_model, obf_full_input, y_te_gen, priv_g_perf_dict, pruning_model_name)

            time.sleep(1)

    perf_acc_key = 'ACC'
    perf_f1_key = 'F1'
    perf_title = f'Model Performance'
    # plot_models_perf(priv_e_perf_dict, priv_g_perf_dict, util_sv_perf_dict, perf_title, perf_acc_key, model_size_list)
    plot_models_perf(priv_e_perf_dict, priv_g_perf_dict, util_sv_perf_dict, perf_title, perf_f1_key, model_size_list)

    exec_timepd = pd.DataFrame.from_dict(execution_time_dict)
    exec_timepd.to_csv(index=False)
    # plot_model_exec_time(execution_time_dict, model_size_list)


def prune_model(model_name, original_model, model_features, x_train, x_test, util_sv_trlabels, util_sv_telabels, priv_g_labels, priv_e_labels, pruning_schedule):

    fmask_x_train_data = x_train[:, model_features]
    fmask_x_test_data = x_test[:, model_features]
    test_size = x_test.shape[0]

    batch_size = 32
    util_xy_tr_batch_dt = tf.data.Dataset.from_tensor_slices((x_train, util_sv_trlabels)).padded_batch(batch_size, drop_remainder=True)
    # Converting the feature specific input to tensors
    masked_x_tr_batch_dt, _ = to_batchdataset(fmask_x_train_data, None, batch_size)

    util_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    priv_e_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    priv_g_loss_fn = tf.keras.losses.BinaryCrossentropy()

    nr_e_classes = priv_e_labels[0].shape[0]
    nr_g_classes = priv_g_labels[0].shape[0]
    nr_sv_classes = util_sv_trlabels[0].shape[0]

    # Target values should match no better than random guess.
    priv_e_label = tf.constant(np.tile(np.ones(nr_e_classes) * 1 / nr_e_classes, (batch_size, 1)))
    priv_g_label = tf.constant(np.tile(np.ones(nr_g_classes) * 1 / nr_g_classes, (batch_size, 1)))

    e_test_label = tf.constant(np.tile(np.ones(nr_e_classes) * 1 / nr_e_classes, (test_size, 1)))
    g_test_label = tf.constant(np.tile(np.ones(nr_g_classes) * 1 / nr_g_classes, (test_size, 1)))

    unused_arg = -1
    epochs = pruning_schedule[1]

    num_images = x_train.shape[0]
    end_step_c = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    initial_sparsity = 0
    final_sparsity = 0.75
    begin_step = 0
    end_step = end_step_c

    pruning_params = {
        'pruning_schedule': pruning_schedule[0]
    }

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(original_model, **pruning_params)

    # Boilerplate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000005)

    # Non-boilerplate.
    model_for_pruning.optimizer = optimizer
    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(model_for_pruning)
    model_for_pruning.compile()
    step_counter = 0

    losses = {}
    util_sv_perf_list = [[], []]
    priv_e_perf_list = [[], []]
    priv_g_perf_list = [[], []]

    # run pruning callback
    step_callback.on_train_begin()

    for epoch in tqdm(range(epochs)):
    # while stop_training_flag:
        # log_callback.on_epoch_begin(epoch=unused_arg)  # run pruning callback

        for (x_tr_batch, util_tr_y_batch), masked_x_tr_batch in zip(util_xy_tr_batch_dt, masked_x_tr_batch_dt):
            # run pruning callback
            step_callback.on_train_batch_begin(batch=unused_arg)

            peloss, pgloss, tloss, uloss, grads = train_pruned_model(masked_x_tr_batch, model_for_pruning,
                                                                     priv_e_label, priv_e_loss_fn, priv_g_label,
                                                                     priv_g_loss_fn, util_loss_fn, util_tr_y_batch, x_tr_batch)

            optimizer.apply_gradients(zip(grads, model_for_pruning.trainable_variables))
            step_counter += 1

        # run pruning callback
        step_callback.on_epoch_end(batch=unused_arg)

        print(f"\n ------- EPOCH NUMBER: {epoch} for MODEL: {model_name} -------")

        stripped_weights = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        num_of_w_org, num_of_nz_w_org, num_of_z_w_org = inspect_weigths('original (unpruned)', stripped_weights)
        zero_ratio = num_of_z_w_org / num_of_w_org
        print(f"Number of Non-Zeroes:{num_of_nz_w_org}, Number of Zeroes: {num_of_z_w_org} Sparsity level: {zero_ratio}")

        add_list_in_dict(losses, "Emotion model Loss", peloss)
        add_list_in_dict(losses, "Gender model Loss", pgloss)
        add_list_in_dict(losses, "SV model Loss", uloss)

        current_train_loss = uloss+pgloss+peloss
        add_list_in_dict(losses, "Train Loss", current_train_loss)

        obf_input = obfuscate_input(stripped_weights, fmask_x_test_data, x_test)

        sv_loss = 7 * util_sv_model.evaluate(obf_input, util_sv_telabels, batch_size=32, verbose=False)[0]
        emo_loss = 2 * priv_emo_model.evaluate(obf_input, e_test_label, batch_size=32, verbose=False)[0]
        gen_loss = 2 * priv_gen_model.evaluate(obf_input, g_test_label, batch_size=32, verbose=False)[0]

        test_loss = sv_loss + emo_loss + gen_loss
        add_list_in_dict(losses, "Test Loss", test_loss)
        print(f"Step ({step_counter}) E-loss {peloss} G-loss {pgloss} SV-loss {uloss} Total loss: {current_train_loss} Test Loss: {test_loss}")

        collect_perf_metrics(util_sv_model, obf_input, util_sv_telabels, util_sv_perf_list)
        collect_perf_metrics(priv_emo_model, obf_input, priv_e_labels, priv_e_perf_list)
        collect_perf_metrics(priv_gen_model, obf_input, priv_g_labels, priv_g_perf_list)

        # Plotting results.
        if (epoch + 1) % plot_epoch_rate == 0:
            x_label = "Number of Epochs"
            # priv_util_plot_acc_data(priv_e_perf_list[0], priv_g_perf_list[0], util_sv_perf_list[0],
            #                         f'Pruned NN Obfuscator ACC Performance', x_label)
            priv_util_plot_f1_data(priv_e_perf_list[1], priv_g_perf_list[1], util_sv_perf_list[1],
                                   f"Pruned NN Obfuscator F1 Performance for {model_name}", x_label)
            plot_obf_loss_from_dict(losses)

    x_label = "Number of epochs"
    priv_util_plot_f1_data(priv_e_perf_list[1], priv_g_perf_list[1], util_sv_perf_list[1],
                           f"Pruned NN Obfuscator F1 Performance for {model_name}", x_label)
    plot_obf_loss_from_dict(losses)

    return model_for_pruning


@tf.function
def train_pruned_model(masked_x_tr_batch, model_for_pruning, priv_e_label, priv_e_loss_fn, priv_g_label,
                       priv_g_loss_fn, util_loss_fn, util_tr_y_batch, x_tr_batch):
    with tf.GradientTape() as tape:
        logits = model_for_pruning(masked_x_tr_batch, training=True)
        tape.watch(logits)
        obf_input = logits + x_tr_batch

        epriv_mdl_logits = priv_emo_model(obf_input, training=False)
        peloss = 2 * priv_e_loss_fn(priv_e_label, epriv_mdl_logits)

        gpriv_mdl_logits = priv_gen_model(obf_input, training=False)
        pgloss = 2 * priv_g_loss_fn(priv_g_label, gpriv_mdl_logits)

        svpriv_mdl_logits = util_sv_model(obf_input, training=False)
        uloss = 7 * util_loss_fn(util_tr_y_batch, svpriv_mdl_logits)

        tloss = peloss + pgloss + uloss

        grads = tape.gradient(tloss, model_for_pruning.trainable_variables)

    return peloss, pgloss, tloss, uloss, grads


def get_gzipped_model_size(model, mdl_name, zip_name):
    # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    if not Path(mdl_name).exists():
        model.save(mdl_name, include_optimizer=False)

    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(mdl_name)

    return os.path.getsize(zip_name)


def quantize_model(original_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
    # converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    tflite_model_quant = converter.convert()
    print("Loading lite model")
    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return input_details, interpreter, output_details


def quantize_float16_model(original_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
    # converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.experimental_new_converter = True
    tflite_model_quant = converter.convert()
    print("Loading lite model")
    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return input_details, interpreter, output_details


def full_integer_quantize_model(original_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
    # converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.int8  # or tf.uint8
    # converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()
    print("Loading lite model")
    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return input_details, interpreter, output_details


def representative_dataset():
    pass
    #
    # for data in tf.data.Dataset.from_tensor_slices((dt_input)).batch(1).take(100):
    #     yield [tf.dtypes.cast(data, tf.float32)]


def representative_dataset2():
    for _ in range(100):
        data = np.random.rand(1, 244, 244, 3)
        yield [data.astype(np.float32)]


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
    plt.bar(br1, plot_data, width=barWidth, edgecolor='grey', label='Time to execute lite model')
    # plt.bar_label(plot_data)

    # Adding Xticks
    plt.xlabel('Model input size', fontweight='bold', fontsize=15)
    plt.ylabel('Time in seconds', fontweight='bold', fontsize=15)

    x_plot_labels = x_labels

    plt.xticks([r for r in range(len(plot_data))], x_plot_labels)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Used for Full integer model quantization!

    lite_keys = ["Lite Model"]
    full_key = "Full Model"

    emo_model_path = "emo_checkpoint/emodel_scalarized_ravdess.h5"
    id_model_path = "sv_model_checkpoint/sver_model_scalarized_data.h5"
    gender_model_path = "gmodel_checkpoint/gmodel_scaled_ravdess.h5"

    # Evaluating models performance with lite model
    util_sv_model = load_model(id_model_path)
    priv_emo_model = load_model(emo_model_path)
    priv_gen_model = load_model(gender_model_path)

    plot_epoch_rate = 100

    main()
