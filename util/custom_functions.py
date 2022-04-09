import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage


# from util.PerClassMetrics import PerClassMetrics


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=hop_length * 2, hop_length=hop_length)
    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)


# def calculate_and_plot_conf_matrix(model, x_test, y_test, model_id):
#     from sklearn.metrics import confusion_matrix
#
#     y_predict = np.asarray(model.predict(x_test))
#     true = np.argmax(y_test, axis=1)
#     pred = np.argmax(y_predict, axis=1)
#     cm = confusion_matrix(true, pred)
#
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     cm = np.round(cm, 2)
#
#     plot_confusion_matrix(cm, model_id, 0)


def plot_confusion_matrix(cm_values, model_id, top_k):
    import matplotlib.pyplot as plt
    import seaborn as sn
    import pandas as pd
    cm_df = pd.DataFrame(cm_values)
    plt.figure(figsize=(10, 7))
    plt.title("Confusion Matrix for top K = {}".format(top_k))

    if model_id == 0:
        axis_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
        plt.title("Emotion Confusion Matrix for top K = {}".format(top_k))
    elif model_id == 1:
        axis_labels = ['Male', 'Female']
        plt.title("Gender Confusion Matrix for top K = {}".format(top_k))
    elif model_id == 2:
        axis_labels = [str(x) for x in range(cm_df.shape[0])]
        plt.title("Speaker Verification Confusion Matrix for top K = {}".format(top_k))

    s = sn.heatmap(cm_df, xticklabels=axis_labels, yticklabels=axis_labels, annot=True)
    s.set(xlabel='Predicted', ylabel='Actual')
    plt.show()


def calc_confusion_matrix(y_predict, y_test, top_k):
    from sklearn.metrics import confusion_matrix

    true_labels = np.argmax(y_test, axis=1)
    pred_labels = np.argmax(y_predict, axis=1)

    cm = confusion_matrix(true_labels, pred_labels)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)

    if len(y_test[0]) == 7:
        model_id = 0
    elif len(y_test[0]) == 2:
        model_id = 1
    elif len(y_test[0]) == 24:
        model_id = 2

    plot_confusion_matrix(cm, model_id, top_k)


def reject_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def replace_outliers_by_std(data, m=3.):
    u = np.mean(data)
    s = np.std(data)
    f1 = u - m * s
    f2 = u + m * s
    dt_median = np.median(data)
    data1 = np.where(data > f1, data, f1)
    data2 = np.where(data1 < f2, data1, f2)

    return data2


def replace_outliers_by_quartile(data, m=1.5):
    q1 = np.quantile(data, 0.25, axis=0)
    q3 = np.quantile(data, 0.75, axis=0)

    inter_quantile_range = q3 - q1

    upper_range = m * inter_quantile_range + q3
    lower_range = q1 - m * inter_quantile_range

    data1 = np.where(data > upper_range, upper_range, data)
    data2 = np.where(data1 < lower_range, lower_range, data1)

    return data2


def priv_util_plot_f1_data(priv_e_model_data, priv_g_model_data, util_model_data, title, x_label, x_ticks=None):
    lbl1 = "Speaker Verification F1 (Utility)"
    lbl2 = "Emotion F1 (Private)"
    lbl3 = "Gender F1 (Private)"

    x_lenght = len(priv_e_model_data)

    x_placeholder = [x for x in range(0, x_lenght)]

    fig = plt.figure()
    fig.set_dpi(100)

    plt.plot(x_placeholder, priv_e_model_data, label=lbl2)
    plt.plot(x_placeholder, priv_g_model_data, label=lbl3)
    plt.plot(x_placeholder, util_model_data, label=lbl1)

    if x_ticks is not None:
        plt.xticks(ticks=x_placeholder, labels=x_ticks)

    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("F1")

    plt.ylim([0, 1.1])
    plt.show()

    return x_ticks


def collect_perf_metrics(model, obf_input, util_sv_labels, perf_list):
    from sklearn.metrics import classification_report

    y_predictions = model.predict(obf_input)
    y_labels_pred = np.argmax(y_predictions, axis=1)
    y_labels_true = np.argmax(util_sv_labels, axis=1)
    class_report = classification_report(y_labels_true, y_labels_pred, zero_division=0, output_dict=True)
    perf_list[0].append(class_report['accuracy'])
    perf_list[1].append(class_report['macro avg']['f1-score'])

    return y_predictions


def collect_benchmark_metrics(model, obf_input, util_sv_labels, perf_dict, key):
    from sklearn.metrics import classification_report

    y_predictions = model.predict(obf_input)
    y_labels_pred = np.argmax(y_predictions, axis=1)
    y_labels_true = np.argmax(util_sv_labels, axis=1)
    class_report = classification_report(y_labels_true, y_labels_pred, zero_division=0, output_dict=True)

    if "ACC" in perf_dict:
        acc_dict = perf_dict["ACC"]
        model_key = key + " ACC"
        if model_key in acc_dict:
            acc_dict[model_key].append(class_report['accuracy'])
        else:
            acc_dict[model_key] = [class_report['accuracy']]
    else:
        model_dict = {}
        perf_dict["ACC"] = model_dict
        model_key = key + " ACC"
        model_dict[model_key] = [class_report['accuracy']]

    if "F1" in perf_dict:
        f1_dict = perf_dict["F1"]
        model_key = key + " F1"
        if model_key in f1_dict:
            f1_dict[model_key].append(class_report['macro avg']['f1-score'])
        else:
            f1_dict[model_key] = [class_report['macro avg']['f1-score']]
    else:
        model_dict = {}
        perf_dict["F1"] = model_dict
        model_key = key + " F1"
        model_dict[model_key] = [class_report['macro avg']['f1-score']]

    return y_predictions


def priv_util_plot_acc_data(priv_e_model_data, priv_g_model_data, util_model_data, title, x_label, x_ticks=None):
    lbl1 = "Speaker Verification ACC (Utility)"
    lbl2 = "Emotion ACC (Private)"
    lbl3 = "Gender ACC (Private)"

    x_lenght = len(priv_e_model_data)

    x_placeholder = [x for x in range(x_lenght)]

    fig = plt.figure()
    fig.set_dpi(100)

    plt.plot(x_placeholder, priv_e_model_data, label=lbl2)
    plt.plot(x_placeholder, priv_g_model_data, label=lbl3)
    plt.plot(x_placeholder, util_model_data, label=lbl1)

    if x_ticks is not None:
        plt.xticks(ticks=x_placeholder, labels=x_ticks)

    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")

    plt.ylim([0, 1.1])
    plt.show()

    return x_ticks


def obfuscate_input(model, obfuscator_x_input, clean_x_innput, model_features):
    input_temp = clean_x_innput.copy()

    # Generating the mask
    obf_masks = model.predict(obfuscator_x_input)

    # Adding the mask to the input
    input_temp[:, model_features] = input_temp[:, model_features] + obf_masks

    return input_temp


def add_list_in_dict(dictionary, key, val):

    if key in dictionary:
        dictionary[key].append(val)
    else:
        dictionary[key] = [val]


def plot_models_perf(priv_e_model_data, priv_g_model_data, util_model_data, title, key, x_ticks=None):
    import pandas as pd

    fig = plt.figure()
    fig.set_dpi(100)

    style_list = ["-", ":"]
    eperf_metric_data = priv_e_model_data[key]
    gperf_metric_data = priv_g_model_data[key]
    svperf_metric_data = util_model_data[key]

    for (ekey, evalues), (gkey, gvalues), (svkey, svvalues) in zip(eperf_metric_data.items(),
                                                                   gperf_metric_data.items(),
                                                                   svperf_metric_data.items()):

        full_lbl_sv = f"Speaker Verification {svkey} (Utility)"
        full_lbl_e = f"Emotion {ekey} (Private)"
        full_lbl_g = f"Gender {gkey} (Private)"

        x_lenght = len(list(evalues))
        x_placeholder = [x for x in range(x_lenght)]

        plt.plot(x_placeholder, list(evalues), label=full_lbl_e, antialiased=True, color='green')
        plt.plot(x_placeholder, list(gvalues), label=full_lbl_g, antialiased=True, color='red')
        plt.plot(x_placeholder, list(svvalues), label=full_lbl_sv, color='blue')

        if x_ticks is not None:
            plt.xticks(ticks=x_placeholder, labels=x_ticks)

        plt.legend()
        plt.title(title)
        x_label = "Model Input Size"
        plt.xlabel(x_label)
        plt.ylabel(key)

        plt.ylim([0, 1.1])
        plt.show()

        temp = {}
        temp[f"sv_{svkey}"] = svvalues
        temp[f"e_{ekey}"] = evalues
        temp[f"g_{gkey}"] = gvalues
        pd.DataFrame.from_dict(temp).to_csv(f"./obf_checkpoint/bench/{svkey}.csv")

        # np.savetxt(f"./obf_checkpoint/bench/sv_{svkey}.csv", svvalues, delimiter=",")
        # np.savetxt(f"./obf_checkpoint/bench/e_{ekey}.csv", evalues, delimiter=",")
        # np.savetxt(f"./obf_checkpoint/bench/g_{gkey}.csv", gvalues, delimiter=",")

    return x_ticks


def priv_plot_perf_data_by_class(priv_model_data):
    title = "ACC by class"

    epochs = len(priv_model_data[0][0])
    x_list = [x for x in range(1, epochs + 1)]

    fig = plt.figure()
    fig.set_dpi(100)

    class_id = 0
    for perf_dt in priv_model_data[1]:
        plt.plot(x_list, perf_dt, label="Class %i" % class_id)
        plt.bar_label(perf_dt)
        class_id += 1

    plt.legend()
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.ylim([0, 1.1])
    plt.show()

    return x_list


def priv_util_bar_plot_perf_data(priv_model_data, title):
    lbl1 = "Gender ACC (Private)"
    lbl2 = "Emotion ACC (Utility)"
    priv_model_data = np.squeeze(priv_model_data)
    topk_features = priv_model_data.shape[0]
    x_list = [x for x in range(1, topk_features + 1)]

    fig = plt.figure()
    fig.set_dpi(100)

    plt.bar(x_list, priv_model_data, label=lbl1)

    plt.legend()
    plt.title(title)
    plt.xlabel("Top K features used to train the Obfuscator")
    plt.ylabel("Accuracy")

    plt.ylim([0, 1])
    plt.show()

    return x_list


def validate_model(model, emo_model, gender_model, emo_test_dataset_batch, gen_test_dataset_batch):
    import tensorflow as tf
    emo_accuracy = tf.keras.metrics.CategoricalAccuracy()
    gen_accuracy = tf.keras.metrics.BinaryAccuracy()

    for (emo_test_x, emo_test_y), (gen_test_x, gen_test_y) in zip(emo_test_dataset_batch, gen_test_dataset_batch):
        mask = emo_test_x
        model_mask = model(mask)

        paddings = tf.constant([[0, 0], [0, 40 - model_mask.shape[1]]])
        final_mask = tf.pad(model_mask, paddings)

        # model_mask = tf.reshape(model_mask, (emo_test_x.shape[0], 40, 1))
        obfuscated_input = final_mask + emo_test_x
        # obfuscated_input = tf.reshape(obfuscated_input, (emo_test_x.shape[0], 40, 1))

        # get results
        preds = emo_model(obfuscated_input, training=False)
        emo_accuracy.update_state(y_true=emo_test_y, y_pred=preds)
        preds = gender_model(obfuscated_input, training=False)
        gen_accuracy.update_state(y_true=gen_test_y, y_pred=preds)

    print(emo_accuracy.result().numpy())
    print(gen_accuracy.result().numpy())


def plot_obf_loss_from_list(losses_perf):
    title = "Obfuscator Training Loss"

    losses_sz = len(losses_perf[0])
    x_list = [x for x in range(1, losses_sz + 1)]

    fig = plt.figure()
    fig.set_dpi(100)
    total_loss_perf = np.array(losses_perf[0])
    tepriv_loss_perf = np.array(losses_perf[1])
    tgpriv_loss_perf = np.array(losses_perf[2])
    trutil_loss_perf = np.array(losses_perf[3])

    plt.plot(x_list, total_loss_perf, label="Total Loss")
    plt.plot(x_list, tepriv_loss_perf, label="Emotion Model Loss")
    plt.plot(x_list, tgpriv_loss_perf, label="Gender Model Loss")
    plt.plot(x_list, trutil_loss_perf, label="Speaker Verification Model Loss")

    plt.legend()
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.show()


def plot_obf_loss_from_dict(losses_perf):
    title = "Obfuscator Training Loss"

    losses_sz = len(losses_perf["Emotion model Loss"])
    x_list = [x for x in range(1, losses_sz + 1)]

    fig = plt.figure()
    fig.set_dpi(100)
    tepriv_loss_perf = np.array(losses_perf["Emotion model Loss"])
    tgpriv_loss_perf = np.array(losses_perf["Gender model Loss"])
    trutil_loss_perf = np.array(losses_perf["SV model Loss"])
    train_loss_perf = np.array(losses_perf["Train Loss"])
    test_loss_perf = np.array(losses_perf["Test Loss"])

    plt.plot(x_list, tepriv_loss_perf, label="Emotion Model Loss")
    plt.plot(x_list, tgpriv_loss_perf, label="Gender Model Loss")
    plt.plot(x_list, trutil_loss_perf, label="Speaker Verification Model Loss")
    plt.plot(x_list, train_loss_perf, label="Train Total Loss")
    plt.plot(x_list, test_loss_perf, label="Test Total Loss")

    plt.legend()
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.show()


def show_spectrogram(file):
    sr, x = scipy.io.wavfile.read(file)
    ## Parameters: 10ms step, 30ms window
    nstep = int(sr * 0.01)
    nwin = int(sr * 0.03)
    nfft = nwin
    window = np.hamming(nwin)
    ## will take windows x[n1:n2].  generate
    ## and loop over n2 such that all frames
    ## fit within the waveform
    nn = range(nwin, len(x), nstep)
    X = np.zeros((len(nn), nfft // 2))
    for i, n in enumerate(nn):
        xseg = x[n - nwin:n]
        z = np.fft.fft(window * xseg, nfft)
        X[i, :] = np.log(np.abs(z[:nfft // 2]))
    plt.imshow(X.T, interpolation='nearest',
               origin='lower',
               aspect='auto')
    plt.show()


def show_amplitude(data, sampling_rate):
    import librosa.display
    plt.figure(figsize=(15, 5))
    librosa.display.waveplot(data, sr=sampling_rate)
    plt.show()


def mean_std_analysis(shap_list):
    class_index = 0
    nr_classes = len(shap_list)
    nr_features = len(shap_list[0][0])

    for shap_class_data in shap_list:
        shap_np = np.array(shap_class_data)
        # temp = replace_outliers_by_std(temp, 3)
        # m_shap_np = whiten(m_shap_np)
        summation = np.mean(shap_np, axis=0)
        std = np.std(shap_np)

        plot_title = "Shap Value Mean for class {}".format(class_index)
        x_list = ['C{}'.format(x) for x in range(nr_features)]
        plt.figure(figsize=(25, 10))
        plt.bar(x_list, summation, yerr=std, ecolor='black', capsize=10)
        plt.title(plot_title, fontsize=28)
        plt.xlabel('Coefficient Order (Higher Order captures higher frequencies)', fontsize=22)
        plt.ylabel('Mel Frequency Cepstrum Coefficient (Mean)', fontsize=22)
        plt.xticks(fontsize=16)
        plt.show()
        plt.clf()
        class_index += 1

    plt.figure(figsize=(18, 12))
    # plt.tight_layout()
    ind = np.arange(nr_features)
    x_list = ['C{}'.format(x) for x in range(nr_features)]
    width = 0.4
    colors = ['r', 'g', 'b']
    bar_charts = []

    for x in range(nr_classes):
        shap_np = np.array(shap_list[x])
        # temp = replace_outliers_by_std(temp, 3)
        # m_shap_np = whiten(m_shap_np)
        summation = np.mean(shap_np, axis=0)
        std = np.std(shap_np)
        bar = plt.bar(ind + width * x, summation, width, yerr=std, capsize=3)
        bar_charts.append(bar)

    plt.xlabel('Coefficient Order (Higher Order captures higher frequencies)', fontsize=22)
    plt.ylabel('Mel Frequency Cepstrum Coefficient (Mean)', fontsize=22)
    plt.title('Shap Value for all classes', fontsize=28)
    plt.xticks(ind + width, x_list)
    legend_label = ['Class {}'.format(x) for x in range(nr_classes)]
    plt.legend(bar_charts, legend_label)
    plt.show()


def get_pruning_configs(end_step):
    from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import PolynomialDecay, \
        ConstantSparsity

    attempt_configs = {
        # 'poly_decay_10/50': PolynomialDecay(
        #     initial_sparsity=0.10,
        #     final_sparsity=0.50,
        #     begin_step=0,
        #     end_step=end_step),
        # 'poly_decay_20/50': PolynomialDecay(
        #     initial_sparsity=0.20,
        #     final_sparsity=0.50,
        #     begin_step=0,
        #     end_step=end_step),
        # 'poly_decay_30/60': PolynomialDecay(
        #     initial_sparsity=0.30,
        #     final_sparsity=0.60,
        #     begin_step=0,
        #     end_step=end_step),
        # 'poly_decay_30/70': PolynomialDecay(
        #     initial_sparsity=0.30,
        #     final_sparsity=0.70,
        #     begin_step=0,
        #     end_step=end_step),
        # 'poly_decay_40/50': PolynomialDecay(
        #     initial_sparsity=0.40,
        #     final_sparsity=0.50,
        #     begin_step=0,
        #     end_step=end_step),
        # 'poly_decay_10/90': PolynomialDecay(
        #     initial_sparsity=0.10,
        #     final_sparsity=0.90,
        #     begin_step=0,
        #     end_step=end_step),
        'const_sparsity_0.1': (ConstantSparsity(
            target_sparsity=0.1, begin_step=0
        ), 5),
        'const_sparsity_0.3': (ConstantSparsity(
            target_sparsity=0.3, begin_step=0
        ), 60),
        'const_sparsity_0.5': (ConstantSparsity(
            target_sparsity=0.5, begin_step=0
        ), 150),
        'const_sparsity_0.7': (ConstantSparsity(
            target_sparsity=0.7, begin_step=0
        ), 250),
        'const_sparsity_0.9': (ConstantSparsity(
            target_sparsity=0.9, begin_step=0
        ), 600)
    }

    return attempt_configs
