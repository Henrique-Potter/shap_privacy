import glob
from pathlib import Path
import shap
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from TrainingPlot import PlotLosses
from data_processing import pre_process_data
from emonet import build_emo_model, build_gender_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def main():

    audio_files_path = "G:\\NNDatasets\\audio"
    audio_files = glob.glob("{}/**/*.wav".format(audio_files_path), recursive=True)

    print("Pre-processing audio files!")
    x_testcnn, x_traincnn, y_emo_test, y_emo_train = pre_process_data(audio_files, get_emotion_label=True)
    _, _, y_gen_test, y_gen_train = pre_process_data(audio_files, get_emotion_label=False)
    print("Pre-processing audio files Complete!")

    print("Building Neural Net")

    emo_model_path = './checkpoints/trained_model'
    gender_model_path = './gmodel_checkpoints/trained_model'

    emo_model = tf.keras.models.load_model(emo_model_path)
    gender_model = tf.keras.models.load_model(gender_model_path)

    test_emo_acc = emo_model.evaluate(x_testcnn, y_emo_test, batch_size=128)
    train_emo_acc = emo_model.evaluate(x_traincnn, y_emo_train, batch_size=128)

    test_gen_acc = gender_model.evaluate(x_testcnn, y_gen_test, batch_size=128)
    train_gen_acc = gender_model.evaluate(x_traincnn, y_gen_train, batch_size=128)

    one_class_emo = []
    one_class_gen = []
    # One class only
    for mfcc_idx in range(x_traincnn.shape[0]):
        if y_emo_train[mfcc_idx][5] == 1:
            one_class_emo.append(x_traincnn[mfcc_idx])
        if y_gen_train[mfcc_idx][0] == 0:
            one_class_gen.append(x_traincnn[mfcc_idx])

    one_class_np = np.array(one_class_emo[:10])
    #background = x_testcnn[:100]
    background = x_traincnn[:100]
    test_images_classes = y_emo_train[:100]

    e = shap.DeepExplainer(emo_model, background)
    shap_values = e.shap_values(one_class_np)

    new_list = []
    for shap_value in shap_values:
        temp = shap_value.reshape((shap_value.shape[0], 1, 259, 1))
        temp2 = np.tile(temp, (100, 1, 1))
        new_list.append(temp2)

    # (# samples x width x height x channels),
    reshaped_input = one_class_np.reshape((one_class_np.shape[0], 1, 259, 1))
    tilling = np.tile(reshaped_input, (100, 1, 1))
    shap.image_plot(new_list, tilling)

    # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    # test_numpy = np.swapaxes(np.swapaxes(one_class_np, 1, -1), 1, 2)
    # shap.image_plot(shap_numpy, -test_numpy)

    


if __name__ == '__main__':
    main()
