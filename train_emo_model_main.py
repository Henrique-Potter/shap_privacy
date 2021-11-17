import glob
from pathlib import Path
import shap
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from TrainingPlot import PlotLosses
from data_processing import pre_process_data
from emonet import build_emo_model, build_gender_model
from util import train_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

emo_model_path = './emo_checkpoint/trained_model'

get_emotion_label = True


def main():

    audio_files_path = "G:\\NNDatasets\\audio"
    audio_files = glob.glob("{}/**/*.wav".format(audio_files_path), recursive=True)

    print("Pre-processing audio files!")
    x_testcnn, x_traincnn, y_test, y_train = pre_process_data(audio_files, get_emotion_label)
    print("Pre-processing audio files Complete!")

    print("Building Neural Net")

    model = build_emo_model()
    model_path = emo_model_path

    if not Path(model_path).exists():
        train_model(model, model_path, x_testcnn, x_traincnn, y_test, y_train, get_emotion_label)
    else:
        # Restore the weights
        model = tf.keras.models.load_model(model_path)
        test_acc = model.evaluate(x_testcnn, y_test, batch_size=128)
        train_acc = model.evaluate(x_traincnn, y_train, batch_size=128)

    one_class = []
    # One class only
    for mfcc_idx in range(x_traincnn.shape[0]):
        if y_train[mfcc_idx][5] == 1:
            one_class.append(x_traincnn[mfcc_idx])

    one_class_np = np.array(one_class[:10])
    #background = x_testcnn[:100]
    background = x_traincnn[:100]
    test_images_classes = y_test[:100]
    #test_images_all_classes = images_all[:10].to(device)
    #shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    e = shap.DeepExplainer(model, background)
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

    cnnhistory = model.evaluate(x_testcnn, y_test, batch_size=128)
    result = model.predict(x_testcnn[0])



if __name__ == '__main__':
    main()
