import glob
import shap
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

from TrainingPlot import PlotLosses
from data_processing import pre_process_data
tf.random.set_seed(42)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def main():
    audio_files_path = "G:\\NNDatasets\\audio"
    audio_files = glob.glob("{}/**/*.wav".format(audio_files_path), recursive=True)

    print("Pre-processing audio files!")
    x_testcnn, x_traincnn, y_emo_test, y_emo_train = pre_process_data(audio_files, get_emotion_label=True)
    _, _, y_gen_test, y_gen_train = pre_process_data(audio_files, get_emotion_label=False)
    print("Pre-processing audio files Complete!")

    print("Building Neural Net")

    emo_model_path = './emo_checkpoint/emodel.h5'
    gender_model_path = './gmodel_checkpoint/gmodel.h5'
    # emo_model = load_model(emo_model_path)
    # # SetBatchNormalizationMomentum(emo_model)
    # test_emo_perf = emo_model.evaluate(x_testcnn, y_emo_test)
    # train_emo_perf = emo_model.evaluate(x_traincnn, y_emo_train)
    # print("Emo Model Train perf is:{}, Test perf is:{}".format(train_emo_perf, test_emo_perf))

    gender_model = load_model(gender_model_path)
    test_gen_perf = gender_model.evaluate(x_testcnn, y_gen_test)
    train_gen_perf = gender_model.evaluate(x_traincnn, y_gen_train)
    print("Gen Model Train perf is:{}, Test perf is:{}".format(train_gen_perf, test_gen_perf))

    one_class_emo = []
    one_class_gen = []

    # # One class only
    # for mfcc_idx in range(x_traincnn.shape[0]):
    #     if y_emo_train[mfcc_idx][5] == 1:
    #         one_class_emo.append(x_traincnn[mfcc_idx])
    #     if y_gen_train[mfcc_idx][0] == 0:
    #         one_class_gen.append(x_traincnn[mfcc_idx])
    #
    # shap_input_np = np.array(one_class_emo[:10])
    # # Model, Data to find Shap values, background, background size
    # shap_values = extract_shap(emo_model, shap_input_np, x_traincnn, 100)
    #
    # plot_shap_values(shap_values, shap_input_np)
    #
    # shap_input_np = np.array(one_class_gen[:10])
    # # Model, Data to find Shap values, background, background size
    # shap_values = extract_shap(gender_model, shap_input_np, x_traincnn, 100)
    #
    # plot_shap_values(shap_values, shap_input_np)

def SetBatchNormalizationMomentum(model, new_value=0.9, prefix='', verbose=False):
  for ii, layer in enumerate(model.layers):
    if hasattr(layer, 'layers'):
      SetBatchNormalizationMomentum(layer, new_value, f'{prefix}Layer {ii}/', verbose)
      continue
    elif isinstance(layer, tf.keras.layers.BatchNormalization):
      if verbose:
        print(f'{prefix}Layer {ii}: name={layer.name} momentum={layer.momentum} --> set momentum={new_value}')
      layer.momentum = new_value

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
