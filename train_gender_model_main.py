import glob
from pathlib import Path
import tensorflow as tf
from data_processing import pre_process_data
from experiment_neural_nets import build_gender_model2
from util.custom_functions import train_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gender_model_path = './gmodel_checkpoint/gmodel_16.h5'

get_emotion_label = False


def main():

    audio_files_path = "./NNDatasets/audio"

    print("Pre-processing audio files!")
    x_traincnn, y_train, x_testcnn, y_test = pre_process_data(audio_files_path, get_emotion_label, augment_data=True)
    print("Pre-processing audio files Complete!")

    print("Building Neural Net")
    model = build_gender_model2(x_traincnn)
    model_path = gender_model_path

    print("Starting model training!")
    if not Path(model_path).exists():
        train_model(model, model_path, 8, x_traincnn, y_train, x_testcnn, y_test, get_emotion_label)
        test_acc = model.evaluate(x_testcnn, y_test, batch_size=128)
        train_acc = model.evaluate(x_traincnn, y_train, batch_size=128)
    else:
        print("Check point found. Loading existent Gen Model.")
        # Restore the weights
        model = tf.keras.models.load_model(model_path)
        train_model(model, model_path, 8, x_traincnn, y_train, x_testcnn, y_test, get_emotion_label)
        test_acc = model.evaluate(x_testcnn, y_test, batch_size=128)
        train_acc = model.evaluate(x_traincnn, y_train, batch_size=128)

    print("Voice Gender inference model Training complete!")


if __name__ == '__main__':
    main()
