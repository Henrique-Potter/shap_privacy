import glob
from pathlib import Path
import tensorflow as tf
from data_processing import pre_process_data
from emonet import  build_gender_model
from util import train_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gender_model_path = './gmodel_checkpoint/gmodel.h5'
tf.random.set_seed(42)


def main():

    audio_files_path = "G:\\NNDatasets\\audio"
    audio_files = glob.glob("{}/**/*.wav".format(audio_files_path), recursive=True)

    print("Pre-processing audio files!")
    x_testcnn, x_traincnn, y_test, y_train = pre_process_data(audio_files, get_emotion_label=False)
    print("Pre-processing audio files Complete!")

    print("Building Neural Net")
    model = build_gender_model()
    model_path = gender_model_path

    print("Starting model training!")
    if not Path(model_path).exists():
        train_model(model, model_path, x_testcnn, x_traincnn, y_test, y_train, get_emotion_label=False)
        test_acc = model.evaluate(x_testcnn, y_test, batch_size=128)
        train_acc = model.evaluate(x_traincnn, y_train, batch_size=128)
    else:
        print("Check point found. Loading existent Gen Model.")
        # Restore the weights
        model = tf.keras.models.load_model(model_path)
        test_acc = model.evaluate(x_testcnn, y_test, batch_size=128)
        train_acc = model.evaluate(x_traincnn, y_train, batch_size=128)


if __name__ == '__main__':
    main()
