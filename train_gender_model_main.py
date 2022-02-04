from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from data_processing import pre_process_data
from experiment_neural_nets import build_gender_model2, build_gen_model_swish
from util.training_engine import train_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gender_model_path = './gmodel_checkpoint/gmodel_scaled_16.h5'

e_label = False


def main():

    audio_files_path = "./NNDatasets/audio"

    print("Pre-processing audio files!")
    x_traincnn, y_train, x_testcnn, y_test = pre_process_data(audio_files_path, get_emotion_label=e_label, augment_data=True)
    print("Pre-processing audio files Complete!")

    # Scaling and setting type to float32
    sc = StandardScaler()
    x_train_cnn_scaled = sc.fit_transform(x_traincnn)
    x_test_cnn_scaled = sc.transform(x_testcnn)

    print("Building Neural Net")
    model = build_gen_model_swish(x_traincnn)
    model_path = gender_model_path

    epochs = 400
    batch_size = 16

    print("Starting model training!")
    if not Path(model_path).exists():
        train_model(model, model_path, batch_size, epochs, x_train_cnn_scaled, y_train, x_test_cnn_scaled, y_test, e_label)

        test_acc = model.evaluate(x_testcnn, y_test, batch_size=128)
        train_acc = model.evaluate(x_traincnn, y_train, batch_size=128)
    else:
        print("Check point found. Loading existent Gen Model.")
        # Restore the weights
        model = tf.keras.models.load_model(model_path)
        # train_model(model, model_path, 8, x_traincnn, y_train, x_testcnn, y_test, get_emotion_label)
        test_acc = model.evaluate(x_test_cnn_scaled, y_test, batch_size=128)
        train_acc = model.evaluate(x_train_cnn_scaled, y_train, batch_size=128)

    print("Voice Gender inference model Training complete!")


if __name__ == '__main__':
    main()
