from pathlib import Path
import tensorflow as tf
from data_processing import pre_process_data
from experiment_neural_nets import build_emo_model2
from util.training_engine import train_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

emo_model_path = 'emo_checkpoint/emodel_m2.h5'

get_emotion_label = True


def main():

    audio_files_path = "./NNDatasets/audio/"

    print("Pre-processing audio files!")
    x_traincnn, y_train, x_testcnn, y_test = pre_process_data(audio_files_path)
    print("Pre-processing audio files Complete!")

    print("Building Neural Net")
    model = build_emo_model2(x_traincnn)
    model_path = emo_model_path

    epochs = 400
    batch_size = 128

    print("Starting model training!")
    if not Path(model_path).exists():
        train_model(model, model_path, batch_size, epochs, x_traincnn, y_train, x_testcnn, y_test, get_emotion_label)

        test_acc = model.evaluate(x_testcnn, y_test, batch_size=16)
        train_acc = model.evaluate(x_traincnn, y_train, batch_size=16)
        print("Emo Model Train perf is:{}, Test perf is:{}".format(train_acc, test_acc))

        emo_model = tf.keras.models.load_model(emo_model_path)
        test_acc = emo_model.evaluate(x_testcnn, y_test, batch_size=128)
        train_acc = emo_model.evaluate(x_traincnn, y_train, batch_size=128)
        print("Emo Model Train perf is:{}, Test perf is:{}".format(train_acc, test_acc))
    else:
        print("Check point found. Loading existent Emo Model.")
        # Restore the weights
        model = tf.keras.models.load_model(model_path)
        train_model(model, model_path, x_testcnn, x_traincnn, y_test, y_train, get_emotion_label)
        test_acc = model.evaluate(x_testcnn, y_test, batch_size=128)
        train_acc = model.evaluate(x_traincnn, y_train, batch_size=128)

    print("Voice Emotion inference model Training complete!")


if __name__ == '__main__':
    main()
