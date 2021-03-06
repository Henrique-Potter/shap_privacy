import tensorflow as tf

from data_processing import *
from experiment_neural_nets import build_fser_emo_model
from util.training_engine import train_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

emo_model_path = './emo_checkpoint/fser_mel_emodel.h5'

get_emotion_label = True
tf.random.set_seed(42)


def main():
    audio_files_path = "./NNDatasets/audio/"
    n_mels = 64

    print("Pre-processing audio files!")
    x_traincnn, y_train, x_testcnn, y_test = pre_process_audio_to_mel_data(audio_files_path, n_mels, augment_data=True)
    print("Pre-processing audio files Complete!")
    import numpy as np
    print("Building Neural Net")
    x_traincnn = np.reshape(x_traincnn, (x_traincnn.shape[0], n_mels, n_mels, 1))
    x_testcnn = np.reshape(x_testcnn, (x_testcnn.shape[0], n_mels, n_mels, 1))

    model = build_fser_emo_model(x_traincnn)
    model_path = emo_model_path

    print("Starting model training!")
    if not Path(model_path).exists():
        train_model(model, model_path, 64, 1000, x_traincnn, y_train, x_testcnn, y_test, get_emotion_label)
        train_model(model, model_path, 16, 400, x_traincnn, y_train, x_testcnn, y_test, get_emotion_label)

        test_acc = model.evaluate(x_testcnn, y_test, batch_size=128)
        train_acc = model.evaluate(x_traincnn, y_train, batch_size=128)
        print("Emo Model Train perf is:{}, Test perf is:{}".format(train_acc, test_acc))

        emo_model = tf.keras.models.load_model(emo_model_path)
        test_acc = emo_model.evaluate(x_testcnn, y_test, batch_size=128)
        train_acc = emo_model.evaluate(x_traincnn, y_train, batch_size=128)
        print("Emo Model Train perf is:{}, Test perf is:{}".format(train_acc, test_acc))
    else:
        print("Check point found. Loading existent Emo Model.")
        # Restore the weights
        model = tf.keras.models.load_model(model_path)
        train_model(model, model_path, 64, 20000, x_traincnn, y_train, x_testcnn, y_test, get_emotion_label)

        test_acc = model.evaluate(x_testcnn, y_test, batch_size=128)
        train_acc = model.evaluate(x_traincnn, y_train, batch_size=128)

    print("Voice Emotion inference model Training complete!")


if __name__ == '__main__':
    main()
