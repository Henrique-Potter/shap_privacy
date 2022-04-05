from pathlib import Path

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from data_processing import pre_process_data
from experiment_neural_nets import build_emo_model_swish
from util.training_engine import train_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model_id = 0


def main():
    audio_files_path = "./NNDatasets/audio/ravdess"
    db_name = 'ravdess'

    emo_model_path = 'emo_checkpoint/emodel_scalarized_{}.h5'.format(db_name)

    print("Pre-processing audio files!")
    x_train_mfcc, x_test_mfcc, y_emo_train_encoded, y_emo_test_encoded, _, _, _, _ = pre_process_data(audio_files_path,
                                                                                                      db_name)
    print("Pre-processing audio files Complete!")

    # Scaling and setting type to float32
    sc = StandardScaler()
    x_train_cnn_scaled = sc.fit_transform(x_train_mfcc)
    x_test_cnn_scaled = sc.transform(x_test_mfcc)

    # x_train_cnn_scaled = np.reshape(x_train_cnn_scaled, x_traincnn.shape)
    # x_test_cnn_scaled = np.reshape(x_test_cnn_scaled, x_testcnn.shape)

    print("Building Neural Net")
    model = build_emo_model_swish(x_train_cnn_scaled)
    model_path = emo_model_path

    y_train = y_emo_train_encoded
    y_test = y_emo_test_encoded

    epochs = 400
    batch_size = 16

    print("Starting model training!")
    if not Path(model_path).exists():
        train_model(model, model_path, batch_size, epochs, x_train_cnn_scaled, y_train, x_test_cnn_scaled, y_test,
                    model_id)

        test_acc = model.evaluate(x_test_cnn_scaled, y_test, batch_size=16)
        train_acc = model.evaluate(x_train_cnn_scaled, y_train, batch_size=16)
        print("Emo Model Train perf is:{}, Test perf is:{}".format(train_acc, test_acc))

        emo_model = tf.keras.models.load_model(emo_model_path)
        test_acc = emo_model.evaluate(x_test_cnn_scaled, y_test, batch_size=128)
        train_acc = emo_model.evaluate(x_train_cnn_scaled, y_train, batch_size=128)
        print("Emo Model Train perf is:{}, Test perf is:{}".format(train_acc, test_acc))
    else:
        print("Check point found. Loading existent Emo Model.")
        # Restore the weights
        pass

    print("Voice Emotion inference model Training complete!")


if __name__ == '__main__':
    main()
