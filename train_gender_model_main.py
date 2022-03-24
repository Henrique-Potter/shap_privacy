from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from data_processing import pre_process_data
from experiment_neural_nets import build_gen_model_swish
from util.training_engine import train_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

e_label = False


def main():

    audio_files_path = "./NNDatasets/audio/ravdess"
    db_name = 'ravdess'

    gender_model_path = 'gmodel_checkpoint/gmodel_scaled_{}.h5'.format(db_name)

    print("Pre-processing audio files!")
    x_train, x_test, _, _, y_tr_gen, y_te_gen, _, _ = pre_process_data(audio_files_path, db_name)
    print("Pre-processing audio files Complete!")

    # Scaling and setting type to float32
    sc = StandardScaler()
    x_train_cnn_scaled = sc.fit_transform(x_train)
    x_test_cnn_scaled = sc.transform(x_test)

    print("Building Neural Net")
    model = build_gen_model_swish(x_train)
    model_path = gender_model_path

    epochs = 400
    batch_size = 16

    print("Starting model training!")
    if not Path(model_path).exists():
        train_model(model, model_path, batch_size, epochs, x_train_cnn_scaled, y_tr_gen, x_test_cnn_scaled, y_te_gen, e_label)

        test_acc = model.evaluate(x_test, y_te_gen, batch_size=128)
        train_acc = model.evaluate(x_train, y_tr_gen, batch_size=128)
    else:
        print("Check point found. Loading existent Gen Model.")
        # Restore the weights
        pass

    print("Voice Gender inference model Training complete!")


if __name__ == '__main__':
    main()
