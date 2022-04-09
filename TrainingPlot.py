import tensorflow as tf
from matplotlib import pyplot as plt

plot_per_epoch = 50
save_per_epoch = 90


class PlotLosses(tf.keras.callbacks.Callback):

    def __init__(self, model_path, model_id):
        self.model_id = model_id
        self.model_path = model_path

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.accs = []
        self.val_accs = []

        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        self.accs.append(logs.get('accuracy'))
        self.val_accs.append(logs.get('val_accuracy'))

        self.i += 1

        if epoch % plot_per_epoch == 0 and epoch != 0:

            if self.model_id:
                plt.title("Voice Emotion CNN Model")
            else:
                plt.title("Voice Gender CNN Model")

            figure, axis = plt.subplots(2)
            axis[0].plot(self.x, self.losses, label="Train loss")
            axis[0].plot(self.x, self.val_losses, label="Validation loss")
            axis[0].set_title('Loss')
            axis[0].set_ylabel('loss')
            axis[0].set_xlabel('epoch')
            axis[0].set_yscale('log')
            axis[0].legend()

            axis[1].plot(self.x, self.accs, label="Train Accuracy")
            axis[1].set_title('Accuracy')
            axis[1].set_ylabel('accuracy')
            axis[1].set_xlabel('epoch')
            axis[1].plot(self.x, self.val_accs, label="Validation Accuracy")
            axis[1].legend()

            plt.subplots_adjust(hspace=0.7)
            plt.show()

        if epoch % save_per_epoch == 0 and epoch != 0:
            self.model.save(self.model_path)
