import tensorflow as tf
from matplotlib import pyplot as plt

emo_model_path = './emo_checkpoint/emodel.h5'
gender_model_path = './gmodel_checkpoint/gmodel.h5'

plot_per_epoch = 200
save_per_epoch = 500


class PlotLosses(tf.keras.callbacks.Callback):

	def __init__(self, get_emotion_label):
		self.get_emotion_label = get_emotion_label

		if get_emotion_label:
			self.model_path = emo_model_path
		else:
			self.model_path = gender_model_path

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
			figure, axis = plt.subplots(2)
			if self.get_emotion_label:
				plt.title("Voice Emotion CNN Model")
			else:
				plt.title("Voice Gender CNN Model")

			axis[0].plot(self.x, self.losses, label="Train loss")
			axis[0].plot(self.x, self.val_losses, label="Validation loss")
			axis[0].set_title('Loss')
			axis[0].set_ylabel('loss')
			axis[0].set_xlabel('epoch')
			axis[0].legend()

			axis[1].plot(self.x, self.accs, label="Train Accuracy")
			axis[1].set_title('Accuracy')
			axis[1].set_ylabel('accuracy')
			axis[1].set_xlabel('epoch')
			axis[1].plot(self.x, self.val_accs, label="Validation Accuracy")
			axis[1].legend()

			plt.subplots_adjust(hspace=0.7)
			plt.show()

		elif epoch % save_per_epoch == 0 and epoch != 0:
			self.model.save(self.model_path)
