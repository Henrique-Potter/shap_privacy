import tensorflow as tf
from matplotlib import pyplot as plt


class PlotLosses(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.i = 0
		self.x = []
		self.losses = []
		self.val_losses = []
		self.fig = plt.figure()
		self.logs = []

	def on_epoch_end(self, epoch, logs={}):

		self.logs.append(logs)
		self.x.append(self.i)
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.i += 1

		if epoch % 400 == 0 and epoch != 0:
			plt.plot(self.x, self.losses, label="Train loss")
			plt.plot(self.x, self.val_losses, label="Validation loss")
			plt.legend()
			plt.show()
