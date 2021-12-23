import tensorflow as tf

# from keras import Sequential
from keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D


def build_emo_model(input_sample):

	input_shape_width = input_sample.shape[1]
	input_shape_channels = input_sample.shape[2]

	model = Sequential()

	model.add(Dense(128, input_shape=(input_shape_width, input_shape_channels)))

	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Conv1D(1024, 3, padding='same', ))
	model.add(Activation('relu'))

	model.add(Conv1D(128, 5, padding='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(MaxPooling1D(pool_size=8))

	model.add(Conv1D(128, 5, padding='same', ))
	model.add(Activation('relu'))

	model.add(Conv1D(256, 1, padding='same', ))
	model.add(Activation('relu'))

	model.add(Conv1D(128, 5, padding='same', ))
	model.add(Activation('relu'))
	model.add(Flatten())

	model.add(Dense(17424))
	model.add(Dense(1024))
	model.add(Dense(500))

	model.add(Dense(7))
	model.add(Activation('softmax'))
	opt = optimizers.Adam(learning_rate=0.00005)
	#opt = optimizers.RMSprop(learning_rate=0.00005, rho=0.9, epsilon=None, decay=0.0)

	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model


def build_emo_model2(input_sample):

	input_shape_width = input_sample.shape[1]
	input_shape_channels = input_sample.shape[2]

	model = Sequential()
	model.add(Dense(1024))
	model.add(Conv1D(128, 5, padding='same', input_shape=(input_shape_width, input_shape_channels)))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	model.add(MaxPooling1D(pool_size=8))
	model.add(Conv1D(128, 5, padding='same', ))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	model.add(Dense(1024))
	model.add(Flatten())
	model.add(Dense(7))
	model.add(Activation('softmax'))
	opt = optimizers.RMSprop(learning_rate=0.00005, rho=0.9, epsilon=None, decay=0.0)
	# opt = optimizers.Adam(learning_rate=0.00005)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model


def build_fser_emo_model(input_sample):

	input_shape = input_sample.shape

	model = Sequential()

	model.add(Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1),  input_shape=(input_shape[1], input_shape[2], input_shape[3])))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.summary()

	model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1)))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.summary()

	model.add(Conv2D(filters=100, kernel_size=(5, 5), strides=(1, 1)))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.summary()

	model.add(Conv2D(filters=200, kernel_size=(2, 2), strides=(1, 1)))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(17424))
	model.add(Dense(1024))
	model.add(Dense(500))
	model.add(Dense(7))

	model.summary()

	model.add(Activation('softmax'))
	opt = optimizers.Adam(learning_rate=0.00001)
	# opt = optimizers.Adam()
	# opt = optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)
	# opt = optimizers.RMSprop(learning_rate=0.00005, rho=0.9, epsilon=None, decay=0.0)

	#opt = optimizers.SGD(learning_rate=0.001)

	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()

	return model


def build_gender_model():

	model = Sequential()

	model.add(Dense(259), )
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Conv1D(1024, 3, padding='same', ))
	model.add(Activation('relu'))

	model.add(Conv1D(128, 5, padding='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(MaxPooling1D(pool_size=8))

	model.add(Conv1D(128, 5, padding='same', ))
	model.add(Activation('relu'))

	# #
	# model.add(Conv1D(128, 5, padding='same',))
	# model.add(Activation('relu'))
	# model.add(Conv1D(128, 5, padding='same',))
	# model.add(Activation('relu'))
	# model.add(Dropout(0.2))
	# #
	model.add(Conv1D(256, 1, padding='same', ))
	model.add(Activation('relu'))

	model.add(Conv1D(128, 5, padding='same', ))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(2))
	model.add(Activation('softmax'))
	opt = optimizers.Adam(learning_rate=0.000005)
	# opt = optimizers.Adam()
	# opt = optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)

	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	# model.summary()

	return model

