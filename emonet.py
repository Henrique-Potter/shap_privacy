import tensorflow as tf

# from keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D


def build_emo_model():

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
	model.add(Dense(10))
	model.add(Activation('softmax'))
	opt = optimizers.Adam(learning_rate=0.000005)
	# opt = optimizers.Adam()
	# opt = optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)

	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	# model.summary()

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

