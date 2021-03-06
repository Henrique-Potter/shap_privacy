import tensorflow as tf

from keras.layers import Conv2D, MaxPool2D, AlphaDropout
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout, Activation
from tensorflow.keras.models import Sequential


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
    # opt = optimizers.RMSprop(learning_rate=0.00005, rho=0.9, epsilon=None, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def build_emo_model2(input_sample):
    input_shape_width = input_sample.shape[1]
    input_shape_channels = input_sample.shape[2]

    model = Sequential()
    model.add(Conv1D(128, 5, padding='same', input_shape=(input_shape_width, input_shape_channels)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=8))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(7))
    model.add(Activation('softmax'))
    opt = optimizers.RMSprop(learning_rate=0.00005, rho=0.9, epsilon=None, decay=0.0)
    # opt = optimizers.Adam(learning_rate=0.00005)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def build_emo_model_relu(input_sample):
    input_shape_width = input_sample.shape[1]
    # input_shape_channels = input_sample.shape[2]

    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape_width,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(96))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    opt = optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def build_emo_model_swish(input_sample):
    input_shape_width = input_sample.shape[1]
    # input_shape_channels = input_sample.shape[2]

    model = Sequential()

    model.add(Dense(192, input_shape=(input_shape_width,), ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(160, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(128, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(96, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(64, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(32, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(7))
    model.add(Activation('softmax'))
    opt = optimizers.Adam(learning_rate=0.0001)

    loss_fn_emo = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fn_emo, optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def build_sv_model_swish(input_sample):
    input_shape_width = input_sample.shape[1]
    # input_shape_channels = input_sample.shape[2]

    model = Sequential()

    model.add(Dense(192, input_shape=(input_shape_width,), ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(160, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(128, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(96, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(64, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(32, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(24))
    model.add(Activation('softmax'))
    opt = optimizers.Adam(learning_rate=0.0001)

    loss_fn_emo = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fn_emo, optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def build_gen_model_swish(input_sample):
    input_shape_width = input_sample.shape[1]
    # input_shape_channels = input_sample.shape[2]

    model = Sequential()

    model.add(Dense(192, input_shape=(input_shape_width,), ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(160, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(128, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(96, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(64, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(32, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.2))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    opt = optimizers.Adam(learning_rate=0.001)

    # loss_fn_emo = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.14)
    loss_fn_emo = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss_fn_emo, optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def build_emo_model_selu(input_sample):
    input_shape_width = input_sample.shape[1]
    # input_shape_channels = input_sample.shape[2]

    model = Sequential()

    model.add(Dense(192, input_shape=(input_shape_width,), kernel_initializer='lecun_normal'))
    model.add(Activation('selu'))
    model.add(AlphaDropout(0.2))

    model.add(Dense(160, ))
    model.add(Activation('selu'))
    model.add(AlphaDropout(0.1))

    model.add(Dense(128, ))
    model.add(Activation('selu'))
    model.add(AlphaDropout(0.1))

    model.add(Dense(96, ))
    model.add(Activation('selu'))

    model.add(Dense(64, ))
    model.add(Activation('selu'))

    model.add(Dense(32, ))
    model.add(Activation('selu'))

    model.add(Dense(7))
    model.add(Activation('softmax'))
    opt = optimizers.Adam(learning_rate=0.001)

    loss_fn_emo = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.14)
    model.compile(loss=loss_fn_emo, optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def build_gender_model2(input_sample):
    input_shape_width = input_sample.shape[1]
    input_shape_channels = input_sample.shape[2]

    model = Sequential()
    model.add(Conv1D(128, 5, padding='same', input_shape=(input_shape_width, input_shape_channels)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=8))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    opt = optimizers.RMSprop(learning_rate=0.00005, rho=0.9, epsilon=None, decay=0.0)
    # opt = optimizers.Adam(learning_rate=0.00005)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def build_fser_emo_model(input_sample):
    input_shape = input_sample.shape

    model = Sequential()

    model.add(Conv2D(filters=8, kernel_size=(5, 5), input_shape=(input_shape[1], input_shape[2], input_shape[3])))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=16, kernel_size=(5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=100, kernel_size=(5, 5), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=200, kernel_size=(2, 2), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(17424))
    model.add(Dense(1024))
    model.add(Dense(500))
    model.add(Dense(7))

    model.add(Activation('softmax'))
    opt = optimizers.Adam(learning_rate=0.0001)
    # opt = optimizers.Adam()
    # opt = optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)
    # opt = optimizers.RMSprop(learning_rate=0.00005, rho=0.9, epsilon=None, decay=0.0)

    # opt = optimizers.SGD(learning_rate=0.001)

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


def get_obfuscation_model_relu():
    model = Sequential()
    model.add(Dense(128, input_shape=(40,), ))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    # model.add(Flatten())
    model.add(Dense(40, ))
    model.add(Activation('relu'))

    print(model.summary())
    return model


def get_obfuscation_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), kernel_regularizer='l2'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(128, kernel_regularizer='l2'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(64, kernel_regularizer='l2'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    # model.add(Flatten())
    model.add(Dense(40, kernel_regularizer='l2'))
    model.add(Activation('linear'))

    print(model.summary())
    return model


def get_obfuscation_model_tanh2(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), kernel_regularizer='l2'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(256, kernel_regularizer='l2'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(256, kernel_regularizer='l2'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(256, kernel_regularizer='l2'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(64, kernel_regularizer='l2'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))

    # model.add(Flatten())
    model.add(Dense(40, kernel_regularizer='l2'))
    model.add(Activation('linear'))

    print(model.summary())
    return model


def get_obfuscation_model_swish(input_shape, output_shape):

    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), ))
    model.add(Activation('swish'))
    model.add(Dropout(0.1))
    model.add(Dense(128, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.1))
    model.add(Dense(64, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.1))
    model.add(Dense(output_shape, ))
    model.add(Activation('linear'))

    print(model.summary())
    return model


def get_obfuscation_model_large_swish(input_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_shape,), ))
    model.add(Activation('swish'))
    model.add(Dropout(0.1))

    model.add(Dense(512, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.1))

    model.add(Dense(256, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.1))

    model.add(Dense(256, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.1))

    model.add(Dense(128, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.1))

    model.add(Dense(64, ))
    model.add(Activation('swish'))
    model.add(Dropout(0.1))

    model.add(Dense(40, ))
    model.add(Activation('linear'))

    print(model.summary())
    return model


def get_obfuscation_model_gelu(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), ))
    model.add(Activation('gelu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, ))
    model.add(Activation('gelu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, ))
    model.add(Activation('gelu'))
    model.add(Dropout(0.1))
    model.add(Dense(40, ))
    model.add(Activation('linear'))

    print(model.summary())
    return model


def get_obfuscation_model_selu(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), kernel_initializer='lecun_normal'))
    model.add(Activation('selu'))
    model.add(AlphaDropout(0.1))
    model.add(Dense(128, kernel_initializer='lecun_normal'))
    model.add(Activation('selu'))
    model.add(AlphaDropout(0.1))
    model.add(Dense(64, kernel_initializer='lecun_normal'))
    model.add(Activation('selu'))
    model.add(AlphaDropout(0.1))

    # model.add(Flatten())
    model.add(Dense(40, kernel_initializer='lecun_normal'))
    model.add(Activation('linear'))

    print(model.summary())
    return model
