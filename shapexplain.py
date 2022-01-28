import time
import os 

import pandas as pd
import shap
import numpy as np
import tensorflow as tf
from pathlib import Path

from blume.table import table
from scipy.cluster.vq import whiten
from tqdm import tqdm
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from data_processing import pre_process_data
from obfuscation_functions import *
from util.custom_functions import replace_outliers_by_std, mean_std_analysis, replace_outliers_by_quartile, \
    calc_confusion_matrix
from shap_experiment import extract_shap, extract_shap_values, parse_shap_values_by_class

from keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout, Activation
from sklearn.preprocessing import StandardScaler

tf.compat.v1.enable_v2_behavior()


def get_obfuscation_model():
    
    model = Sequential()
    model.add(Dense(128, input_shape=(40, ), kernel_regularizer='l2'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(128, kernel_regularizer='l2'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(64, kernel_regularizer='l2'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    
    #model.add(Flatten())
    model.add(Dense(15, kernel_regularizer='l2'))
    model.add(Activation('tanh'))

    print(model.summary())
    return model

@tf.function
def train_step(model, gender_model, emo_model, mask, emo_train_x, emo_train_y, gen_train_y, optimizer, loss_fn_emo, loss_fn_gen):

    with tf.GradientTape() as tape:
        model_mask = model(mask, training=True)#sk
        # model_mask = tf.reshape(model_mask, (emo_train_x.shape[0], 40, 1))
        emo_train_x = tf.cast(emo_train_x, tf.float32)
        
        paddings = tf.constant([[0, 0], [0, 40 - model_mask.shape[1]]])
        final_mask = tf.pad(model_mask, paddings)
        
        #tf.print(final_mask)
        obfuscated_input = final_mask + emo_train_x #model_mask * emo_train_x + (1 - model_mask) * emo_train_x * noise 
        #obfuscated_input = tf.reshape(obfuscated_input, (emo_train_x.shape[0], 40, 1))
        #tf.print(tf.shape(obfuscated_input))
        
        # calculate loss 
        gen_loss_logits = gender_model(obfuscated_input, training=False)
        gen_loss = loss_fn_gen(gen_train_y, gen_loss_logits)

        emo_loss_logits = emo_model(obfuscated_input, training=False)  
        emo_loss = loss_fn_emo(emo_train_y, emo_loss_logits)
        tape.watch(model_mask)
        
        if genderPrivacy:
            loss = (lambd) * emo_loss - (1-lambd) * gen_loss
        else:
            loss = (lambd) * gen_loss - (1-lambd) * emo_loss        
        #loss = -emo_loss + gen_loss tf.math.reduce_mean(tf.math.abs(model_mask)) + 

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def validate_model(model, emo_test_dataset_batch, gen_test_dataset_batch):
    emo_accuracy = tf.keras.metrics.CategoricalAccuracy()
    gen_accuracy = tf.keras.metrics.BinaryAccuracy()

    for (emo_test_x, emo_test_y), (gen_test_x, gen_test_y) in zip(emo_test_dataset_batch, gen_test_dataset_batch):
        emo_test_x = tf.cast(emo_test_x, tf.float32)
        mask = emo_test_x
        # mask = tf.cast(mask, tf.float32)
        #mask = tf.reshape(emo_test_x, (emo_test_x.shape[0], number_features))
        model_mask = model(mask)

        paddings = tf.constant([[0, 0], [0, 40 - model_mask.shape[1]]])
        final_mask = tf.pad(model_mask, paddings)

        #model_mask = tf.reshape(model_mask, (emo_test_x.shape[0], 40, 1))
        obfuscated_input = final_mask + emo_test_x 
        #obfuscated_input = tf.reshape(obfuscated_input, (emo_test_x.shape[0], 40, 1))

        # get results
        preds = emo_model(obfuscated_input, training=False)
        emo_accuracy.update_state(y_true=emo_test_y, y_pred=preds)
        preds = gender_model(obfuscated_input, training=False)
        gen_accuracy.update_state(y_true=gen_test_y, y_pred=preds)

    print(emo_accuracy.result().numpy())
    print(gen_accuracy.result().numpy())


def train_obfuscation_model(model, x_test_emo_cnn_scaled, x_train_emo_cnn_scaled, y_test_emo_encoded, y_test_gen_encoded, y_train_emo_encoded,
        y_train_gen_encoded, obf_gender=True):

    # convert to tensor
    emo_test_dataset_batch, emo_train_dataset_batch, gen_test_dataset_batch, gen_train_dataset_batch = to_batchdataset(
        x_test_emo_cnn_scaled, x_train_emo_cnn_scaled, y_test_emo_encoded, y_test_gen_encoded, y_train_emo_encoded,
        y_train_gen_encoded)

    #optimizer = tf.keras.optimizers.Adam()
    #optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.01)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0, decay=6e-8)

    loss_fn_emo = tf.keras.losses.CategoricalCrossentropy()
    loss_fn_gen = tf.keras.losses.BinaryCrossentropy()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    with tf.device('gpu:0'):
        for e in tqdm(range(epochs)):    
            for (emo_train_x, emo_train_y), (gen_train_x, gen_train_y) in zip(emo_train_dataset_batch, gen_train_dataset_batch):
                
                mask = emo_train_x#*-1
                mask = tf.cast(mask, tf.float32)
                #mask = tf.reshape(emo_train_x, (emo_train_x.shape[0], number_features))
                
                for i in range(max_iter):
                    loss = train_step(model, gender_model, emo_model, mask, emo_train_x, emo_train_y, gen_train_y, optimizer, loss_fn_emo, loss_fn_gen)
                train_loss(loss)
            
            tf.print(train_loss.result())
            train_loss.reset_states()
            
            if e % 5 == 0:
                obf_mask = model.predict(x_test_emo_cnn_scaled)
                validate_model(model, emo_test_dataset_batch, gen_test_dataset_batch)
                calc_confusion_matrix(emo_model, x_test_emo_cnn_scaled, y_test_emo_encoded)
                calc_confusion_matrix(gender_model, x_test_emo_cnn_scaled, y_test_gen_encoded)
                model.save(obf_model_path)
                # test 
                # mask = emo_train_x#*-1
                # mask = tf.reshape(mask, (emo_train_x.shape[0], number_features))
                # obfuscated_input = model(mask)
                # obfuscated_input = tf.reshape(obfuscated_input, (emo_train_x.shape[0], 40, 1))
                # res = emo_model.evaluate(obfuscated_input, emo_train_y)
                # print("epoch:", e, " - ", res)
                # res = gender_model.evaluate(obfuscated_input, gen_train_y)
                # print(res)

    return model


def main():
    
    # get dataset
    print("Pre-processing audio files!")
    x_train_emo_cnn, y_train_emo_encoded, x_test_emo_cnn, y_test_emo_encoded = pre_process_data(audio_files_path, get_emotion_label=True)
    x_train_gen_cnn, y_train_gen_encoded, x_test_gen_cnn, y_test_gen_encoded = pre_process_data(audio_files_path, get_emotion_label=False)
    print("Pre-processing audio files Complete!")

    sc = StandardScaler()
    # Squeeze extra dimension
    x_train_emo_cnn = np.reshape(x_train_emo_cnn, (x_train_emo_cnn.shape[0], x_train_emo_cnn.shape[1]))
    x_test_emo_cnn = np.reshape(x_test_emo_cnn, (x_test_emo_cnn.shape[0], x_test_emo_cnn.shape[1]))

    x_train_emo_cnn_scaled = sc.fit_transform(x_train_emo_cnn)
    x_test_emo_cnn_scaled = sc.transform(x_test_emo_cnn)

    # x_train_emo_cnn = sc.fit_transform(x_train_emo_cnn)
    # x_test_emo_cnn = sc.transform(x_test_emo_cnn)

    #if os.path.exists(model_path):
    #    print("Loading existing model.", model_path)
    #    model = tf.keras.models.load_model(model_path)
    #else:
    model = get_obfuscation_model()
    # exit()
    if not Path(obf_model_path).exists():
        model = train_obfuscation_model(model,
        x_test_emo_cnn_scaled, x_train_emo_cnn_scaled, y_test_emo_encoded, y_test_gen_encoded, y_train_emo_encoded,
        y_train_gen_encoded)
    else:
        model = tf.keras.models.load_model(obf_model_path)

    print("saving obfuscation model:", model_path)

    model.save(model_path)


def to_batchdataset(x_test_emo_cnn_scaled, x_train_emo_cnn_scaled, y_test_emo_encoded, y_test_gen_encoded,
                    y_train_emo_encoded, y_train_gen_encoded):

    emo_train_dataset = tf.data.Dataset.from_tensor_slices((x_train_emo_cnn_scaled, y_train_emo_encoded))
    gen_train_dataset = tf.data.Dataset.from_tensor_slices((x_train_emo_cnn_scaled, y_train_gen_encoded))  # same input
    emo_test_dataset = tf.data.Dataset.from_tensor_slices((x_test_emo_cnn_scaled, y_test_emo_encoded))
    gen_test_dataset = tf.data.Dataset.from_tensor_slices((x_test_emo_cnn_scaled, y_test_gen_encoded))
    emo_train_dataset_batch = emo_train_dataset.batch(batch_size)
    gen_train_dataset_batch = gen_train_dataset.batch(batch_size)
    emo_test_dataset_batch = emo_test_dataset.batch(batch_size)
    gen_test_dataset_batch = gen_test_dataset.batch(batch_size)

    return emo_test_dataset_batch, emo_train_dataset_batch, gen_test_dataset_batch, gen_train_dataset_batch


if __name__ == "__main__":
    #emo_model_path = './emo_checkpoint/emodel_m2_all_aug_5k_16.h5'
    emo_model_path = "emo_checkpoint/emo_model_simple.h5"
    gender_model_path = "gmodel_checkpoint/gender_model_simple.h5"

    genderPrivacy = True
    # gender_model_path = './gmodel_checkpoint/gmodel_m2_all_aug_5k_16.h5'
    if genderPrivacy:
        model_path = 'emo_checkpoint/model_gender_simple.h5'
    else:
        model_path = 'emo_checkpoint/model_emo_simple.h5'

    obf_model_path = 'obf_checkpoint/model_obf.h5'

    # datasets
    audio_files_path = "./NNDatasets/audio"
    gen_shap_df_path = './data/gen_shap_df.npy'
    emo_shap_df_path = './data/emo_shap_df.npy'

    print("Loading trained Neural Nets")
    gender_model = load_model(gender_model_path)
    emo_model = load_model(emo_model_path)

    batch_size = 4
    epochs = 40
    max_iter = 50  
    number_features = 40
    #metrics
    lambd = .95

    main()
