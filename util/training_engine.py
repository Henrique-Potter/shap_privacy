import matplotlib.pyplot as plt
import numpy as np

from TrainingPlot import PlotLosses
from util.PerClassMetrics import PerClassMetrics
import tensorflow as tf


def train_model(model, model_path, batch, epoch, x_traincnn, y_train, x_testcnn, y_test, get_emotion_label):
    cl_backs = [PlotLosses(model_path, get_emotion_label), PerClassMetrics(model, (x_testcnn, y_test), 64, int(get_emotion_label))]
    cnnhistory = model.fit(x_traincnn, y_train, batch_size=batch, epochs=epoch, validation_data=(x_testcnn, y_test), callbacks=cl_backs)
    # Save the weights
    model.save(model_path)
    figure, axis = plt.subplots(2)
    axis[0].plot(cnnhistory.history['loss'])
    axis[0].plot(cnnhistory.history['val_loss'])
    axis[0].set_title('Loss')
    axis[0].set_ylabel('loss')
    axis[0].set_xlabel('epoch')
    axis[0].legend(['train', 'test'], loc='upper left')
    axis[1].plot(cnnhistory.history['accuracy'])
    axis[1].plot(cnnhistory.history['val_accuracy'])
    axis[1].set_title('Accuracy')
    axis[1].set_ylabel('accuracy')
    axis[1].set_xlabel('epoch')
    axis[1].legend(['train', 'test'], loc='upper left')
    plt.subplots_adjust(hspace=0.7)
    plt.show()


@tf.function
def train_step(model, priv_mdl, util_mdl, x_input, y_priv_mdl, y_util_mdl, optimizer, priv_mdl_loss_fn, util_mdl_loss_fn, lambd, mdl_tgt_id):

    with tf.GradientTape() as tape:

        batch_sz = x_input.shape[0]
        feature_sz = x_input.shape[1]
        nr_priv_classes = y_priv_mdl.shape[1]
        model_mask = model(x_input, training=True)

        paddings = tf.constant([[0, 0], [0, feature_sz - model_mask.shape[1]]])
        final_mask = tf.pad(model_mask, paddings)

        # Applying the mask to the input
        obfuscated_input = final_mask + x_input

        # Calculating loss
        priv_mdl_logits = priv_mdl(obfuscated_input, training=False)
        priv_mdl_w_loss = util_mdl_loss_fn(y_priv_mdl, priv_mdl_logits)
        # true_wrong_loss = priv_mdl_loss_fn(tf.cast(wrong_y, tf.float64), tf.cast(priv_mdl_logits, tf.float64))
        # priv_mdl_true_loss = priv_mdl_loss_fn(tf.cast(y_priv_mdl, tf.float64), tf.cast(priv_mdl_logits, tf.float64))

        util_mdl_logits = util_mdl(obfuscated_input, training=False)
        # umanual_logit_smoothing = tf.fill(y_util_mdl.shape, 0.00001)
        # util_mdl_logits2 = tf.add(util_mdl_logits, umanual_logit_smoothing)

        util_mdl_loss = util_mdl_loss_fn(y_util_mdl, util_mdl_logits)
        # util_mdl_loss2 = util_mdl_loss_fn(y_util_mdl, util_mdl_logits2)

        tape.watch(model_mask)

        if mdl_tgt_id:
            wrong_y = tf.fill(y_priv_mdl.shape, 0.5)
            # priv_mdl_w_loss = priv_mdl_loss_fn(wrong_y, priv_mdl_logits)

            # manual_logit_smoothing = tf.fill(y_priv_mdl.shape, 0.00001)
            # priv_mdl_logits2 = tf.add(priv_mdl_logits, manual_logit_smoothing)
            priv_mdl_w_loss = priv_mdl_loss_fn(wrong_y, priv_mdl_logits)
        else:
            wrong_y = tf.fill(y_util_mdl.shape, 1/nr_priv_classes)
            util_mdl_loss = util_mdl_loss_fn(wrong_y, util_mdl_logits)
            # loss = lambd * priv_mdl_loss - (1 - lambd) * util_mdl_loss

        final_loss = lambd * util_mdl_loss + (1-lambd) * priv_mdl_w_loss
        loss = final_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def estimate_logits_from_loss(batch_size, nr_priv_classes, priv_mdl_loss_fn, priv_mdl_true_loss, wrong_y):

    cross_entropy = tf.divide(priv_mdl_true_loss, batch_size)
    highest_logit_value = tf.pow(2, -tf.cast(cross_entropy, tf.float64))
    left_over = 1 - highest_logit_value
    other_logits_val = left_over / nr_priv_classes
    np_estimanted_logits = np.full((nr_priv_classes), other_logits_val.numpy())
    random_chance_vector = tf.fill(nr_priv_classes, 1 / nr_priv_classes)
    np_estimanted_logits[0] = 1.0
    estimante_model_logit = tf.constant(np_estimanted_logits, dtype=tf.float64)
    estimated_wrong_loss = priv_mdl_loss_fn(wrong_y, estimante_model_logit)

    return estimated_wrong_loss


def triplet_loss(alpha, lambd, priv_mdl_loss, priv_mdla_loss, util_mdl_anchor_loss, util_mdl_loss):
    pos = tf.sqrt(tf.norm(util_mdl_anchor_loss - util_mdl_loss))
    # - (1 - lambd) * priv_mdl_loss
    neg = tf.sqrt(tf.norm(priv_mdla_loss - priv_mdl_loss))
    final_loss = lambd * pos - (1 - lambd) * neg + alpha
    return final_loss


def loss_function(lambd, loss, mdl_tgt_id, priv_mdl_loss, util_mdl_loss):
    if mdl_tgt_id:
        loss = lambd * util_mdl_loss - (1 - lambd) * priv_mdl_loss
        # print((1-lambd) * priv_mdl_loss)
        # print(lambd * util_mdl_loss)
    else:
        loss = lambd * priv_mdl_loss - (1 - lambd) * util_mdl_loss
    return loss