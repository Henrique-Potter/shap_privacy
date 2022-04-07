import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from TrainingPlot import PlotLosses
from util.PerClassMetrics import PerClassMetrics


def train_model(model, model_path, batch, epoch, x_traincnn, y_train, x_testcnn, y_test, model_id):
    cl_backs = [PlotLosses(model_path, model_id), PerClassMetrics(model, (x_testcnn, y_test), 64, int(model_id))]
    cnnhistory = model.fit(x_traincnn, y_train, batch_size=batch, epochs=epoch, validation_data=(x_testcnn, y_test),
                           callbacks=cl_backs)
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
def train_step(model, feature_map, priv_emo_model, priv_gen_model, util_mdl, x_input, masked_x_input, y_util, y_e_priv, y_g_priv,
               util_loss_fn, priv_e_loss_fn, priv_g_loss_fn, lambd):

    with tf.GradientTape() as tape:
        # cls_targets = tf.constant([2, 3])
        # cls_size = cls_targets.shape[0]
        #
        # batch_sz = x_input.shape[0]
        # feature_sz = x_input.shape[1]
        # nr_priv_classes = y_priv.shape[1]
        model_mask = model(masked_x_input, training=True)
        tape.watch(model_mask)
        v_copy2 = tf.identity(x_input)
        # paddings = tf.constant([[0, 0], [0, 40 - model_mask.shape[1]]])
        # final_mask = tf.pad(model_mask, paddings)

        output_list = []
        feature_index = 0
        for index in range(x_input.shape[1]):
            tensor_cl = x_input[:, index]
            equal = tf.math.equal(index, feature_map)
            contains = tf.reduce_any(equal)
            if contains:
                tensor_cl = tensor_cl + model_mask[:, feature_index]
                feature_index += 1

            output_list.append(tensor_cl)
        obfuscated_input = tf.stack(output_list, axis=1)
        # Applying the mask to the input
        # obfuscated_input = model_mask + x_input
        # obfuscated_input = model_mask

        # Calculating emotion loss
        epriv_mdl_logits = priv_emo_model(obfuscated_input, training=False)
        # priv_mdl_true_loss = priv_mdl_loss_fn(tf.cast(y_priv_mdl, tf.float64), tf.cast(priv_mdl_logits, tf.float64))
        # y_priv should already be masked by class
        # peloss = (1-lambd)/6 * 5 * priv_e_loss_fn(y_e_priv, priv_mdl_logits)
        peloss = 2 * priv_e_loss_fn(y_e_priv, epriv_mdl_logits)

        # Calculating gen loss
        gpriv_mdl_logits = priv_gen_model(obfuscated_input, training=False)
        # pgloss = (1-lambd)/6 * priv_g_loss_fn(y_g_priv, priv_mdl_logits)
        pgloss = 1 * priv_g_loss_fn(y_g_priv, gpriv_mdl_logits)

        util_mdl_logits = util_mdl(obfuscated_input, training=False)
        # uloss = -1*tf.math.pow(0.25, util_loss_fn(y_util, util_mdl_logits))+1
        # uloss = lambd * util_loss_fn(y_util, util_mdl_logits)
        uloss = 10 * util_loss_fn(y_util, util_mdl_logits)

        tloss = peloss + pgloss + uloss

    gradients = tape.gradient(tloss, model.trainable_variables)

    return tloss, peloss, pgloss, uloss, gradients, epriv_mdl_logits


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
