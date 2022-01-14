

def obfuscate_by_class(priv_shap_data, util_shap_data, x_input, y_input_test, obf_intensity, **kwargs):

    class_index = kwargs['class_index']
    priv_class_shap = priv_shap_data[class_index]
    negative_shap_mask = priv_class_shap < 0
    priv_class_shap[negative_shap_mask] = 0

    local_class_index = 0
    print("Parsing Shap values.")
    for index in range(x_input.shape[0]):
        # This indexing is expected to match priv_shap data order.
        class_value = y_input_test[index, class_index]

        if class_value:
            x_shap_values = priv_class_shap[local_class_index]
            x_target = x_input[index, :, 0]
            obs_x = norm_noise(x_shap_values, x_target, obf_intensity)
            x_input[index, :, 0] = obs_x
            local_class_index += 1

    return x_input


def obfuscate_by_topk_class(priv_shap_data, util_shap_data, x_input, y_input_test, obf_intensity, **kwargs):
    import numpy as np

    class_index = kwargs['class_index']
    # top k number
    k = kwargs['k']
    priv_class_shap = priv_shap_data[class_index]
    nr_features = priv_class_shap[0].shape[0]

    local_class_index = 0
    print("Parsing Shap values.")
    for index in range(x_input.shape[0]):
        class_value = y_input_test[index, class_index]

        if class_value:
            priv_shap_row = priv_class_shap[local_class_index]
            shap_sorted_indexes = np.argsort(priv_shap_row)
            topk_shaps = shap_sorted_indexes[-k:]
            # Creating mask for the non top k
            mask_array = np.ones(nr_features, dtype=int)
            mask_array[topk_shaps] = 0
            mask_array = mask_array.astype(bool)
            # Setting only the non top k to 0. Creating a Top k shap where all other values are 0.
            priv_shap_row[mask_array] = 0

            x_target = x_input[index, :, 0]
            obs_x = norm_noise(priv_shap_row, x_target, obf_intensity)
            x_input[index, :, 0] = obs_x

            local_class_index += 1

    return x_input


def norm_noise(shap_values, x_target, sigma):

    import numpy as np
    random_noise = sigma * np.random.randn(x_target.shape[0])
    shap_scaled_noise = shap_values * random_noise
    return x_target + shap_scaled_noise
