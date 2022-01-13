

def obfuscate_by_class(priv_shap_data, util_shap_data, x_input, y_input_test, obf_intensity, **kwargs):
    import numpy as np

    class_index = kwargs['class_index']
    priv_class_shap = priv_shap_data[class_index]
    negative_shap_mask = priv_class_shap < 0
    priv_class_shap[negative_shap_mask] = 0

    x_obs_input = x_input.copy()

    print("Parsing Shap values.")
    for index in range(x_input.shape[0]):
        # This indexing is expected to match priv_shap data order.
        class_value = y_input_test[index, class_index]

        if class_value:
            x_shap_values = priv_class_shap[index]
            x_target = x_obs_input[index, :, 0]
            obs_x = norm_noise(x_shap_values, x_target, obf_intensity)
            x_obs_input[index, :, 0] = obs_x

    return x_obs_input


def obfuscate_only_by_top_k(priv_shap_data, util_shap_data, x_input, y_input_test, obf_intensity, **kwargs):
    class_index = kwargs['class_index']
    shap_values[shap_values < 0] = 0

    x_obs_input = x_input.copy()

    print("Parsing Shap values.")
    for index in range(shap_values.shape[1]):
        class_value = y_input_test[index, class_index]

        if class_value:
            x_shap_values = shap_values[class_index][index]
            x_target = x_obs_input[index, :, 0]
            obs_x = norm_noise(x_shap_values, x_target, obf_intensity)
            x_obs_input[index, :, 0] = obs_x

    return x_obs_input


def norm_noise(shap_values, x_target, sigma):

    import numpy as np
    shap_values = np.squeeze(shap_values, axis=1)
    random_noise = sigma * np.random.randn(x_target.shape[0])
    shap_scaled_noise = shap_values * random_noise
    return x_target + shap_scaled_noise
