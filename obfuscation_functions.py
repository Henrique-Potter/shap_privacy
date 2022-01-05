

# Male index is 0. The value will be 1 if male. Female is index 1.
def obfuscate_by_class(shap_values, x_input, y_input_test, obf_intensity, **kwargs):
    class_index = kwargs['class_index']
    shap_values[shap_values < 0] = 0

    x_obs_input = x_input.copy()

    print("Parsing Shap values. ")
    for index in range(shap_values.shape[1]):
        # Male index is 0. The value will be 1 if male.
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
