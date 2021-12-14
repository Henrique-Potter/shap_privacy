

def obfuscate_by_gender(shap_values, x_input, y_input_test, noise_str, obs_f):
    # gen_shap_values_norm = normalizeData_0_1(gen_shap_values)
    shap_values[shap_values < 0] = 0

    x_obs_input = x_input.copy()

    print("Parsing Shap values. ")
    for index in range(shap_values.shape[1]):
        #ismale = y_input_test[:][0]
        ismale = y_input_test[:][0][0]

        # Male
        if ismale:
            x_shap_values = shap_values[0][index]
            x_target = x_obs_input[index, :, 0]
            obs_x = obs_f(x_shap_values, x_target, noise_str)
            x_obs_input[index, :, 0] = obs_x

        else:
            pass
            # shap_value = np.squeeze(gen_shap_values[1][index], axis=1)
            # random_noise = sigma * np.random.randn(x_gen_test.shape[1])
            # shap_scaled_noise = shap_value * random_noise
            #
            # x_gen_test[index, :, 0] = x_gen_test[index, :, 0] + shap_scaled_noise

    return x_obs_input


def norm_noise(shap_values, x_target, sigma):

    import numpy as np
    shap_values = np.squeeze(shap_values, axis=1)
    random_noise = sigma * np.random.randn(x_target.shape[0])
    shap_scaled_noise = shap_values * random_noise
    return x_target + shap_scaled_noise
