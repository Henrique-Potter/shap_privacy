import scipy


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


def obfuscate_by_topk_class(priv_shap_data, util_shap_data, x_input, priv_target_mdl,
                            util_target_mdl, curr_y_gt_labels, obf_intensity, **kwargs):
    import numpy as np

    priv_target_y_input = priv_target_mdl['ground_truth']
    util_target_y_input = util_target_mdl['ground_truth']
    util_nr_classes = len(util_target_y_input[0])

    class_index = priv_target_mdl['priv_class']
    force_y_match = kwargs['force_y_match']
    utility_prot_mode = kwargs['protec_util']
    # top k features to privatize
    k = kwargs['k']
    # top p features to protect
    p = kwargs['p']

    priv_class_shap = priv_shap_data[class_index]
    nr_features = priv_class_shap.shape[1]
    target_class_size = priv_class_shap.shape[0]
    class_x_input = np.ndarray(shape=(target_class_size, nr_features, 1), dtype=float)

    x_data_y_match = []
    local_class_index = 0

    util_class_index_tracker = [0 for x in range(util_nr_classes)]

    for index in range(x_input.shape[0]):
        is_target_class_data = priv_target_y_input[index, class_index]

        util_class_index = np.argmax(util_target_y_input[index])
        util_shap_index = util_class_index_tracker[util_class_index]
        util_gt_shaps = util_shap_data[util_class_index][util_shap_index]
        util_class_index_tracker[util_class_index] += 1

        if is_target_class_data:

            util_shap_sorted_indexes = np.argsort(util_gt_shaps)

            priv_shap_row = priv_class_shap[local_class_index]
            priv_shap_sorted_indexes = np.argsort(priv_shap_row)
            if k > 0:
                # Get the Top K is positive
                priv_topk_shaps_idx = priv_shap_sorted_indexes[-k:]
                util_topk_shaps_idx = util_shap_sorted_indexes[-p:]

            else:
                # Get the Bottom K if negative
                priv_topk_shaps_idx = priv_shap_sorted_indexes[:-k]
                util_topk_shaps_idx = util_shap_sorted_indexes[:-p]

            if utility_prot_mode == 1 and p > 0:
                util_topk_shaps_idx = np.flip(util_topk_shaps_idx)
                for shap_val in util_topk_shaps_idx:
                    shap_map = priv_topk_shaps_idx == shap_val
                    if np.any(shap_map):
                        shap_map = shap_map == False
                        priv_topk_shaps_idx = priv_topk_shaps_idx[shap_map]

            # Creating mask for the non top k
            mask_array = np.ones(nr_features, dtype=int)
            mask_array[priv_topk_shaps_idx] = 0
            mask_array = mask_array.astype(bool)
            # Setting only the non top k to 0. Creating a Top k shap where all other values are 0.
            priv_shap_row[mask_array] = 0

            x_target = x_input[index, :, 0]
            obs_x = norm_noise(priv_shap_row, x_target, obf_intensity)
            x_input[index, :, 0] = obs_x

            if force_y_match:
                class_x_input[local_class_index, :, 0] = obs_x
                x_data_y_match.append(curr_y_gt_labels[index])

            local_class_index += 1

    if force_y_match:
        x_input = class_x_input
        priv_target_y_input = np.array(x_data_y_match)

    return x_input, priv_target_y_input


def general_obf_topk_class(x_input, priv_target_mdl, util_target_mdl, curr_y_gt_labels, obf_intensity, **kwargs):
    import numpy as np

    priv_class = priv_target_mdl['priv_class']
    priv_gt_y_labels = priv_target_mdl['ground_truth']
    util_class = util_target_mdl['util_class']
    topk_size = kwargs['k']
    topp_size = kwargs['p']
    feature_size = x_input.shape[1]

    pmodel = priv_target_mdl
    pmodel_shap_list = pmodel['shap_values']
    pclass_shap_list = pmodel_shap_list[priv_class]
    pclass_shap_mean = np.mean(pclass_shap_list, axis=0)
    pclass_shap_mean_abs = np.mean(np.abs(pclass_shap_list), axis=0)
    p_shap_mean_sorted_idxs = np.argsort(pclass_shap_mean)
    priv_input = x_input[np.argmax(priv_gt_y_labels, axis=1) == priv_class]

    # Move the features against the direction that should increase its SHAP value
    priv_pear = calculate_correlation(pclass_shap_list, priv_input)
    direction_mask = priv_pear.copy()
    # Negative correlation
    direction_mask[priv_pear < 0] = +1
    # Positive correlation
    direction_mask[priv_pear > 0] = -1

    umodel = util_target_mdl
    umodel_shap_list = umodel['shap_values']
    uclass_shap_list = umodel_shap_list[util_class]
    uclass_shap_mean = np.mean(uclass_shap_list, axis=0)
    u_shap_mean_sorted_idxs = np.argsort(uclass_shap_mean)

    if topk_size > 0:
        # Get the Top K is positive
        priv_feature_mask = p_shap_mean_sorted_idxs[-topk_size:]
        util_feature_mask = u_shap_mean_sorted_idxs[-topp_size:]

    else:
        # Get the Bottom K if negative
        priv_feature_mask = p_shap_mean_sorted_idxs[:-topk_size]
        util_feature_mask = u_shap_mean_sorted_idxs[:-topp_size]
    features_removed = []
    origi_pmask = priv_feature_mask.copy()
    if topp_size > 0:
        # util_topk_shaps_idx = np.flip(util_feature_mask)
        for shap_val in util_feature_mask:
            shap_map = priv_feature_mask == shap_val
            if np.any(shap_map):
                features_removed.append(shap_val)
                shap_map = shap_map == False
                priv_feature_mask = priv_feature_mask[shap_map]
    else:
        util_feature_mask = []

    input_signal_increment = np.zeros(x_input.shape)
    input_signal_increment[:, priv_feature_mask, 0] = x_input[:, priv_feature_mask, 0]
    input_signal_increment = input_signal_increment * 0.01 * obf_intensity

    obf_mask_targets = np.ones(feature_size)

    obf_size = priv_feature_mask.shape[0]
    obf_mask_weights = np.arange(1, 1.1, 0.00255)

    obf_mask_targets[priv_feature_mask] = obf_mask_weights[-obf_size:]
    obf_mask_targets[priv_feature_mask] = obf_mask_targets[priv_feature_mask] * direction_mask[priv_feature_mask]

    obf_increment = np.abs(input_signal_increment[:, :, 0]) * obf_mask_targets

    x_input[:, :, 0] = x_input[:, :, 0] + obf_increment

    return x_input, curr_y_gt_labels


def calculate_correlation(pclass_shap_list, priv_input):
    import numpy as np
    nr_rows = pclass_shap_list.shape[0]
    nr_cols = pclass_shap_list.shape[1]

    pear_corr_matrix = np.zeros(nr_cols)

    for sample_col in range(nr_cols):
        X = priv_input[:, sample_col, 0]
        Y = pclass_shap_list[:, sample_col]
        sample_shap_pcorr = scipy.stats.pearsonr(X, Y)[0]
        sample_shap_spcorr = scipy.stats.spearmanr(X, Y)[0]
        sample_shap_kencorr = scipy.stats.kendalltau(X, Y)[0]
        pear_corr_matrix[sample_col] = sample_shap_spcorr

    return pear_corr_matrix


def plot_linear_gam(X, y):
    import pygam as pg
    from pygam import LinearGAM, s, f, te
    import matplotlib.pyplot as plt

    gam = pg.LinearGAM(f(0) + s(0) + s(0) + s(0) + s(0)).fit(X, y)
    XX = gam.generate_X_grid(term=0, n=500)
    plt.plot(XX, gam.predict(XX), 'r--')
    plt.plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--')
    plt.scatter(X, y, facecolor='gray', edgecolors='none')
    plt.title('95% prediction interval');
    plt.show()


def general_by_class_mask(priv_model_id, util_model_id, priv_class_id, util_class_id, model_list, topk_size, topp_size):
    import numpy as np

    pmodel = model_list[priv_model_id]
    pmodel_shap_list = pmodel['shap_values']
    pclass_shap_list = pmodel_shap_list[priv_class_id]
    pclass_shap_mean = np.mean(pclass_shap_list, axis=0)
    p_shap_mean_sorted_idxs = np.argsort(pclass_shap_mean)

    umodel = model_list[util_model_id]
    umodel_shap_list = umodel['shap_values']
    uclass_shap_list = umodel_shap_list[util_class_id]
    uclass_shap_mean = np.mean(uclass_shap_list, axis=0)
    u_shap_mean_sorted_idxs = np.argsort(uclass_shap_mean)

    if topk_size > 0:
        # Get the Top K is positive
        priv_feature_mask = p_shap_mean_sorted_idxs[-topk_size:]
        util_feature_mask = u_shap_mean_sorted_idxs[-topp_size:]

    else:
        # Get the Bottom K if negative
        priv_feature_mask = p_shap_mean_sorted_idxs[:-topk_size]
        util_feature_mask = u_shap_mean_sorted_idxs[:-topp_size]
    features_removed = []
    origi_pmask = priv_feature_mask.copy()
    if topp_size > 0:
        # util_topk_shaps_idx = np.flip(util_feature_mask)
        for shap_val in util_feature_mask:
            shap_map = priv_feature_mask == shap_val
            if np.any(shap_map):
                features_removed.append(shap_val)
                shap_map = shap_map == False
                priv_feature_mask = priv_feature_mask[shap_map]
    else:
        util_feature_mask = []

    return priv_feature_mask, util_feature_mask, features_removed, origi_pmask


def norm_noise(shap_values, x_target, sigma):
    import numpy as np

    shap_size = np.sum(shap_values>0)
    if shap_size == 0:
        return x_target

    shap_impact_mask = [1 + x / 2000 for x in range(1, 1000, 25)]
    hi_vals_index = np.argsort(shap_values)[-shap_size:]
    # Get the last shap masks in increase order
    shap_values[hi_vals_index] = shap_impact_mask[-shap_size:]
    rand_modulation = np.where(np.random.randint(2, size=x_target.shape[0]) == 0, 0.1, 0.2)
    rand_direction = np.where(np.random.randint(2, size=x_target.shape[0]) == 0, -1, 1)
    random_noise = sigma * rand_modulation
    shap_scaled_noise = shap_values * random_noise * rand_direction
    scaled_input = x_target * shap_scaled_noise

    obfuscated_input = x_target + scaled_input
    temp = np.vstack((x_target, obfuscated_input, shap_values))
    return obfuscated_input
