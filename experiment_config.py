from obfuscation_functions import obfuscate_by_topk_class


def set_experiment_config(emo_model, gender_model, emo_gt_shap_list, gen_gt_shap_list, y_test_emo_encoded, y_test_gen_encoded):

    emo_model_dict = {'model_name': "emotion_model",
                      'model': emo_model,
                      'ground_truth': y_test_emo_encoded,
                      'privacy_target': True,
                      'priv_class': 2,
                      'shap_values': emo_gt_shap_list,
                      'utility_target': False}

    gen_model_dict = {'model_name': "gen_model",
                      'model': gender_model,
                      'ground_truth': y_test_gen_encoded,
                      'privacy_target': False,
                      'shap_values': gen_gt_shap_list,
                      'utility_target': True}

    model_list = [emo_model_dict, gen_model_dict]

    # Building Obfuscation list functions
    # Noise intensity List
    norm_noise_list = [1 + x / 100 for x in range(10, 100, 3)]
    obfuscation_f_list = []
    obf_by_topk_class2 = {'obf_f_handler': obfuscate_by_topk_class,
                          'intensities': norm_noise_list,
                          'kwargs': {'k': 6, 'force_y_match': 1, 'avg_reps': 2, 'protec_util': 1,
                                     'p': 0},
                          'label': 'obf_totk_6_'}

    obf_by_topk_class3 = {'obf_f_handler': obfuscate_by_topk_class,
                          'intensities': norm_noise_list,
                          'kwargs': {'k': 6, 'force_y_match': 1, 'avg_reps': 2, 'protec_util': 1,
                                     'p': 0},
                          'label': 'obf_totk_3_female'}
    obfuscation_f_list.append(obf_by_topk_class2)
    obfuscation_f_list.append(obf_by_topk_class3)
    return model_list, obfuscation_f_list