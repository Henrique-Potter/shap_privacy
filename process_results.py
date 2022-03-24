
# datasets
import glob
import numpy as np


def main():
    experiment_data = "./data/nn_obfuscator_perf/sv_privacy"

    emo_data_paths = glob.glob("{}/**/priv_emo*.npy".format(experiment_data), recursive=True)
    gen_data_paths = glob.glob("{}/**/priv_gen*.npy".format(experiment_data), recursive=True)
    sv_data_paths = glob.glob("{}/**/util*.npy".format(experiment_data), recursive=True)

    emo_best_acc_list, emo_best_f1_list = get_best_model_perf(emo_data_paths)
    gen_best_acc_list, gen_best_f1_list = get_best_model_perf(gen_data_paths)
    sv_best_acc_list, sv_best_f1_list = get_best_model_perf(sv_data_paths)

    plot_best_acc_perf_by_k(emo_best_acc_list, gen_best_acc_list, sv_best_acc_list)
    plot_best_f1_perf_by_k(emo_best_f1_list, gen_best_f1_list, sv_best_f1_list)


def plot_best_acc_perf_by_k(emo_best_acc_list, gen_best_acc_list, sv_best_acc_list):

    import numpy as np
    import matplotlib.pyplot as plt
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(emo_best_acc_list))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    # Make the plot
    plt.bar(br1, emo_best_acc_list, color='r', width=barWidth, edgecolor='grey', label='Emotion ACC')
    plt.bar(br2, gen_best_acc_list, color='g', width=barWidth, edgecolor='grey', label='Gender ACC')
    plt.bar(br3, sv_best_acc_list, color='b', width=barWidth, edgecolor='grey', label='SV ACC')
    # Adding Xticks
    plt.xlabel('Private Bottom K removed', fontweight='bold', fontsize=15)
    plt.ylabel('ACC', fontweight='bold', fontsize=15)
    x_plot_labels = [-x for x in range(len(emo_best_acc_list))]
    plt.xticks([r + barWidth for r in range(len(emo_best_acc_list))], x_plot_labels)
    plt.legend()
    plt.show()


def plot_best_f1_perf_by_k(emo_best_f1_list, gen_best_f1_list, sv_best_f1_list):

    import numpy as np
    import matplotlib.pyplot as plt
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(emo_best_f1_list))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    # Make the plot
    plt.bar(br1, emo_best_f1_list, color='r', width=barWidth, edgecolor='grey', label='Emotion F1')
    plt.bar(br2, gen_best_f1_list, color='g', width=barWidth, edgecolor='grey', label='Gender F1')
    plt.bar(br3, sv_best_f1_list, color='b', width=barWidth, edgecolor='grey', label='SV F1')
    # Adding Xticks
    plt.xlabel('Private Bottom K removed', fontweight='bold', fontsize=15)
    plt.ylabel('F1', fontweight='bold', fontsize=15)
    x_plot_labels = [-x for x in range(len(emo_best_f1_list))]
    plt.xticks([r + barWidth for r in range(len(emo_best_f1_list))], x_plot_labels)
    plt.legend()
    plt.show()


def get_best_model_perf(data_paths):
    best_f1_list = []
    best_acc_list = []
    for path in data_paths:
        experiment_data = np.load(path)
        best_acc_list.append(experiment_data[0][-1:][0])
        best_f1_list.append(experiment_data[1][-1:][0])

    return best_acc_list, best_f1_list


if __name__ == "__main__":
    main()
