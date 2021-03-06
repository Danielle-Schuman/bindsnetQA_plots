import argparse
import numpy as np
import csv
import time as clock
from typing import List, Optional, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from examples.mnist import supervised_mnist
from bindsnet_qa.examples_qa import supervised_mnist_qa
from bindsnet_bad_ones.examples_bad_ones import supervised_mnist_bad_ones
from bindsnet_bad_zeros.examples_bad_zeros import supervised_mnist_bad_zeros


def write_to_csv(directory: str, heading: str, arguments: list, column_names: Optional[list] = None, content: Optional[List[list]] = None, averages: Optional[list] = None, std: Optional[list] = None, seeds: Optional[list] = None):
    with open((directory + '/' + heading + '.csv'), 'a+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([heading])
        filewriter.writerow(['Parameters:', 'runs', 'n_neurons', 'n_train', 'n_test', 'n_clamp', 'exc', 'inh', 'time', 'dt', 'intensity', 'update_interval', 'train', 'plot', 'num_repeats', 'gpu'])
        parameters = ['']
        parameters.extend(arguments)
        filewriter.writerow(parameters)
        filewriter.writerow([])
        if seeds is not None:
            seeds_with_title = ['Seeds:']
            seeds_with_title.extend(seeds)
            filewriter.writerow(seeds_with_title)
            filewriter.writerow([])
        filewriter.writerow([])
        if column_names is not None:
            filewriter.writerow(['Column names:'])
            filewriter.writerow(column_names)
            filewriter.writerow([])
        if content is not None:
            filewriter.writerows(content)
            filewriter.writerow([])
            if averages is None:
                averages = np.mean(content, axis=0)
            if std is None:
                std = np.std(content, axis=0)
        filewriter.writerow(['Averages:'])
        filewriter.writerow(averages)
        filewriter.writerow(['Standard deviations:'])
        filewriter.writerow(std)
        filewriter.writerow([])
        filewriter.writerow([])
        filewriter.writerow([])


def plot_one_training_accuracy(
    acc_avgs: List[float],
    stds: List[float],
    ax: Axes,
    x: np.ndarray,
    label: str
) -> None:
    y = np.array([0.0] + [a for a in acc_avgs])
    std = np.array([0.0] + [s for s in stds])
    colors = {"b_all": 'tab:blue', "b_proportion": 'tab:orange', "qa_all": 'tab:green', "qa_proportion": 'tab:red', "0_all": 'darkgrey', "0_proportion": 'lightgrey', "1_all": 'dimgrey', "1_proportion": 'grey'}
    ax.plot(x, y, label=label, marker='.', color=colors[label])
    ax.fill_between(x, y - std, y + std, color=colors[label], alpha=0.2)


def plot_training_accuracy(
    acc_avgs: Dict[str, List[float]],
    stds: Dict[str, List[float]],
    update_interval: int,
    directory: str,
    name: str,
    figsize: Tuple[float, float] = (10.5, 6)
) -> None:
    # language=rst
    """
    Plot training accuracy curves.

    :param acc_avgs: Dict with lists of average accuracies
    :param stds: Dict with lists of standard deviation of accuracies
    :param update_interval: Number of examples per accuracy estimate.
    :param directory: Directory where the training accuracy plot will be saved.
    :param name: name for the figure
    :param figsize: Horizontal, vertical figure size in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # does not matter, of which list in acc_avgs we take the length: lengths are all equal
    list_length = len(list(acc_avgs.values())[0])
    x = np.array([0.0] + [(i * update_interval) + update_interval for i in range(list_length)])
    for runtype in acc_avgs:
        plot_one_training_accuracy(acc_avgs[runtype], stds[runtype], ax, x, runtype)

    ax.set_ylim([0, 110])
    end = list_length * update_interval
    ax.set_xlim([0, end])
    ax.set_title("Estimated classification accuracy")
    ax.set_xlabel("No. of examples")
    ax.set_ylabel("Training accuracy in %")
    # to have readable number on x-axis, there can be at most 20 ticks; ticks should be multiples of update_interval
    if list_length > 20:
        xticks = int(list_length / 20) * update_interval
    else:
        xticks = update_interval
    ax.set_xticks(range(0, (end + update_interval), xticks))
    ax.set_yticks(range(0, 110, 10))
    ax.legend()

    file = directory + '/' + name
    fig.savefig(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--n_neurons", type=int, default=100)
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--n_clamp", type=int, default=1)
    parser.add_argument("--exc", type=float, default=22.5)
    parser.add_argument("--inh", type=float, default=22.5)
    parser.add_argument("--time", type=int, default=500)
    parser.add_argument("--dt", type=int, default=1.0)
    parser.add_argument("--intensity", type=float, default=128)
    parser.add_argument("--update_interval", type=int, default=250)
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--test", dest="train", action="store_false")
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--directory", type=str, default=".")
    parser.add_argument("--other_plots_directory", type=str, default=".")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--gpu", dest="gpu", action="store_true")
    parser.set_defaults(plot=False, gpu=False, train=True)

    args = parser.parse_args()

    runs = args.runs
    n_neurons = args.n_neurons
    n_train = args.n_train
    n_test = args.n_test
    n_clamp = args.n_clamp
    exc = args.exc
    inh = args.inh
    time = args.time
    dt = args.dt
    intensity = args.intensity
    num_repeats = args.num_repeats
    update_interval = args.update_interval
    train = args.train
    plot = args.plot
    directory = args.directory
    other_plots_directory = args.other_plots_directory
    gpu = args.gpu
    arguments_list = [runs, n_neurons, n_train, n_test, n_clamp, exc, inh, time, dt, intensity, update_interval, train, plot, num_repeats, gpu]

    # determine total number of neurons in network for evalutation, which percentage of the qubo is filled
    total_n_neurons = 784 + 2 * n_neurons

    seeds = []
    accuracies_b_all = []
    accuracies_b_proportion = []
    accuracies_qa_all = []
    accuracies_qa_proportion = []
    accuracies_z_all = []
    accuracies_z_proportion = []
    accuracies_o_all = []
    accuracies_o_proportion = []
    wallclocktime = []
    qb_solv_energies_for_runs = []
    qb_solv_energies_for_runs_layout = []
    filled_for_runs = []

    for run in range(runs):
        seed = clock.time()
        seeds.append(seed)

        print("BindsNET: Run %d" % run)
        accuracies_b, wallclocktime_b = supervised_mnist.supervised_mnist(seed, n_neurons, n_train, n_test, n_clamp, exc, inh, time, dt, intensity, update_interval, train, plot, other_plots_directory, gpu)
        accuracies_b_all.append(accuracies_b["all"])
        accuracies_b_proportion.append(accuracies_b["proportion"])

        print("BindsNET_QA: Run %d" % run)
        accuracies_qa, wallclocktime_qa, qb_solv_energies, filled = supervised_mnist_qa.supervised_mnist(seed, n_neurons, n_train, n_test, n_clamp, exc, inh, time, dt, intensity, num_repeats, update_interval, train, plot, other_plots_directory, gpu)
        accuracies_qa_all.append(accuracies_qa["all"])
        accuracies_qa_proportion.append(accuracies_qa["proportion"])

        print("BindsNET_Bad_Zeros: Run %d" % run)
        accuracies_z, wallclocktime_z = supervised_mnist_bad_zeros.supervised_mnist(seed, n_neurons, n_train, n_test, n_clamp, exc, inh, time, dt, intensity, update_interval, train, plot, other_plots_directory, gpu)
        accuracies_z_all.append(accuracies_z["all"])
        accuracies_z_proportion.append(accuracies_z["proportion"])

        print("BindsNET_Bad_Ones: Run %d" % run)
        accuracies_o, wallclocktime_o = supervised_mnist_bad_ones.supervised_mnist(seed, n_neurons, n_train, n_test, n_clamp, exc, inh, time, dt, intensity, update_interval, train, plot, other_plots_directory, gpu)
        accuracies_o_all.append(accuracies_o["all"])
        accuracies_o_proportion.append(accuracies_o["proportion"])

        wallclocktime.append([wallclocktime_b, wallclocktime_qa, wallclocktime_z, wallclocktime_o])
        qb_solv_energies_for_runs.extend(qb_solv_energies)
        qb_solv_energies_for_runs_layout.append('')
        qb_solv_energies_for_runs_layout.extend(qb_solv_energies)
        filled_for_runs.extend(filled)

    acc_averages_b_all = np.mean(accuracies_b_all, axis=0)
    acc_averages_b_proportion = np.mean(accuracies_b_proportion, axis=0)
    acc_averages_qa_all = np.mean(accuracies_qa_all, axis=0)
    acc_averages_qa_proportion = np.mean(accuracies_qa_proportion, axis=0)
    acc_averages_z_all = np.mean(accuracies_z_all, axis=0)
    acc_averages_z_proportion = np.mean(accuracies_z_proportion, axis=0)
    acc_averages_o_all = np.mean(accuracies_o_all, axis=0)
    acc_averages_o_proportion = np.mean(accuracies_o_proportion, axis=0)
    acc_averages_diff_all = np.subtract(acc_averages_qa_all, acc_averages_b_all)
    acc_averages_diff_proportion = np.subtract(acc_averages_qa_proportion, acc_averages_b_proportion)

    acc_stds_b_all = np.std(accuracies_b_all, axis=0)
    acc_stds_b_proportion = np.std(accuracies_b_proportion, axis=0)
    acc_stds_qa_all = np.std(accuracies_qa_all, axis=0)
    acc_stds_qa_proportion = np.std(accuracies_qa_proportion, axis=0)
    acc_stds_z_all = np.std(accuracies_z_all, axis=0)
    acc_stds_z_proportion = np.std(accuracies_z_proportion, axis=0)
    acc_stds_o_all = np.std(accuracies_o_all, axis=0)
    acc_stds_o_proportion = np.std(accuracies_o_proportion, axis=0)
    acc_stds_diff_all = np.subtract(acc_stds_qa_all, acc_stds_b_all)
    acc_stds_diff_proportion = np.subtract(acc_stds_qa_proportion, acc_stds_b_proportion)

    qb_solv_averages = np.mean(qb_solv_energies_for_runs, axis=0)
    qb_solv_stds = np.std(qb_solv_energies_for_runs, axis=0)

    filled_array = np.array(filled_for_runs)
    filled_percentage = (filled_array / (sum(range(total_n_neurons)))) * 100
    mean_filled = np.mean(filled_percentage)
    std_filled = np.std(filled_percentage)

    # append average difference to the array
    acc_averages_diff_all = np.append(acc_averages_diff_all, np.array(np.mean(acc_averages_diff_all)))
    acc_averages_diff_proportion = np.append(acc_averages_diff_proportion, np.array(np.mean(acc_averages_diff_proportion)))
    acc_stds_diff_all = np.append(acc_stds_diff_all, np.array(np.mean(acc_stds_diff_all)))
    acc_stds_diff_proportion = np.append(acc_stds_diff_proportion, np.mean(acc_stds_diff_proportion))
    # should all have the same length -> does not matter, which one we use
    diff_column_names = [i for i in range(len(acc_averages_diff_all) -1)]
    diff_column_names.append("Average")

    write_to_csv(directory, "Accuracies BindsNET all", arguments_list, [i for i in range(len(accuracies_b_all[0]))], accuracies_b_all, acc_averages_b_all, acc_stds_b_all, seeds)
    write_to_csv(directory, "Accuracies BindsNET proportion", arguments_list, [i for i in range(len(accuracies_b_proportion[0]))], accuracies_b_proportion, acc_averages_b_proportion, acc_stds_b_proportion, seeds)
    write_to_csv(directory, "Accuracies BindsNET_QA all", arguments_list, [i for i in range(len(accuracies_qa_all[0]))], accuracies_qa_all, acc_averages_qa_all, acc_stds_qa_all, seeds)
    write_to_csv(directory, "Accuracies BindsNET_QA proportion", arguments_list, [i for i in range(len(accuracies_qa_proportion[0]))], accuracies_qa_proportion, acc_averages_qa_proportion, acc_stds_qa_proportion, seeds)
    write_to_csv(directory, "Accuracies BindsNET_bad_zero all", arguments_list, [i for i in range(len(accuracies_z_all[0]))],
                 accuracies_z_all, acc_averages_z_all, acc_stds_z_all, seeds)
    write_to_csv(directory, "Accuracies BindsNET_bad_zero proportion", arguments_list,
                 [i for i in range(len(accuracies_z_proportion[0]))], accuracies_z_proportion,
                 acc_averages_z_proportion, acc_stds_z_proportion, seeds)
    write_to_csv(directory, "Accuracies BindsNET_bad_one all", arguments_list, [i for i in range(len(accuracies_o_all[0]))],
                 accuracies_o_all, acc_averages_o_all, acc_stds_o_all, seeds)
    write_to_csv(directory, "Accuracies BindsNET_bad_one proportion", arguments_list,
                 [i for i in range(len(accuracies_o_proportion[0]))], accuracies_o_proportion,
                 acc_averages_o_proportion, acc_stds_o_proportion, seeds)
    write_to_csv(directory, "Wall clock time taken", arguments_list, ["BindsNet (in sec)", "BindsNET_QA (in sec)", "BindsNET_Bad_Zero (in sec)", "BindsNET_Bad_Ones (in sec)"], wallclocktime, np.median(wallclocktime, axis=0), None, seeds)
    write_to_csv(directory, "Qb_solv_energies", arguments_list, [i for i in range(int(time / dt))], qb_solv_energies_for_runs_layout, qb_solv_averages, qb_solv_stds, seeds)
    write_to_csv(directory, "Differences all-accuracies", arguments_list, diff_column_names, None, acc_averages_diff_all, acc_stds_diff_all)
    write_to_csv(directory, "Differences proportion-accuracies", arguments_list, diff_column_names, None, acc_averages_diff_proportion, acc_stds_diff_proportion)
    write_to_csv(directory, "Filled Percentage of QUBO", arguments_list, None, None, [mean_filled], [std_filled], seeds)

    acc_averages = {"b_all": acc_averages_b_all, "b_proportion": acc_averages_b_proportion, "qa_all": acc_averages_qa_all, "qa_proportion": acc_averages_qa_proportion, "0_all": acc_averages_z_all, "0_proportion": acc_averages_z_proportion, "1_all": acc_averages_o_all, "1_proportion": acc_averages_o_proportion}
    acc_stds = {"b_all": acc_stds_b_all, "b_proportion": acc_stds_b_proportion, "qa_all": acc_stds_qa_all, "qa_proportion": acc_stds_qa_proportion, "0_all": acc_stds_z_all, "0_proportion": acc_stds_z_proportion, "1_all": acc_stds_o_all, "1_proportion": acc_stds_o_proportion}
    acc_averages_all = {"b_all": acc_averages_b_all, "qa_all": acc_averages_qa_all, "0_all": acc_averages_z_all, "1_all": acc_averages_o_all}
    acc_stds_all = {"b_all": acc_stds_b_all, "qa_all": acc_stds_qa_all, "0_all": acc_stds_z_all, "1_all": acc_stds_o_all}
    acc_averages_proportion = {"b_proportion": acc_averages_b_proportion, "qa_proportion": acc_averages_qa_proportion, "0_proportion": acc_averages_z_proportion, "1_proportion": acc_averages_o_proportion}
    acc_stds_proportion = {"b_proportion": acc_stds_b_proportion, "qa_proportion": acc_stds_qa_proportion, "0_proportion": acc_stds_z_proportion, "1_proportion": acc_stds_o_proportion}

    plot_training_accuracy(acc_averages, acc_stds, update_interval, directory, "training_accuracy")
    plot_training_accuracy(acc_averages_all, acc_stds_all, update_interval, directory, "training_accuracy_all")
    plot_training_accuracy(acc_averages_proportion, acc_stds_proportion, update_interval, directory, "training_accuracy_proportion")

    print("\nDone.")





