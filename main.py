import argparse
import numpy as np
import csv
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from examples.mnist import supervised_mnist
from examples_qa.mnist import supervised_mnist as supervised_mnist_qa


def write_to_csv(directory: str, heading: str, arguments: list, column_names: Optional[list] = None, content: List[list] = None, averages: Optional[list] = None, std: Optional[list] = None):
    with open((directory + '/' + heading + '.csv'), 'a+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([heading])
        filewriter.writerow(['Parameters:', 'runs', 'seed', 'n_neurons', 'n_train', 'n_test', 'n_clamp', 'exc', 'inh', 'time', 'dt', 'intensity', 'update_interval', 'train', 'plot', 'num_repeats', 'gpu'])
        parameters = ['']
        parameters.extend(arguments)
        filewriter.writerow(parameters)
        filewriter.writerow([])
        filewriter.writerow([])
        if column_names is not None:
            filewriter.writerow(['Column names:'])
            filewriter.writerow(column_names)
            filewriter.writerow([])
        filewriter.writerows(content)
        filewriter.writerow([])
        if averages is None:
            averages = np.mean(content, axis=0)
        filewriter.writerow(['Averages:'])
        filewriter.writerow(averages)
        if std is None:
            std = np.std(content, axis=0)
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
    ax.plot(x, y, label=label, marker='.')
    ax.fill_between(x, y - std, y + std, alpha=0.2)


def plot_training_accuracy(
    acc_avg_b_all: List[float],
    acc_avg_b_proportion: List[float],
    acc_avg_qa_all: List[float],
    acc_avg_qa_proportion: List[float],
    std_b_all: List[float],
    std_b_proportion: List[float],
    std_qa_all: List[float],
    std_qa_proportion: List[float],
    update_interval: int,
    directory: str,
    figsize: Tuple[float, float] = (10.5, 6)
) -> None:
    # language=rst
    """
    Plot training accuracy curves.

    :param acc_avg_b_all: average BindsNET all accuracy
    :param acc_avg_b_proportion: average BindsNET proportion accuracy
    :param acc_avg_qa_all: average BindsNET_QA all accuracy
    :param acc_avg_qa_proportion: average BindsNET_QA proportion accuracy
    :param std_b_all: Standard deviation of BindsNET all accuracy
    :param std_b_proportion: Standard deviation of BindsNET proportion accuracy
    :param std_qa_all: Standard deviation of BindsNET_QA all accuracy
    :param std_qa_proportion: Standard deviation of BindsNET_QA proportion accuracy
    :param update_interval: Number of examples_qa per accuracy estimate.
    :param directory: Directory where the training accuracy plot will be saved.
    :param figsize: Horizontal, vertical figure size in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # does not matter, of which acc_avg we take the length: lengths are all equal
    x = np.array([0.0] + [(i * update_interval) + update_interval for i in range(len(acc_avg_b_all))])
    plot_one_training_accuracy(acc_avg_b_all, std_b_all, ax, x, "b_all")
    plot_one_training_accuracy(acc_avg_b_proportion, std_b_proportion, ax, x, "b_proportion")
    plot_one_training_accuracy(acc_avg_qa_all, std_qa_all, ax, x, "qa_all")
    plot_one_training_accuracy(acc_avg_qa_proportion, std_qa_proportion, ax, x, "qa_proportion")

    ax.set_ylim([0, 110])
    # does not matter, of which acc_avg we take the length: lengths are all equal
    end = len(acc_averages_b_all) * update_interval
    ax.set_xlim([0, end])
    ax.set_title("Estimated classification accuracy")
    ax.set_xlabel("No. of examples")
    ax.set_ylabel("Training accuracy in %")
    ax.set_xticks(range(0, (end + update_interval), update_interval))
    ax.set_yticks(range(0, 110, 10))
    ax.legend()

    file = directory + '/training_accuracy'
    fig.savefig(file)


parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
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
seed = args.seed
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
arguments_list = [runs, seed, n_neurons, n_train, n_test, n_clamp, exc, inh, time, dt, intensity, update_interval, train, plot, num_repeats, gpu]

accuracies_b_all = []
accuracies_b_proportion = []
accuracies_qa_all = []
accuracies_qa_proportion = []
wallclocktime = []
qb_solv_energies_for_runs = []
qb_solv_energies_for_runs_layout = []

for run in range(runs):
    print("BindsNET: Run %d" % run)
    accuracies_b, wallclocktime_b = supervised_mnist.supervised_mnist(seed, n_neurons, n_train, n_test, n_clamp, exc, inh, time, dt, intensity, update_interval, train, plot, other_plots_directory, gpu)
    accuracies_b_all.append(accuracies_b["all"])
    accuracies_b_proportion.append(accuracies_b["proportion"])
    print("BindsNET_QA: Run %d" % run)
    accuracies_qa, wallclocktime_qa, qb_solv_energies = supervised_mnist_qa.supervised_mnist(seed, n_neurons, n_train, n_test, n_clamp, exc, inh, time, dt, intensity, num_repeats, update_interval, train, plot, other_plots_directory, gpu)
    accuracies_qa_all.append(accuracies_qa["all"])
    accuracies_qa_proportion.append(accuracies_qa["proportion"])
    wallclocktime.append([wallclocktime_b, wallclocktime_qa])
    qb_solv_energies_for_runs.extend(qb_solv_energies)
    qb_solv_energies_for_runs_layout.append('')
    qb_solv_energies_for_runs_layout.extend(qb_solv_energies)

acc_averages_b_all = np.mean(accuracies_b_all, axis=0)
acc_averages_b_proportion = np.mean(accuracies_b_proportion, axis=0)
acc_averages_qa_all = np.mean(accuracies_qa_all, axis=0)
acc_averages_qa_proportion = np.mean(accuracies_qa_proportion, axis=0)

acc_stds_b_all = np.std(accuracies_b_all, axis=0)
acc_stds_b_proportion = np.std(accuracies_b_proportion, axis=0)
acc_stds_qa_all = np.std(accuracies_qa_all, axis=0)
acc_stds_qa_proportion = np.std(accuracies_qa_proportion, axis=0)

qb_solv_averages = np.mean(qb_solv_energies_for_runs, axis=0)
qb_solv_stds = np.std(qb_solv_energies_for_runs, axis=0)

write_to_csv(directory, "Accuracies BindsNET all", arguments_list, [i for i in range(len(accuracies_b_all[0]))], accuracies_b_all, acc_averages_b_all, acc_stds_b_all)
write_to_csv(directory, "Accuracies BindsNET proportion", arguments_list, [i for i in range(len(accuracies_b_proportion[0]))], accuracies_b_proportion, acc_averages_b_proportion, acc_stds_b_proportion)
write_to_csv(directory, "Accuracies BindsNET_QA all", arguments_list, [i for i in range(len(accuracies_qa_all[0]))], accuracies_qa_all, acc_averages_qa_all, acc_stds_qa_all)
write_to_csv(directory, "Accuracies BindsNET_QA proportion", arguments_list, [i for i in range(len(accuracies_qa_proportion[0]))], accuracies_qa_proportion, acc_averages_qa_proportion, acc_stds_qa_proportion)
write_to_csv(directory, "Wall clock time taken", arguments_list, ["BindsNet (in sec)", "BindsNET_QA (in sec)"], wallclocktime)
write_to_csv(directory, "Qb_solv_energies", arguments_list, [i for i in range(int(time / dt))], qb_solv_energies_for_runs_layout, qb_solv_averages, qb_solv_stds)

plot_training_accuracy(acc_averages_b_all, acc_averages_b_proportion, acc_averages_qa_all, acc_averages_qa_proportion, acc_stds_b_all, acc_stds_b_proportion, acc_stds_qa_all, acc_stds_qa_proportion, update_interval, directory)

print("\nDone.")





