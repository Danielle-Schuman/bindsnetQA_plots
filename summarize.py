#used to summarize the results of several runs with the same parameter setting (calculating averages, making plots, etc.)
import argparse
import os
import csv
import numpy as np
from typing import List

from more import get_arguments_list_from_csv
from more import get_wall_clock_times_from_csv
from more import get_avgs_and_stds_from_csv
from more import write_filled_to_csv
from main import plot_training_accuracy
from main import write_to_csv


def get_content_from_csv(filename: str):  # -> List[float], List[float], int
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = [row for row in file]
        # content always starts in 11th row from top
        content = [list(map(float, x)) for x in rows[10:-8]]
    return content


def get_seeds_from_csv(filename: str):  # -> List[float], List[float], int
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = [row for row in file]
        # seeds always start in 5th row from top, all but the first item
        seeds = list(map(float, rows[4][1:]))
    return seeds


def average_filled (this_dir: str, subdirs: List[str]):
    means = []
    stds = []
    for subdir in subdirs:
        filename = this_dir + '/' + subdir + '/' + "Filled Percentage of QUBO.csv"
        if os.path.isfile(filename):
            mean, std, update_interval = get_avgs_and_stds_from_csv(filename)
            means.extend(mean)
            stds.extend(std)
    mean_avg = np.mean(means)
    mean_std = np.std(means)
    std_avg = np.mean(stds)
    std_std = np.std(stds)
    heading = "Average Percentage QUBO is filled"
    write_filled_to_csv(this_dir, heading, [mean_avg], [std_avg], [mean_std], [std_std])


parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=str)

args = parser.parse_args()

directory = args.directory

seeds = []
accuracies_b_all = []
accuracies_qa_all = []
accuracies_z_all = []
accuracies_o_all = []
wallclocktime = None
arguments_list = None

for this_dir, subdirs, files in os.walk(directory):
    if not this_dir == directory:
        if files:  # if this_dir is not empty
            # get contents from runs
            if arguments_list is None:
                # does not matter from which csv we get arguments
                arguments_list = get_arguments_list_from_csv(this_dir + "/Accuracies BindsNET all.csv")
            b_accs = get_content_from_csv(this_dir + "/Accuracies BindsNET all.csv")
            accuracies_b_all.extend(b_accs)
            qa_accs = get_content_from_csv(this_dir + "/Accuracies BindsNET_QA all.csv")
            accuracies_qa_all.extend(qa_accs)
            z_accs = get_content_from_csv(this_dir + "/Accuracies BindsNET_bad_zero all.csv")
            accuracies_z_all.extend(z_accs)
            o_accs = get_content_from_csv(this_dir + "/Accuracies BindsNET_bad_one all.csv")
            accuracies_o_all.extend(o_accs)
            b_times, qa_times, z_times, o_times = get_wall_clock_times_from_csv(this_dir + "/Wall clock time taken.csv")
            if wallclocktime is None:
                wallclocktime = [b_times, qa_times, z_times, o_times]
            else:
                wallclocktime[0].extend(b_times)
                wallclocktime[1].extend(qa_times)
                wallclocktime[2].extend(z_times)
                wallclocktime[3].extend(o_times)

            # does not matter which file we use for seeds -> are always the same
            seeds_this_run = get_seeds_from_csv(this_dir + "/Accuracies BindsNET all.csv")
            seeds.extend(seeds_this_run)
    else:
        if subdirs:  # if this_dir is not empty
            average_filled(this_dir, subdirs)

acc_averages_b_all = np.mean(accuracies_b_all, axis=0)
acc_averages_qa_all = np.mean(accuracies_qa_all, axis=0)
acc_averages_z_all = np.mean(accuracies_z_all, axis=0)
acc_averages_o_all = np.mean(accuracies_o_all, axis=0)
acc_averages_diff_all = np.subtract(acc_averages_qa_all, acc_averages_b_all)

acc_stds_b_all = np.std(accuracies_b_all, axis=0)
acc_stds_qa_all = np.std(accuracies_qa_all, axis=0)
acc_stds_z_all = np.std(accuracies_z_all, axis=0)
acc_stds_o_all = np.std(accuracies_o_all, axis=0)
acc_stds_diff_all = np.subtract(acc_stds_qa_all, acc_stds_b_all)

# append average difference to the array
acc_averages_diff_all = np.append(acc_averages_diff_all, np.array(np.mean(acc_averages_diff_all)))
acc_stds_diff_all = np.append(acc_stds_diff_all, np.array(np.mean(acc_stds_diff_all)))
# should all have the same length -> does not matter, which one we use
diff_column_names = [i for i in range(len(acc_averages_diff_all) -1)]
diff_column_names.append("Average")

# "rotate" wallclocktimes to have right direction
wallclocktime = [[wallclocktime[j][i] for j in range(4)] for i in range(len(wallclocktime[0]))]

write_to_csv(directory, "Accuracies BindsNET all", arguments_list, [i for i in range(len(accuracies_b_all[0]))], accuracies_b_all, acc_averages_b_all, acc_stds_b_all, seeds)
write_to_csv(directory, "Accuracies BindsNET_QA all", arguments_list, [i for i in range(len(accuracies_qa_all[0]))], accuracies_qa_all, acc_averages_qa_all, acc_stds_qa_all, seeds)
write_to_csv(directory, "Accuracies BindsNET_bad_zero all", arguments_list, [i for i in range(len(accuracies_z_all[0]))],
             accuracies_z_all, acc_averages_z_all, acc_stds_z_all, seeds)
write_to_csv(directory, "Accuracies BindsNET_bad_one all", arguments_list, [i for i in range(len(accuracies_o_all[0]))],
             accuracies_o_all, acc_averages_o_all, acc_stds_o_all, seeds)
write_to_csv(directory, "Wall clock time taken", arguments_list, ["BindsNet (in sec)", "BindsNET_QA (in sec)", "BindsNET_Bad_Zero (in sec)", "BindsNET_Bad_Ones (in sec)"], wallclocktime, np.median(wallclocktime, axis=0), None, seeds)
write_to_csv(directory, "Differences all-accuracies", arguments_list, diff_column_names, None, acc_averages_diff_all, acc_stds_diff_all)

acc_averages_all = {"b_all": acc_averages_b_all, "qa_all": acc_averages_qa_all, "0_all": acc_averages_z_all, "1_all": acc_averages_o_all}
acc_stds_all = {"b_all": acc_stds_b_all, "qa_all": acc_stds_qa_all, "0_all": acc_stds_z_all, "1_all": acc_stds_o_all}
update_interval = int(arguments_list[10])

plot_training_accuracy(acc_averages_all, acc_stds_all, update_interval, directory, "training_accuracy_all")

print("\nDone.")
