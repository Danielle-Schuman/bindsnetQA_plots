# used to make additional plots from existing data (in csv files)
from main import plot_training_accuracy
import os
import csv


def get_avgs_and_stds_from_csv(filename: str):  # -> List[float], List[float], int
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = [row for row in file]
        end = len(rows)
        # Standard deviations are always the 4th row from below, in a file created by write_to_csv from main
        stds = list(map(float, rows[end - 4]))
        # Averages are always the 6th row from below, in a file created by write_to_csv from main
        avgs = list(map(float, rows[end - 6]))
        # update_interval is always in the 3rd row from the top, in the 13th column, in a file created by write_to_csv from main
        update_interval = int(rows[2][12])
    return avgs, stds, update_interval


def plot_another_training_accuracy(this_dir: str, kind: str):
    b_kind = "b_" + kind
    qa_kind = "qa_" + kind
    acc_averages = {b_kind: [], qa_kind: []}
    acc_stds = {b_kind: [], qa_kind: []}
    b_filename = (this_dir + "/Accuracies BindsNET " + kind + ".csv")
    qa_filename = (this_dir + "/Accuracies BindsNET_QA " + kind + ".csv")
    acc_averages[b_kind], acc_stds[b_kind], update_interval = get_avgs_and_stds_from_csv(b_filename)
    acc_averages[qa_kind], acc_stds[qa_kind], update_interval = get_avgs_and_stds_from_csv(qa_filename)
    plot_training_accuracy(acc_averages, acc_stds, update_interval, this_dir, ("training_accuracy_" + kind))


def plot_shorter_training_accuracies(this_dir: str, length: int):
    b_all_filename = (this_dir + "/Accuracies BindsNET all.csv")
    b_proportion_filename = (this_dir + "/Accuracies BindsNET proportion.csv")
    qa_all_filename = (this_dir + "/Accuracies BindsNET_QA all.csv")
    qa_proportion_filename = (this_dir + "/Accuracies BindsNET_QA proportion.csv")

    acc_avgs_b_all, acc_stds_b_all, update_interval = get_avgs_and_stds_from_csv(b_all_filename)
    acc_avgs_b_proportion, acc_stds_b_proportion, update_interval = get_avgs_and_stds_from_csv(b_proportion_filename)
    acc_avgs_qa_all, acc_stds_qa_all, update_interval = get_avgs_and_stds_from_csv(qa_all_filename)
    acc_avgs_qa_proportion, acc_stds_qa_proportion, update_interval = get_avgs_and_stds_from_csv(qa_proportion_filename)

    list_length = length / update_interval
    acc_avgs_b_all = acc_avgs_b_all[:list_length]
    acc_avgs_b_proportion = acc_avgs_b_proportion[:list_length]
    acc_avgs_qa_all = acc_avgs_qa_all[:list_length]
    acc_avgs_qa_proportion = acc_avgs_qa_proportion[:list_length]
    acc_stds_b_all = acc_stds_b_all[:list_length]
    acc_stds_b_proportion = acc_stds_b_proportion[:list_length]
    acc_stds_qa_all = acc_stds_qa_all[:list_length]
    acc_stds_qa_proportion = acc_stds_qa_proportion[:list_length]

    acc_avgs_dict = {"b_all": acc_avgs_b_all, "b_proportion": acc_avgs_b_proportion, "qa_all": acc_avgs_qa_all, "qa_proportion": acc_avgs_qa_proportion}
    acc_stds_dict = {"b_all": acc_stds_b_all, "b_proportion": acc_stds_b_proportion, "qa_all": acc_stds_qa_all, "qa_proportion": acc_stds_qa_proportion}
    acc_avgs_all_dict = {"b_all": acc_avgs_b_all, "qa_all": acc_avgs_qa_all}
    acc_stds_all_dict = {"b_all": acc_stds_b_all, "qa_all": acc_stds_qa_all}
    acc_avgs_proportion_dict = {"b_proportion": acc_avgs_b_proportion, "qa_proportion": acc_avgs_qa_proportion}
    acc_stds_proportion_dict = {"b_proportion": acc_stds_b_proportion, "qa_proportion": acc_stds_qa_proportion}

    plot_training_accuracy(acc_avgs_dict, acc_stds_dict, update_interval, this_dir, ("training_accuracy_" + str(length)))
    plot_training_accuracy(acc_avgs_all_dict, acc_stds_all_dict, update_interval, this_dir,
                           ("training_accuracy_all_" + str(length)))
    plot_training_accuracy(acc_avgs_proportion_dict, acc_stds_proportion_dict, update_interval, this_dir,
                           ("training_accuracy_proportion_" + str(length)))


# figures that should exist (currently):
# "training_accuracy.png", "training_accuracy_all.png", "training_accuracy_proportion.png"
rootdir = "/Users/Daantje/Sourcecodes/bindsnet_qa_plots/plots"

for this_dir, subdirs, files in os.walk(rootdir):
    if this_dir == rootdir:
        pass
    else:
        if "training_accuracy_all.png" not in files:
            plot_another_training_accuracy(this_dir, "all")
        if "training_accuracy_proportion.png" not in files:
            plot_another_training_accuracy(this_dir, "proportion")
        if "--n_train 100" not in this_dir:
            plot_shorter_training_accuracies(this_dir, 100)
print("Done.")
