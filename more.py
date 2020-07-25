# used to make additional plots from existing data (in csv files)
from main import plot_training_accuracy
from main import write_to_csv
import os
import csv
import numpy as np
import argparse
from typing import Optional, List


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


def get_arguments_list_from_csv(filename: str):
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = [row for row in file]
        # arguments_list is always in the 3rd row from the top, all but the fist, in a file created by write_to_csv from main
        arguments_list = rows[2][1:]
    return arguments_list

def get_avg_wall_clock_times_from_csv(filename: str):
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = [row for row in file]
        end = len(rows)
        # Averages are always the 6th row from below, in a file created by write_to_csv from main
        b_time_avg = float(rows[end - 6][0])
        qa_time_avg = float(rows[end - 6][1])
    return b_time_avg, qa_time_avg


def get_wall_clock_times_from_csv(filename: str):
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = [row for row in file]
        # Wallclocktimes always start at 9th row from top, in a file created by write_to_csv from main
        i = 8
        b_times = []
        qa_times = []
        while rows[i]:  # while row is not empty
            b_times.append(float(rows[i][0]))
            qa_times.append(float(rows[i][1]))
            i = i + 1
    return b_times, qa_times


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


def plot_new_training_accuracies(this_dir: str, length: Optional[int] = None):
    b_all_filename = (this_dir + "/Accuracies BindsNET all.csv")
    b_proportion_filename = (this_dir + "/Accuracies BindsNET proportion.csv")
    qa_all_filename = (this_dir + "/Accuracies BindsNET_QA all.csv")
    qa_proportion_filename = (this_dir + "/Accuracies BindsNET_QA proportion.csv")

    acc_avgs_b_all, acc_stds_b_all, update_interval = get_avgs_and_stds_from_csv(b_all_filename)
    acc_avgs_b_proportion, acc_stds_b_proportion, update_interval = get_avgs_and_stds_from_csv(b_proportion_filename)
    acc_avgs_qa_all, acc_stds_qa_all, update_interval = get_avgs_and_stds_from_csv(qa_all_filename)
    acc_avgs_qa_proportion, acc_stds_qa_proportion, update_interval = get_avgs_and_stds_from_csv(qa_proportion_filename)

    name_suffix = ""
    if length is not None:
        list_length = int(length / update_interval)
        acc_avgs_b_all = acc_avgs_b_all[:list_length]
        acc_avgs_b_proportion = acc_avgs_b_proportion[:list_length]
        acc_avgs_qa_all = acc_avgs_qa_all[:list_length]
        acc_avgs_qa_proportion = acc_avgs_qa_proportion[:list_length]
        acc_stds_b_all = acc_stds_b_all[:list_length]
        acc_stds_b_proportion = acc_stds_b_proportion[:list_length]
        acc_stds_qa_all = acc_stds_qa_all[:list_length]
        acc_stds_qa_proportion = acc_stds_qa_proportion[:list_length]
        name_suffix = "_" + str(length)


    acc_avgs_dict = {"b_all": acc_avgs_b_all, "b_proportion": acc_avgs_b_proportion, "qa_all": acc_avgs_qa_all, "qa_proportion": acc_avgs_qa_proportion}
    acc_stds_dict = {"b_all": acc_stds_b_all, "b_proportion": acc_stds_b_proportion, "qa_all": acc_stds_qa_all, "qa_proportion": acc_stds_qa_proportion}
    acc_avgs_all_dict = {"b_all": acc_avgs_b_all, "qa_all": acc_avgs_qa_all}
    acc_stds_all_dict = {"b_all": acc_stds_b_all, "qa_all": acc_stds_qa_all}
    acc_avgs_proportion_dict = {"b_proportion": acc_avgs_b_proportion, "qa_proportion": acc_avgs_qa_proportion}
    acc_stds_proportion_dict = {"b_proportion": acc_stds_b_proportion, "qa_proportion": acc_stds_qa_proportion}

    plot_training_accuracy(acc_avgs_dict, acc_stds_dict, update_interval, this_dir, ("training_accuracy" + name_suffix))
    plot_training_accuracy(acc_avgs_all_dict, acc_stds_all_dict, update_interval, this_dir,
                           ("training_accuracy_all" + name_suffix))
    plot_training_accuracy(acc_avgs_proportion_dict, acc_stds_proportion_dict, update_interval, this_dir,
                           ("training_accuracy_proportion" + name_suffix))


def calculate_differences(this_dir: str):
    b_all_filename = (this_dir + "/Accuracies BindsNET all.csv")
    b_proportion_filename = (this_dir + "/Accuracies BindsNET proportion.csv")
    qa_all_filename = (this_dir + "/Accuracies BindsNET_QA all.csv")
    qa_proportion_filename = (this_dir + "/Accuracies BindsNET_QA proportion.csv")

    acc_averages_b_all, acc_stds_b_all, update_interval = get_avgs_and_stds_from_csv(b_all_filename)
    acc_averages_b_proportion, acc_stds_b_proportion, update_interval = get_avgs_and_stds_from_csv(b_proportion_filename)
    acc_averages_qa_all, acc_stds_qa_all, update_interval = get_avgs_and_stds_from_csv(qa_all_filename)
    acc_averages_qa_proportion, acc_stds_qa_proportion, update_interval = get_avgs_and_stds_from_csv(qa_proportion_filename)
    # does not matter, which one we take: arguments the same for all
    arguments_list = get_arguments_list_from_csv(b_all_filename)

    acc_averages_diff_all = np.subtract(acc_averages_qa_all, acc_averages_b_all)
    acc_averages_diff_proportion = np.subtract(acc_averages_qa_proportion, acc_averages_b_proportion)
    acc_stds_diff_all = np.subtract(acc_stds_qa_all, acc_stds_b_all)
    acc_stds_diff_proportion = np.subtract(acc_stds_qa_proportion, acc_stds_b_proportion)

    # append average difference to the array
    acc_averages_diff_all = np.append(acc_averages_diff_all, np.array(np.mean(acc_averages_diff_all)))
    acc_averages_diff_proportion = np.append(acc_averages_diff_proportion, np.array(np.mean(acc_averages_diff_proportion)))
    acc_stds_diff_all = np.append(acc_stds_diff_all, np.array(np.mean(acc_stds_diff_all)))
    acc_stds_diff_proportion = np.append(acc_stds_diff_proportion, np.mean(acc_stds_diff_proportion))
    # should all have the same length -> does not matter, which one we use
    diff_column_names = [i for i in range(len(acc_averages_diff_all) -1)]
    diff_column_names.append("Average")

    write_to_csv(this_dir, "Differences all-accuracies", arguments_list, diff_column_names, None, acc_averages_diff_all, acc_stds_diff_all)
    write_to_csv(this_dir, "Differences proportion-accuracies", arguments_list, diff_column_names, None, acc_averages_diff_proportion, acc_stds_diff_proportion)


def write_differences_to_csv(directory: str, heading: str, averages: list, std: list, averages_stds: list, std_stds: list, column_names: Optional[list] = None):
    with open((directory + '/' + heading + '.csv'), 'w+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([heading])
        filewriter.writerow([])
        if column_names is not None:
            filewriter.writerow(['Column names:'])
            filewriter.writerow(column_names)
            filewriter.writerow([])
        filewriter.writerow(['Difference Averages:'])
        filewriter.writerow(averages)
        filewriter.writerow(['Standard Deviations of Difference Averages:'])
        filewriter.writerow(averages_stds)
        filewriter.writerow([])
        if std is not None:
            filewriter.writerow(['Difference Standard deviations:'])
            filewriter.writerow(std)
            filewriter.writerow(['Standard Deviations of Difference Standard deviations:'])
            filewriter.writerow(std_stds)
            filewriter.writerow([])


def calculate_averages_and_stds(data: List[list]):
    max_length = 0
    averages = []
    for item in data:
        averages.append(item[-1])
        item = item[:-1]
        if len(item) > max_length:
            max_length = len(item)
    result_avg = []
    result_std = []
    for i in range(max_length):
        array_i = []
        for item in data:
            if i < len(item):
                array_i.append(item[i])
        result_avg.append(np.mean(array_i))
        result_std.append(np.std(array_i))
    result_avg.append(np.mean(averages))
    result_std.append(np.std(averages))
    return result_avg, result_std


def average_differences(this_dir:str, subdirs: list, over: int, in_name: str, not_in_name: str):
    values_averages_all = []
    values_stds_all = []
    values_averages_proportion = []
    values_stds_proportion = []
    for subdir in subdirs:
        directory = this_dir + '/' + subdir
        filename_all = directory + '/' + 'Differences all-accuracies.csv'
        filename_proportion = directory + '/' + 'Differences proportion-accuracies.csv'
        value_average_all, value_std_all, update_interval = get_avgs_and_stds_from_csv(filename_all)
        value_average_proportion, value_std_proportion, update_interval = get_avgs_and_stds_from_csv(filename_proportion)
        if over is not None:
            length = int(over / update_interval)
            if length < (len(value_average_all) - 1):
                value_average_all = value_average_all[:length]
                value_average_all.append(np.mean(value_average_all))
                value_std_all = value_std_all[:length]
                value_std_all.append(np.mean(value_std_all))
                value_average_proportion = value_average_proportion[:length]
                value_average_proportion.append(np.mean(value_average_proportion))
                value_std_proportion = value_std_proportion[:length]
                value_std_proportion.append(np.mean(value_std_proportion))
        values_averages_all.append(value_average_all)
        values_stds_all.append(value_std_all)
        values_averages_proportion.append(value_average_proportion)
        values_stds_proportion.append(value_std_proportion)

    averages_all, averages_all_stds = calculate_averages_and_stds(values_averages_all)
    stds_all, stds_all_stds = calculate_averages_and_stds(values_stds_all)
    averages_proportion, averages_proportion_stds = calculate_averages_and_stds(values_averages_proportion)
    stds_proportion, stds_proportion_stds = calculate_averages_and_stds(values_stds_proportion)

    heading_proportion = "Average Differences proportion-accuracies"
    heading_all = "Average Differences all-accuracies"
    if in_name is not None:
        heading_all = heading_all + " with " + in_name
        heading_proportion = heading_proportion + " with " + in_name
    if not_in_name is not None:
        heading_all = heading_all + " without " + not_in_name
        heading_proportion = heading_proportion + " without " + not_in_name
    if over is not None:
        heading_all = heading_all + " over " + str(over) + "images"
        heading_proportion = heading_proportion + " over " + str(over) + "images"

    # does not matter which one we take
    column_names = [i for i in range(len(averages_all) -1)]
    column_names.append("Average")

    write_differences_to_csv(this_dir, heading_proportion, averages_proportion, stds_proportion, averages_proportion_stds, stds_proportion_stds, column_names)
    write_differences_to_csv(this_dir, heading_all, averages_all, stds_all, averages_all_stds, stds_all_stds, column_names)


def average_wallclocktime_difference(this_dir: str, subdirs: List[str], in_name: str, not_in_name: str):
    factors = []
    for subdir in subdirs:
        directory = this_dir + '/' + subdir
        filename = directory + '/' + 'Wall clock time taken.csv'
        b_time_avg, qa_time_avg = get_avg_wall_clock_times_from_csv(filename)
        factors.append(qa_time_avg / b_time_avg)
    avg_factor = [np.mean(factors)]
    std_factor = [np.std(factors)]

    heading = "Average Factor QA is slower"
    if in_name is not None:
        heading = heading + " with " + in_name
    if not_in_name is not None:
        heading = heading + " without " + not_in_name

    write_differences_to_csv(this_dir, heading, avg_factor, None, std_factor, None)


def replace_avg_in_csv(filename: str, new_row_content: list):
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = [row for row in file]
        end = len(rows)
        # Averages are always the 6th row from below, in a file created by write_to_csv from main
        rows[end - 6] = new_row_content

    with open(filename, 'w+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerows(rows)


def recalculate_average_wallclocktime(this_dir: str):
    filename = this_dir + '/' + 'Wall clock time taken.csv'
    b_times, qa_times = get_wall_clock_times_from_csv(filename)
    b_median = np.median(b_times)
    qa_median = np.median(qa_times)
    replace_avg_in_csv(filename, [b_median, qa_median])


# figures that should exist (currently):
# "training_accuracy.png", "training_accuracy_all.png", "training_accuracy_proportion.png"
rootdir = "/Users/Daantje/Sourcecodes/bindsnet_qa_plots/plots"

parser = argparse.ArgumentParser()
parser.add_argument("--in_name", type=str)
parser.add_argument("--not_in_name", type=str, default="--n_neurons 10,--num_repeats")
parser.add_argument("--over", type=int)

args = parser.parse_args()

in_name = args.in_name
if in_name is not None:
    in_name_list = in_name.split(",")
not_in_name = args.not_in_name
if not_in_name is not None:
    not_in_name_list = not_in_name.split(",")
over = args.over

for this_dir, subdirs, files in os.walk(rootdir):
    if this_dir == rootdir:
        subdirs_to_use = []
        for name in subdirs:
            if os.listdir(this_dir + '/' + name):  # i.e. directory is not still empty
                use_this = True
                if in_name is not None:
                    for option in in_name_list:
                        if not option in name:
                            use_this = False
                if not_in_name is not None:
                    for option in not_in_name_list:
                        if option in name:
                            use_this = False
                if use_this:
                    subdirs_to_use.append(name)
        average_differences(this_dir, subdirs_to_use, over, in_name, not_in_name)
        average_wallclocktime_difference(this_dir, subdirs_to_use, in_name, not_in_name)

    elif files:  # if it's not the root directory and it's not empty
        if "training_accuracy.png" not in files:
            plot_new_training_accuracies(this_dir)
        if "training_accuracy_all.png" not in files:
            plot_another_training_accuracy(this_dir, "all")
        if "training_accuracy_proportion.png" not in files:
            plot_another_training_accuracy(this_dir, "proportion")
        if "Differences all-accuracies.csv" not in files:
            calculate_differences(this_dir)
        if "--n_train 100" not in this_dir:
            if "training_accuracy_100.png" not in files:
                plot_new_training_accuracies(this_dir, 100)
        # recalculate_average_wallclocktime(this_dir)
print("Done.")
