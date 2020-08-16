# used to make additional plots from existing data (in csv files)
from main import plot_training_accuracy
from main import write_to_csv
import os
import csv
import numpy as np
import argparse
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt


def get_avgs_and_stds_from_csv(filename: str):  # -> List[float], List[float], int
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = [row for row in file]
        end = len(rows)
        # Standard deviations are always the 4th row from below, in a file created by write_to_csv from main
        stds = list(map(float, rows[end - 4]))
        # Averages are always the 6th row from below, in a file created by write_to_csv from main
        avgs = list(map(float, rows[end - 6]))
        # update_interval is always in the 3rd row from the top, in the 12th column, in a file created by write_to_csv from main
        update_interval = int(rows[2][11])
    return avgs, stds, update_interval


def get_avgs_filled_from_csv(filename: str):  # -> List[float], List[float], int
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = [row for row in file]
        # Averages are always the 3th row from top, in a file created by write_filled_to_csv
        avgs = list(map(float, rows[3]))
        # Standard deviations are always the 8th row from top, in a file created by write_filled_to_csv
        stds = list(map(float, rows[8]))
    return avgs, stds


def get_arguments_list_from_csv(filename: str):
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = [row for row in file]
        # arguments_list is always in the 3rd row from the top, all but the first, in a file created by write_to_csv from main
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
        # Wallclocktimes always start at 11th row from top, in a file created by write_to_csv from main
        i = 10
        b_times = []
        qa_times = []
        z_times = []
        o_times = []
        while rows[i]:  # while row is not empty
            b_times.append(float(rows[i][0]))
            qa_times.append(float(rows[i][1]))
            z_times.append(float(rows[i][2]))
            o_times.append(float(rows[i][3]))
            i = i + 1
    return b_times, qa_times, z_times, o_times


def plot_average_differences(
    diff_avgs: List[float],
    stds: List[float],
    update_interval: int,
    directory: str,
    name: str,
    figsize: Tuple[float, float] = (10.5, 6)
) -> None:
    # language=rst
    """
    Plot average differences between accuracies of BindsNET and BindsNET_QA code.

    :param diff_avgs: list of average differences between accuracies
    :param stds: list of standard deviation of differences between accuracies
    :param update_interval: Number of examples per accuracy estimate.
    :param directory: Directory where the differences plot will be saved.
    :param name: name for the figure
    :param figsize: Horizontal, vertical figure size in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)

    list_length = len(diff_avgs)
    x = np.array([0.0] + [(i * update_interval) + update_interval for i in range(list_length)])
    y = np.array([0.0] + [d for d in diff_avgs])
    std = np.array([0.0] + [s for s in stds])
    c = 'tab:purple'
    ax.plot(x, y, marker='.', color=c)
    ax.fill_between(x, y - std, y + std, color=c, alpha=0.2)

    ax.set_ylim([-12, 12])
    end = list_length * update_interval
    ax.set_xlim([0, end])
    ax.set_title("Difference between accuracy of QA-like code and BindsNET code")
    ax.set_xlabel("No. of examples")
    ax.set_ylabel("Average of (qa_all - b_all) in %")
    # to have readable number on x-axis, there can be at most 20 ticks; ticks should be multiples of update_interval
    if list_length > 20:
        xticks = int(list_length / 20) * update_interval
    else:
        xticks = update_interval
    ax.set_xticks(range(0, (end + update_interval), xticks))
    ax.set_yticks(range(-11, 12, 1))
    ax.axhline(0, color='k')

    file = directory + '/' + name
    fig.savefig(file)


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
    #b_proportion_filename = (this_dir + "/Accuracies BindsNET proportion.csv")
    qa_all_filename = (this_dir + "/Accuracies BindsNET_QA all.csv")
    #qa_proportion_filename = (this_dir + "/Accuracies BindsNET_QA proportion.csv")

    acc_avgs_b_all, acc_stds_b_all, update_interval = get_avgs_and_stds_from_csv(b_all_filename)
    #acc_avgs_b_proportion, acc_stds_b_proportion, update_interval = get_avgs_and_stds_from_csv(b_proportion_filename)
    acc_avgs_qa_all, acc_stds_qa_all, update_interval = get_avgs_and_stds_from_csv(qa_all_filename)
    #acc_avgs_qa_proportion, acc_stds_qa_proportion, update_interval = get_avgs_and_stds_from_csv(qa_proportion_filename)

    name_suffix = ""
    if length is not None:
        list_length = int(length / update_interval)
        # does not matter which list we take -> should all have the same length
        if list_length > len(acc_avgs_b_all):
            acc_avgs_b_all = acc_avgs_b_all[:list_length]
            #acc_avgs_b_proportion = acc_avgs_b_proportion[:list_length]
            acc_avgs_qa_all = acc_avgs_qa_all[:list_length]
            #acc_avgs_qa_proportion = acc_avgs_qa_proportion[:list_length]
            acc_stds_b_all = acc_stds_b_all[:list_length]
            #acc_stds_b_proportion = acc_stds_b_proportion[:list_length]
            acc_stds_qa_all = acc_stds_qa_all[:list_length]
            #acc_stds_qa_proportion = acc_stds_qa_proportion[:list_length]
            name_suffix = "_" + str(length)
        else:
            return


    #acc_avgs_dict = {"b_all": acc_avgs_b_all, "b_proportion": acc_avgs_b_proportion, "qa_all": acc_avgs_qa_all, "qa_proportion": acc_avgs_qa_proportion}
    #acc_stds_dict = {"b_all": acc_stds_b_all, "b_proportion": acc_stds_b_proportion, "qa_all": acc_stds_qa_all, "qa_proportion": acc_stds_qa_proportion}
    acc_avgs_all_dict = {"b_all": acc_avgs_b_all, "qa_all": acc_avgs_qa_all}
    acc_stds_all_dict = {"b_all": acc_stds_b_all, "qa_all": acc_stds_qa_all}
    #acc_avgs_proportion_dict = {"b_proportion": acc_avgs_b_proportion, "qa_proportion": acc_avgs_qa_proportion}
    #acc_stds_proportion_dict = {"b_proportion": acc_stds_b_proportion, "qa_proportion": acc_stds_qa_proportion}

    #plot_training_accuracy(acc_avgs_dict, acc_stds_dict, update_interval, this_dir, ("training_accuracy" + name_suffix))
    plot_training_accuracy(acc_avgs_all_dict, acc_stds_all_dict, update_interval, this_dir,
                           ("training_accuracy_all" + name_suffix))
    #plot_training_accuracy(acc_avgs_proportion_dict, acc_stds_proportion_dict, update_interval, this_dir,
                           #("training_accuracy_proportion" + name_suffix))


def calculate_differences(this_dir: str):
    b_all_filename = (this_dir + "/Accuracies BindsNET all.csv")
    # b_proportion_filename = (this_dir + "/Accuracies BindsNET proportion.csv")
    qa_all_filename = (this_dir + "/Accuracies BindsNET_QA all.csv")
    # qa_proportion_filename = (this_dir + "/Accuracies BindsNET_QA proportion.csv")

    acc_averages_b_all, acc_stds_b_all, update_interval = get_avgs_and_stds_from_csv(b_all_filename)
    # acc_averages_b_proportion, acc_stds_b_proportion, update_interval = get_avgs_and_stds_from_csv(b_proportion_filename)
    acc_averages_qa_all, acc_stds_qa_all, update_interval = get_avgs_and_stds_from_csv(qa_all_filename)
    # acc_averages_qa_proportion, acc_stds_qa_proportion, update_interval = get_avgs_and_stds_from_csv(qa_proportion_filename)
    # does not matter, which one we take: arguments the same for all
    arguments_list = get_arguments_list_from_csv(b_all_filename)

    acc_averages_diff_all = np.subtract(acc_averages_qa_all, acc_averages_b_all)
    # acc_averages_diff_proportion = np.subtract(acc_averages_qa_proportion, acc_averages_b_proportion)
    acc_stds_diff_all = np.subtract(acc_stds_qa_all, acc_stds_b_all)
    # acc_stds_diff_proportion = np.subtract(acc_stds_qa_proportion, acc_stds_b_proportion)

    # append average difference to the array
    acc_averages_diff_all = np.append(acc_averages_diff_all, np.array(np.mean(acc_averages_diff_all)))
    # acc_averages_diff_proportion = np.append(acc_averages_diff_proportion, np.array(np.mean(acc_averages_diff_proportion)))
    acc_stds_diff_all = np.append(acc_stds_diff_all, np.array(np.mean(acc_stds_diff_all)))
    # acc_stds_diff_proportion = np.append(acc_stds_diff_proportion, np.mean(acc_stds_diff_proportion))
    # should all have the same length -> does not matter, which one we use
    diff_column_names = [i for i in range(len(acc_averages_diff_all) -1)]
    diff_column_names.append("Average")

    write_to_csv(this_dir, "Differences all-accuracies", arguments_list, diff_column_names, None, acc_averages_diff_all, acc_stds_diff_all)
    # write_to_csv(this_dir, "Differences proportion-accuracies", arguments_list, diff_column_names, None, acc_averages_diff_proportion, acc_stds_diff_proportion)


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


def write_filled_to_csv(directory: str, heading: str, averages: list, std: list, averages_stds: list, std_stds: list, column_names: Optional[list] = None):
    with open((directory + '/' + heading + '.csv'), 'w+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([heading])
        filewriter.writerow([])
        if column_names is not None:
            filewriter.writerow(['Column names:'])
            filewriter.writerow(column_names)
            filewriter.writerow([])
        filewriter.writerow(['Averages:'])
        filewriter.writerow(averages)
        filewriter.writerow(['Standard Deviations of Averages:'])
        filewriter.writerow(averages_stds)
        filewriter.writerow([])
        if std is not None:
            filewriter.writerow(['Standard deviations:'])
            filewriter.writerow(std)
            filewriter.writerow(['Standard Deviations of Standard deviations:'])
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
    print("Minimum average difference:" + str(np.amin(averages)))
    result_std.append(np.std(averages))
    return result_avg, result_std


def average_differences(this_dir:str, subdirs: list, over: int, in_name: str, not_in_name: str):
    values_averages_all = []
    values_stds_all = []
    # values_averages_proportion = []
    # values_stds_proportion = []
    for subdir in subdirs:
        directory = this_dir + '/' + subdir
        filename_all = directory + '/' + 'Differences all-accuracies.csv'
        # filename_proportion = directory + '/' + 'Differences proportion-accuracies.csv'
        value_average_all, value_std_all, update_interval = get_avgs_and_stds_from_csv(filename_all)
        # value_average_proportion, value_std_proportion, update_interval = get_avgs_and_stds_from_csv(filename_proportion)
        if over is not None:
            length = int(over / update_interval)
            if length < (len(value_average_all) - 1):
                value_average_all = value_average_all[:length]
                value_average_all.append(np.mean(value_average_all))
                value_std_all = value_std_all[:length]
                value_std_all.append(np.mean(value_std_all))
                # value_average_proportion = value_average_proportion[:length]
                # value_average_proportion.append(np.mean(value_average_proportion))
                # value_std_proportion = value_std_proportion[:length]
                # value_std_proportion.append(np.mean(value_std_proportion))
        values_averages_all.append(value_average_all)
        values_stds_all.append(value_std_all)
        # values_averages_proportion.append(value_average_proportion)
        # values_stds_proportion.append(value_std_proportion)

    averages_all, averages_all_stds = calculate_averages_and_stds(values_averages_all)
    stds_all, stds_all_stds = calculate_averages_and_stds(values_stds_all)
    # averages_proportion, averages_proportion_stds = calculate_averages_and_stds(values_averages_proportion)
    # stds_proportion, stds_proportion_stds = calculate_averages_and_stds(values_stds_proportion)

    # heading_proportion = "Average Differences proportion-accuracies"
    heading_all = "Average Differences all-accuracies"
    if in_name is not None:
        heading_all = heading_all + " with " + in_name
        # heading_proportion = heading_proportion + " with " + in_name
    if not_in_name is not None:
        heading_all = heading_all + " without " + not_in_name
        # heading_proportion = heading_proportion + " without " + not_in_name
    if over is not None:
        heading_all = heading_all + " over " + str(over) + "images"
        # heading_proportion = heading_proportion + " over " + str(over) + "images"

    # does not matter which one we take
    column_names = [i for i in range(len(averages_all) -1)]
    column_names.append("Average")

    # write_differences_to_csv(this_dir, heading_proportion, averages_proportion, stds_proportion, averages_proportion_stds, stds_proportion_stds, column_names)
    write_differences_to_csv(this_dir, heading_all, averages_all, stds_all, averages_all_stds, stds_all_stds, column_names)
    if over is None and in_name is None and not_in_name == "--num_repeats":
        plot_average_differences(averages_all[:-1], averages_all_stds[:-1], update_interval, this_dir, heading_all)


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
    b_times, qa_times, z_times, o_times = get_wall_clock_times_from_csv(filename)
    b_median = np.median(b_times)
    qa_median = np.median(qa_times)
    replace_avg_in_csv(filename, [b_median, qa_median])


def average_filled (this_dir: str, subdirs: List[str]):
    means = []
    stds = []
    for subdir in subdirs:
        filename = this_dir + '/' + subdir + '/' + "Average Percentage QUBO is filled.csv"
        if os.path.isfile(filename):
            mean, std = get_avgs_filled_from_csv(filename)
            means.extend(mean)
            stds.extend(std)
    mean_avg = np.mean(means)
    mean_std = np.std(means)
    std_avg = np.mean(stds)
    std_std = np.std(stds)
    heading = "Average Percentage QUBO is filled"
    write_filled_to_csv(this_dir, heading, [mean_avg], [std_avg], [mean_std], [std_std])


if __name__ == "__main__":
    rootdir = "/Users/Daantje/Sourcecodes/bindsnet_qa_plots/plots"

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_name", type=str)
    parser.add_argument("--not_in_name", type=str, default="--num_repeats")
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
            if in_name is None and not_in_name == "--num_repeats" and over is None:
                average_filled(this_dir, subdirs_to_use)

        elif files:  # if it's not the root directory and it's not empty
            # if "training_accuracy.png" not in files:
                # plot_new_training_accuracies(this_dir)
            if "training_accuracy_all.png" not in files:
                plot_another_training_accuracy(this_dir, "all")
            # if "training_accuracy_proportion.png" not in files:
                # plot_another_training_accuracy(this_dir, "proportion")
            if "Differences all-accuracies.csv" not in files:
                calculate_differences(this_dir)
            if "--n_train 1000" not in this_dir:
                if "training_accuracy_1000.png" not in files:
                    plot_new_training_accuracies(this_dir, 1000)
            # recalculate_average_wallclocktime(this_dir)
    print("Done.")
