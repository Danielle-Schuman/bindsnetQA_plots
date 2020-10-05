Based on [BindsNET](https://github.com/BindsNET/bindsnet) – a spiking neural network simulation library geared towards the development of biologically inspired algorithms for machine learning – this Bachelor-Project replaces the forward-step of the `supervised_mnist.py`-example (in `network.run(...)` in `bindsnet_qa/bindsnet_qa/network/network_qa.py`) with the usage of a simualtion Quantum Annealing, utilizing D-Wave's qbsolv Package.

Documentation for the BindsNET-package can be found [here](https://bindsnet-docs.readthedocs.io).

## Requirements

- Python 3.6 or higher
- `requirements.txt`

## Setting things up using pip
To use this project's code, one first needs to install the modified `bindsnet` package contained in this code by calling

```
cd bindsnet_qa_plots/bindsnet
pip install -e .
```

## Getting started

To run a near-replication of the SNN from [this paper](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full) with both the classical BindsNET code as well as the version using simulated Quantum Annealing (QA) (and two ‘baselines’ that do not actually run the network in the forward-step), issue for example

```
python main.py --runs 1 --n_clamp 0 --intensity 170 --inh 60 --n_neurons 50 --n_train 1000 --time 80 --update_interval 50 `--directory DIRECTORY`
```

where `DIRECTORY` is the absolute path to the directory you want to save the plots and data created by the program in. The data is saved in the form of csv-files and is comprised of the data used to create the plots, as well as data on wall clock times taken, ‘fill level’ of the QUBOs used in the simulated QA (executed by qbsolv) and the energy values returned by qbsolv. 
Caveat: The files for the latter kind of data can become quite large (e.g. 13 MB for 10 runs with the arguments from above example). Also, the code runs a little slow (one run with the arguments given above takes on average 27 min).

There are a number of optional command-line arguments which can be passed, including `--runs [int]` (determines the number of times the algorithm is run to calculate average training accuracy, wall clock times and QUBO fill level over), `--n_clamp [int]` (the number of neurons in the network whose spike values are being clamped to 1), `--intensity [float]` (the base intensity of the Poisson-distributed input spikes), `--inh [float]` (the strength of the fixed weights of the connections of a layer of inhibitory neurons to the network's output neurons), `--n_neurons [int]` (determines the number of neurons in each non-input layer of the network), `--time [int]` (the number of forward-timesteps per MNIST-Datum),  `--n_train [int]` (total number of training iterations), `--update_interval [int]` (determines how often the current training accuracy is calculated), `--num_repeats [int]` (determines the argument `num_repeats` to the qbsolv algorithm) `--plot` (displays useful monitoring figures), and more. 
Run the script with the `--help` or `-h` flag for more information.

To summarize / average over the data collected in several sets of runs with the same command line arguments, run 
```
python summarize.py --directory SUPER_DIRECTORY
```
where `SUPER_DIRECTORY` is a directory containing subdirectories holding the data collected in the sets of runs that are to be summarized.

To summarize / average over the data collected in runs with different sets of command line arguments, change the variable `rootdir` in the file `more.py` to the absolute path of a folder containing your ‘superdirectories‘ (mentioned above) and run 
```
python more.py
```
You can use the command line arguments `--in_name [str]` and `--not_in_name [str]` to only include or exclude certain sets of runs from being averaged over that have particular strings in their superdirectory's name. You do this by passing these substrings as one comma-separated string. If your superdirectories' names contain the values of certain arguments to the runs – having e.g. “n_neurons_50_time_80” as a substring – you can average only over the runs that used 50 neurons and time 80 by calling 
```
python more.py --in_name "n_neurons_50,time_80"
```
By default, superdirectories containing "num_repeats" are excluded.
You can use the argument `--over [int]` to provide a maximum number of training iterations that is averaged over. This is particularly useful when e.g. some of your runs have 1000 and others have 2000 training iterations, and you thus want to average only over the first 1000 iterations of all runs.

## Background
For more information on the algorithms used, compare “Daniëlle Schuman. Using Quantum Annealing in Spiking Neural Network Execution. Bachelor's thesis, Ludwig-Maximilians-Universität München, Munich, Oct. 2020”

## Benchmarking and Data Collection
As of now, the code using qbsolv runs 20 times slower compared to the original BindsNET-version, but displays about the same ‘all activity accuracy’ during training. Reasons for this can be found in the above-mentioned thesis.

The data collected using this software that was used for the Evaluation in the above-mentioned thesis can be found in the directory `plots`. This contains ‘superdirectories’ named after the commandline arguments used for the runs it holds. These runs where collected in batches of 10, each of these batches having their own directory inside the respective superdirectory. For each setting of commandline arguments, 100 runs where performed to average over. Averaging was performed as described above. The data was obtained using the current version of the code in this folder.

## Citation

As I am using BindsNET, I'm hereby citing [this article](https://www.frontiersin.org/article/10.3389/fninf.2018.00089):

```
@ARTICLE{10.3389/fninf.2018.00089,
	AUTHOR={Hazan, Hananel and Saunders, Daniel J. and Khan, Hassaan and Patel, Devdhar and Sanghavi, Darpan T. and Siegelmann, Hava T. and Kozma, Robert},   
	TITLE={BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python},      
	JOURNAL={Frontiers in Neuroinformatics},      
	VOLUME={12},      
	PAGES={89},     
	YEAR={2018}, 
	URL={https://www.frontiersin.org/article/10.3389/fninf.2018.00089},       
	DOI={10.3389/fninf.2018.00089},      
	ISSN={1662-5196},
}

```

## Contributors and Authorship

### Contributors
- Daniëlle Schuman ([email](mailto:d.schuman@campus.lmu.de)), the author of this READ-ME

To BindsNET:
- Daniel Saunders ([email](mailto:djsaunde@cs.umass.edu))
- Hananel Hazan ([email](mailto:hananel@hazan.org.il))
- Darpan Sanghavi ([email](mailto:dsanghavi@cs.umass.edu))
- Hassaan Khan ([email](mailto:hqkhan@umass.edu))
- Devdhar Patel ([email](mailto:devdharpatel@cs.umass.edu)

### Autorship
Most of the code in this project was taken from the Spiking Neural Network Library [BindsNET](https://github.com/BindsNET/bindsnet).
An almost unmodified version of this code can be found in the directory `bindsnet`. Here, I only modified the file `examples/mnist/supervised_mnist.py` for the contained code to be executable by calling a method instead of running it as a script. Also, I added the recording of the wall clock time and modified the plotting in order to be able to let the plots be automatically saved in a certain directory and look better. To implement this enhanced plotting, I also altered some methods and wrote the method `save_plot(...)` in `bindsnet/bindsnet/analysis/plotting.py`.

The folder `bindsnet_qa` contains the code that executes the network using qb_solv. While most of this code is copied from the BindsNET-code, too, I here modified the method `run(...)` in `bindsnet_qa/bindsnet_qa/network/network_qa.py` to call the method `forward_qa(...)` which executes the network using qbsolv. This method was written by me, apart from some classical parts (such as decaying voltages or resetting the membrane potential) that were copied from BindsNET's `forward(...)`-methods in `bindsnet/bindsnet/network/nodes.py`. Also, I authored the methods `reward_inhibitory(...)` and `penalty_one_spike(...)`, of which only the former is used in the current version of the code. I intentionally left some commented code for determining the wall clock times of certain parts of the code in the methods I wrote, in order to show how this can be monitored.
The authorship of the code in the folders `bindsnet_bad_ones` and `bindsnet_bad_zeros` is distributed analogously to that in `bindsnet_qa`. The code in these folders provides a ‘baseline’ for the comparison of training accuracies by returning only strings of ones or zeros, respectively, at the points in the code where `bindsnet_qa` returns the spike values that were calculated using qbsolv and `bindsnet` returns the classically calculated spike values of the network.

Apart from the plotting methods (which where copied from BindsNET and then modified to fit my purposes), the code in `main.py`, `more.py` and `summarize.py` was entirely written by me.
This READ-ME was mostly written by me, but contains some parts that were taken from BindsNET's READ-ME (can be found in the folder `bindsnet`). Requirements are mostly those from BindsNET.