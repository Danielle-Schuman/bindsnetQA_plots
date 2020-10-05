import tempfile
from typing import Dict, Optional, Type, Iterable
import time as clock

import torch
import dwave_qbsolv as qbs

from bindsnet.network.monitors import AbstractMonitor
from bindsnet.network.nodes import Nodes
from bindsnet.network.topology import AbstractConnection
from bindsnet.learning.reward import AbstractReward


# While classical code would indicate a spike at 0 = thresh - v (as the equation is v >= thresh),
# whether Quantum Annealer would do this is random, as the energy value would normally be = 0 here.
# So to make sure it indicates a spike, we need to give it a nudging value whose absolute value is smaller than
# the smallest "actual" changes to the values in our equation that occur each timestep,
# which should be the changes to the weights (which are largely determined by the decay of the Inputs)
# Problem: Smallest changes are at approximately the fourth digit after the point,
# while Quantum Annealers at the moment have a precision of 3 digits -> No effect -> we don't need it
# NUDGE = -0.0001


def load(file_name: str, map_location: str = "cpu", learning: bool = None) -> "Network":
    # language=rst
    """
    Loads serialized network object from disk.

    :param file_name: Path to serialized network object on disk.
    :param map_location: One of ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
    :param learning: Whether to load with learning enabled. Default loads value from
        disk.
    """
    network = torch.load(open(file_name, "rb"), map_location=map_location)
    if learning is not None and "learning" in vars(network):
        network.learning = learning

    return network


class Network(torch.nn.Module):
    # language=rst
    """
    Central object of the ``bindsnet_qa`` package. Responsible for the simulation and
    interaction of nodes and connections.

    **Example:**

    .. code-block:: python

        import torch
        import matplotlib.pyplot as plt

        from bindsnet_qa         import encoding
        from bindsnet_qa.network import Network, nodes, topology, monitors

        network = Network(dt=1.0)  # Instantiates network.

        X = nodes.Input(100)  # Input layer.
        Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
        C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

        # Spike monitor objects.
        M1 = monitors.Monitor(obj=X, state_vars=['s'])
        M2 = monitors.Monitor(obj=Y, state_vars=['s'])

        # Add everything to the network object.
        network.add_layer(layer=X, name='X')
        network.add_layer(layer=Y, name='Y')
        network.add_connection(connection=C, source='X', target='Y')
        network.add_monitor(monitor=M1, name='X')
        network.add_monitor(monitor=M2, name='Y')

        # Create Poisson-distributed spike train inputs.
        data = 15 * torch.rand(100)  # Generate random Poisson rates for 100 input neurons.
        train = encoding.poisson(datum=data, time=5000)  # Encode input as 5000ms Poisson spike trains.

        # Simulate network on generated spike trains.
        inputs = {'X' : train}  # Create inputs mapping.
        network.run(inputs=inputs, time=5000)  # Run network simulation.

        # Plot spikes of input and output layers.
        spikes = {'X' : M1.get('s'), 'Y' : M2.get('s')}

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        for i, layer in enumerate(spikes):
            axes[i].matshow(spikes[layer], cmap='binary')
            axes[i].set_title('%s spikes' % layer)
            axes[i].set_xlabel('Time'); axes[i].set_ylabel('Index of neuron')
            axes[i].set_xticks(()); axes[i].set_yticks(())
            axes[i].set_aspect('auto')

        plt.tight_layout(); plt.show()
    """

    def __init__(
        self,
        dt: float = 1.0,
        batch_size: int = 1,
        learning: bool = True,
        reward_fn: Optional[Type[AbstractReward]] = None,
    ) -> None:
        # language=rst
        """
        Initializes network object.

        :param dt: Simulation timestep.
        :param batch_size: Mini-batch size.
        :param learning: Whether to allow connection updates. True by default.
        :param reward_fn: Optional class allowing for modification of reward in case of
            reward-modulated learning.
        """
        super().__init__()

        self.dt = dt
        self.batch_size = batch_size

        self.layers = {}
        self.connections = {}
        self.monitors = {}

        self.train(learning)

        if reward_fn is not None:
            self.reward_fn = reward_fn()
        else:
            self.reward_fn = None

    def add_layer(self, layer: Nodes, name: str) -> None:
        # language=rst
        """
        Adds a layer of nodes to the network.

        :param layer: A subclass of the ``Nodes`` object.
        :param name: Logical name of layer.
        """
        self.layers[name] = layer
        self.add_module(name, layer)

        layer.train(self.learning)
        layer.compute_decays(self.dt)
        layer.set_batch_size(self.batch_size)

    def add_connection(
        self, connection: AbstractConnection, source: str, target: str
    ) -> None:
        # language=rst
        """
        Adds a connection between layers of nodes to the network.

        :param connection: An instance of class ``Connection``.
        :param source: Logical name of the connection's source layer.
        :param target: Logical name of the connection's target layer.
        """
        self.connections[(source, target)] = connection
        self.add_module(source + "_to_" + target, connection)

        connection.dt = self.dt
        connection.train(self.learning)

    def add_monitor(self, monitor: AbstractMonitor, name: str) -> None:
        # language=rst
        """
        Adds a monitor on a network object to the network.

        :param monitor: An instance of class ``Monitor``.
        :param name: Logical name of monitor object.
        """
        self.monitors[name] = monitor
        monitor.network = self
        monitor.dt = self.dt

    def save(self, file_name: str) -> None:
        # language=rst
        """
        Serializes the network object to disk.

        :param file_name: Path to store serialized network object on disk.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from pathlib          import Path
            from bindsnet_qa.network import *
            from bindsnet_qa.network import topology

            # Build simple network.
            network = Network(dt=1.0)

            X = nodes.Input(100)  # Input layer.
            Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
            C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

            # Add everything to the network object.
            network.add_layer(layer=X, name='X')
            network.add_layer(layer=Y, name='Y')
            network.add_connection(connection=C, source='X', target='Y')

            # Save the network to disk.
            network.save(str(Path.home()) + '/network.pt')
        """
        torch.save(self, open(file_name, "wb"))

    def clone(self) -> "Network":
        # language=rst
        """
        Returns a cloned network object.
        
        :return: A copy of this network.
        """
        virtual_file = tempfile.SpooledTemporaryFile()
        torch.save(self, virtual_file)
        virtual_file.seek(0)
        return torch.load(virtual_file)

    def _get_inputs(self, layers: Iterable = None) -> Dict[str, torch.Tensor]:
        # language=rst
        """
        Fetches outputs from network layers to use as input to downstream layers.

        :param layers: Layers to update inputs for. Defaults to all network layers.
        :return: Inputs to all layers for the current iteration.
        """
        inputs = {}

        if layers is None:
            layers = self.layers

        # Loop over network connections.
        for c in self.connections:
            if c[1] in layers:
                # Fetch source and target populations.
                source = self.connections[c].source
                target = self.connections[c].target

                if not c[1] in inputs:
                    inputs[c[1]] = torch.zeros(
                        self.batch_size, *target.shape, device=target.s.device
                    )

                # Add to input: source's spikes multiplied by connection weights.
                inputs[c[1]] += self.connections[c].compute(source.s)

        return inputs

    def penalty_one_spike(self, layer: str) -> float:
        # language=rst
        """
        Calculates the penalty-value for the Quantum Annealer which is used to prevent a layer from having
        two spiking nodes. Currently not in use, since using it would ‘artificially’ change the networks spiking behaviour

        :return: a float value to be used in Quantum Annealing to keep a certain node from spiking if another node in
            its layer spikes
        """
        penalty = 1
        # for every incoming connection
        for c in self.connections:
            if c[1] == layer:
                c_v = self.connections[c]
                if c_v.wmax > 0:
                    # increase punishment by the maximum weight * number of incoming connections
                    penalty += (c_v.wmax * c_v.source.n)
                # max_bias = torch.max(c_v.b)
                # if max_bias > 0: # never the case: is always bias = 0 in our example
                     # if that makes a positive impact for QA, add bias
                     # penalty += max_bias
        # punishment is now bigger than the cumulative value of all "new" inputs to the layer in this timestep
        # 2* because is needed for "row node" as well as "column node"
        return 2 * penalty

    def reward_inhibitory(self, layer: str) -> float:
        # language=rst
        """
        Calculates the reward-value for the Quantum Annealer which is used to prevent a layer from "ignoring" inhibitory
        inputs. To be calculated for the inhibitory layer.

        :return: a float value to be used in Quantum Annealing to clamp a certain inhibitory node's qubit, whose node
            spiked in the last time window (and is now in refractory period), to 1
        """
        reward = -1
        # for every outgoing connection
        for c in self.connections:
            if c[0] == layer:
                c_v = self.connections[c]
                if c_v.wmin < 0:
                    # increase reward by the minimum weight * number of outgoing connections
                    reward += (c_v.wmin * c_v.target.n)
        # reward is now bigger than the cumulative value of all "new" outputs from the layer in this timestep
        return reward

    def forward_qa(self, encoding_ae: int, encoding_ai: int, reward_ai: float, num_repeats: int):
        # language=rst
        """
        Runs a single simulation step.
        Only works for batch_size = 1 and DiehlAndCook-Network.
        """
        l_ae_v = self.layers['Ae']
        l_ai_v = self.layers['Ai']

        # Decay voltages.
        l_ae_v.v = l_ae_v.decay * (l_ae_v.v - l_ae_v.rest) + l_ae_v.rest
        l_ai_v.v = l_ai_v.decay * (l_ai_v.v - l_ai_v.rest) + l_ai_v.rest
        # input layer does not have voltage
        # and adaptive thresholds
        l_ae_v.theta *= l_ae_v.theta_decay

        # start_before = clock.time()
        # get tensors as lists for quicker access
        weights_x_ae = self.connections[('X', 'Ae')].w.tolist()
        weights_ae_ai = self.connections[('Ae', 'Ai')].w.tolist()
        weights_ai_ae = self.connections[('Ai', 'Ae')].w.tolist()
        refrac_count_ae = l_ae_v.refrac_count[0].tolist()
        refrac_count_ai = l_ai_v.refrac_count[0].tolist()
        thresh_ae = l_ae_v.thresh.item()
        theta = l_ae_v.theta.tolist()
        thresh_ai = l_ai_v.thresh.item()
        v_ae = l_ae_v.v[0].tolist()
        v_ai = l_ai_v.v[0].tolist()
        n_ae = l_ae_v.n
        n_ai = l_ai_v.n
        # input spikes from layers
        s_view_x = self.layers['X'].s.float().view(self.layers['X'].s.size(0), -1)[0].tolist()
        s_view_ai = self.layers['Ai'].s.float().view(self.layers['Ai'].s.size(0), -1)[0].tolist()
        s_view_ae = self.layers['Ae'].s.float().view(self.layers['Ae'].s.size(0), -1)[0].tolist()

        # prepare Quantum Annealing and get inputs
        qubo = {}  # shape: (number of neurons * number of layers)^2
        # for storing weighted inputs
        inputs_ae = [0] * n_ae
        inputs_ai = [0] * n_ai

        # non-zero indices in input spikes
        ind_x = [i for i, e in enumerate(s_view_x) if e != 0]
        ind_ai = [i for i, e in enumerate(s_view_ai) if e != 0]
        ind_ae = [i for i, e in enumerate(s_view_ae) if e != 0]

        # go through layers

        # Layer X of Input-Neurons: nothing to do

        # Layer Ae of excitatory Neurons
        # determine indices of neurons that are not in refractory period
        refrac_count_ae_0 = [i for i, e in enumerate(refrac_count_ae) if e == 0]

        # Could spike -> need constraints
        for node_ae in refrac_count_ae_0:
            nr_ae = node_ae + encoding_ae
            # diagonal
            # = threshold + adaptive threshold theta - membrane potential
            qubo[(nr_ae, nr_ae)] = thresh_ae + theta[node_ae] - v_ae[node_ae]  # + NUDGE?

            # off-diagonal: go through input spikes from layers / connections
            # Inputs from layer X (connection X->Ae)
            # we work in the upper triangular matrix, connection goes from row to column
            for node_x in ind_x:
                inp = s_view_x[node_x] * weights_x_ae[node_x][node_ae]
                # input layer X is first layer -> node number equals position in qubo
                qubo[(node_x, nr_ae)] = -1 * inp
                inputs_ae[node_ae] += inp

            # Inputs from layer Ai (connection Ai->Ae)
            # we work in the upper triangular matrix, connection goes from column to row
            # actual connections (where weights ≠ 0) are only where node_ae ≠ node_ai
            for node_ai in ind_ai:
                if not node_ai == node_ae:
                    # as spikes in network are always 1
                    # we can skip multiplication of weight with s_view_ai[node_ai]
                    inp = weights_ai_ae[node_ai][node_ae]
                    column_nr = node_ai + encoding_ai
                    qubo[(nr_ae, column_nr)] = -1 * inp
                    inputs_ae[node_ae] += inp
        # other excitatory neurons:
        # reward can be omitted for excitatory neurons -> done here for performance reasons
        # cannot process inputs
        # -> nothing to do

        # Layer Ai of inhibitory Nodes
        # we just need to calculate whether a neuron spikes if it gets any new inputs
        # inputs to this layer can only be where node_ai = node_ae
        # since there are only actual connections (where weights ≠ 0) there
        # so we only need to check there
        for node in ind_ae:
            # Could spike -> needs constraints (making list does not pay off here, since there's always at most
            # one spike in ind_ae -> need to check only once -> cheaper with ‘if’)
            if refrac_count_ai[node] == 0:
                nr_ai = node + encoding_ai
                # diagonal
                # = threshold - membrane potential
                qubo[(nr_ai, nr_ai)] = thresh_ai - v_ai[node]  # + NUDGE?

                # off-diagonal: go through input spikes from layers / connections
                # Inputs from layer Ae (connection Ae->Ai)
                # we work in the upper triangular matrix, connection goes from row to column
                # actual connections (where weights ≠ 0) are only where node_ai = node_ae
                # as spikes in network are always 1
                # we can skip multiplication of weight with s_view_ae[node]
                inp = weights_ae_ai[node][node]
                nr_ae = node + encoding_ae
                qubo[(nr_ae, nr_ai)] = -1 * inp
                inputs_ai[node] += inp

        # neurons that just have spiked need reward to clamp qubit to 1
        # (reward would not harm for neurons in refrac-period that did not spike
        # but easier this way)
        for node_ai in ind_ai:
            nr_ai = node_ai + encoding_ai
            qubo[(nr_ai, nr_ai)] = reward_ai

        #end_before = clock.time()
        #elapsed_before = end_before - start_before
        #print("\n Wall clock time before: %fs" % elapsed_before)

        # call Quantum Annealer or simulator
        if len(qubo) > 1:  # qbsolv can apparently not deal with qubos of length 1
            # start_qb = clock.time()
            # originally num_repeats=40, seems to work well with num_repeats=1, too (-> now default)
            solution = qbs.QBSolv().sample_qubo(qubo, num_repeats=num_repeats, verbosity=-1)
            # end_qb = clock.time()
            # elapsed_qb = end_qb - start_qb
            # print("\n Wall clock time qbsolv: %fs" % elapsed_qb)
            # print("\n Energy of qbsolv-solution: %f" % solution.first.energy) -> return instead
            solution_sample = solution.first.sample
            energy = solution.first.energy
        else:  # if qubo has length 1
            solution_sample = {}
            energy = 0
            for nr in qubo:
                if qubo[nr] < 0:
                    solution_sample[nr] = 1
                    energy = qubo[nr]
                else:
                    print("Error: There should not only be one positive value in qubo.")
                    return None

        #start_after_qb = clock.time()
        # evaluate how much of the qubo is filled (i.e. not zero)
        filled = len(qubo) - [qubo.values()].count(0)

        # Layer Ae
        # write spikes from (first) solution by filtering out 1s from neurons in refractory period
        spikes_ae = [False] * n_ae
        for node in range(n_ae):
            # is not in refractory period (has not just spiked) -> could spike
            if refrac_count_ae[node] == 0:
                if solution_sample[encoding_ae + node] == 1:
                    spikes_ae[node] = True
        spiketensor_ae = torch.tensor([spikes_ae])

        # Integrate inputs into voltage
        # (inputs are zero where in refrac-period)
        l_ae_v.v += torch.tensor([inputs_ae])

        # Decrement refractory counters.
        refrac_ae = l_ae_v.refrac_count  # as a tensor
        l_ae_v.refrac_count = (refrac_ae > 0).float() * (refrac_ae - l_ae_v.dt)

        # Refractoriness, voltage reset, and adaptive thresholds.
        l_ae_v.refrac_count.masked_fill_(spiketensor_ae, l_ae_v.refrac)
        l_ae_v.v.masked_fill_(spiketensor_ae, l_ae_v.reset)
        l_ae_v.theta += l_ae_v.theta_plus * spiketensor_ae.float().sum(0)

        # Choose only a single neuron to spike.
        l_ae_v.s.zero_()  # set all spikes to 0
        # if there are any spikes in spiketensor_ae, set one of these to 1 in l_ae_v.s
        if spiketensor_ae.any():
            _any = spiketensor_ae.view(l_ae_v.batch_size, -1).any(1)
            ind = torch.multinomial(
                spiketensor_ae.float().view(l_ae_v.batch_size, -1)[_any], 1
            )
            _any = _any.nonzero()
            l_ae_v.s.view(l_ae_v.batch_size, -1)[_any, ind] = 1

        # Layer Ai
        # write spikes from (first) solution by filtering out 1s from neurons in refractory period
        spikes_ai = [False] * n_ai
        # here, we can again make use of the fact that only neurons that get any non-zero inputs can spike
        for node in ind_ae:
            # is not in refractory period (has not just spiked) -> could spike
            if refrac_count_ai[node] == 0:
                nr = encoding_ai + node
                if nr in solution_sample:
                    if solution_sample[nr] == 1:
                            spikes_ai[node] = True
        spiketensor_ai = torch.tensor([spikes_ai])
        l_ai_v.s = spiketensor_ai

        # Integrate inputs into voltage
        # (inputs are zero where in refrac-period)
        l_ai_v.v += torch.tensor([inputs_ai])

        # Decrement refractory counters.
        refrac_ai = l_ai_v.refrac_count
        l_ai_v.refrac_count = (refrac_ai > 0).float() * (refrac_ai - l_ai_v.dt)

        # Refractoriness, voltage reset, and adaptive thresholds.
        l_ai_v.refrac_count.masked_fill_(spiketensor_ai, l_ai_v.refrac)
        l_ai_v.v.masked_fill_(spiketensor_ai, l_ai_v.reset)

        #end_after_qb = clock.time()
        #elapsed_after_qb = end_after_qb - start_after_qb
        #print("\n Wall clock time after: %fs" % elapsed_after_qb)
        return energy, filled
    # end of forward_qa


    def run(
        self, inputs: Dict[str, torch.Tensor], time: int, num_repeats: int, one_step=False, **kwargs
    ):
        # language=rst
        """
        Simulate network for given inputs and time. Adjusted for using QA

        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[time, batch_size, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.
        :param int num_repeats: Number of iterations the QA-simulator runs the problem

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from bindsnet_qa.network import Network
            from bindsnet_qa.network.nodes import Input
            from bindsnet_qa.network.monitors import Monitor

            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

            # Run network simulation.
            network.run(inputs={'I' : spikes}, time=500)

            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        """
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        masks = kwargs.get("masks", {})

        # Effective number of timesteps.
        timesteps = int(time / self.dt)
        qb_solv_energies = []
        filled = []

        # start = clock.time()
        # calculate possible Quantum Annealing penalties / rewards once
        #for all inhibitory layers (can be omitted for excitatory layers for performance reasons)
        # -> here only layer Ai
        reward_ai = self.reward_inhibitory(layer='Ai')

        # calculate encoding for QUBO once to remember at which row_nr / column_nr which layer starts
        encoding_ae = 784  # Number of nodes in Input-Layer when learning MNIST
        encoding_ai = 784 + self.layers['Ae'].n
        # end = clock.time()
        # elapsed = end - start
        # print("\n Wall clock time avoidable: %fs" % elapsed)

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):

            # for l in inputs: -> only one, namely X
            # compute spikes of Input-layer X right away
            self.layers['X'].forward(x=inputs['X'][t])

            # forward-step with quantum annealing
            # start_network = clock.time()
            qb_solv_energy, filled_one = self.forward_qa(encoding_ae, encoding_ai, reward_ai,  num_repeats=num_repeats)
            # end_forward = clock.time()
            # elapsed_forward = end_forward - start_network
            # print("\n Wall clock time forward_qa(): %fs" % elapsed_forward)
            # start_append = clock.time()
            qb_solv_energies.append(qb_solv_energy)
            filled.append(filled_one)
            # end_after_append = clock.time()
            # elapsed_after_append = end_after_append - start_append
            # print("\n Wall clock time append(): %fs" % elapsed_after_append)

            # for l in self.layers: -> happens just for layer Ae, if at all

            # Clamp neurons to spike.
            # -> happens just for layer Ae, if at all
            clamp = clamps.get('Ae', None)
            if clamp is not None:
                self.layers['Ae'].s[:, clamp] = 1

            # Run synapse updates.
            for c in self.connections:
                self.connections[c].update(
                    mask=masks.get(c, None), learning=self.learning, **kwargs
                )

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()

        # Re-normalize connections.
        for c in self.connections:
            self.connections[c].normalize()

        return qb_solv_energies, filled

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Reset state variables of objects in network.
        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        for monitor in self.monitors:
            self.monitors[monitor].reset_state_variables()

    def train(self, mode: bool = True) -> "torch.nn.Module":
        # language=rst
        """
        Sets the node in training mode.

        :param mode: Turn training on or off.

        :return: ``self`` as specified in ``torch.nn.Module``.
        """
        self.learning = mode
        return super().train(mode)
