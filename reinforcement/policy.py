import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import networkx as nx

from torchviz import make_dot

from reinforcement.sample import Explore
from reinforcement.gen_data import gen_data
from reinforcement.loss import policy_loss
import trails.transition_probabilities as tp
import trails.randomwalk.randomwalk as rw
import trails.randomwalk.next_state as next_state
import trails.randomwalk.keep_walking as keep_walking
import trails.randomwalk.first_state as first_state
import trails.randomwalk.transition_matrix as tm
import trails.group_assignment as group_assignment
import trails.mtmc.common as common


import trails.utils as utils
import trails.plot as pl
import trails.mtmc.ml.deterministic.default as deterministic


class PolicyNetwork():
    def __init__(self, adjacency_matrix, output_size: int, hidden_layer_size: int):
        self.input_size = np.unique(adjacency_matrix, return_counts=True)[1][1]
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.adjacency_matrix = adjacency_matrix
        self.states = np.transpose(np.nonzero(self.adjacency_matrix))

        self.create_mapping()
        self.create_policy_network()

    def create_mapping(self):
        self.mapping = {}
        for i in range(self.input_size):
            self.mapping[f"{self.states[i]}"] = i


    def create_policy_network(self):
        self.model = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_layer_size, out_features=self.output_size),
            #nn.Softmax(dim=-1)
        )

    def transitions_to_one_hot(self, transitions):
        one_hot = torch.empty(transitions.shape[0], transitions.shape[1], self.input_size)
        for batch, i in zip(transitions, range(transitions.shape[0])):
            for transition, j in zip(transitions[i], range(transitions.shape[1])):
                one_hot_curr = self.single_transition_to_one_hot(transition)
                one_hot[i, j, :] = one_hot_curr
        return one_hot


    def get_group_prediction(self, transition):
        one_hot_tensor = self.single_transition_to_one_hot(transition)
        output = self.model(one_hot_tensor)
        return output


    def single_transition_to_one_hot(self, transition):
        position = self.mapping[f"{transition}"]
        one_hot_vector = np.zeros(self.input_size)
        one_hot_vector[position] = 1
        one_hot_tensor = torch.as_tensor(one_hot_vector, dtype=torch.float32)
        return one_hot_tensor


    def get_group_assignment_p(self):
        group_assignment = np.zeros((self.input_size, self.output_size))
        for i in range(self.input_size):
            one_hot_input = np.zeros(self.input_size)
            one_hot_input[i] = 1
            one_hot_tensor = torch.as_tensor(one_hot_input, dtype=torch.float32)
            output = self.model(one_hot_tensor)
            #make_dot(output, params=dict(self.model.named_parameters())).render("nn", format="png")
            #print(output)

            output = Categorical(logits=output)

            output = output.probs.clone().detach().numpy()

            #output = output.clone().detach().numpy()
            #print(output)
            group_assignment[i] = output
        return group_assignment


if __name__ == "__main__":
    # graph
    n_states = 20
    p_state_classes = [0.5, 0.5]
    n_state_classes = len(p_state_classes)
    graph = nx.barabasi_albert_graph(n_states, int(n_states * 0.10))

    adjacency_matrix = np.asarray(nx.to_numpy_array(graph))
    state_class_distribution = np.random.multinomial(n_states, p_state_classes)
    state_classes = np.repeat( \
        range(0, n_state_classes), \
        state_class_distribution)

    # transitions/hypotheses
    tp_group_homo = tp.group_homo(adjacency_matrix, state_classes)
    tp_matrices = tp_group_homo
    hyp = tp_group_homo

    # data
    n_random_walkers = 100
    n_steps = 10
    p_dist = [0.8, 0.2]
    next = next_state.init_grouped_matrix(group_assignment.walker, tp_group_homo)
    walks, random_walker_classes = gen_data(n_random_walkers, n_steps, adjacency_matrix, state_classes, p_dist, next)

    def calc_group_assignment_p(group_assignment, n_groups):
        p = np.zeros((len(group_assignment), n_groups))
        p[np.arange(p.shape[0]), group_assignment] = 1
        return p

    walk_sizes = [len(walk) - 1 for walker, walk in walks]
    group_assignment_p_gt = calc_group_assignment_p(np.repeat(random_walker_classes, walk_sizes), 2)
    transitions = np.concatenate([list(zip(walk[:-1], walk[1:])) for walker, walk in walks])

    # policy optimization
    policy_network = PolicyNetwork(adjacency_matrix=adjacency_matrix, output_size=2, hidden_layer_size=50)
    explore = Explore(adjacency_matrix, state_classes, policy_network)
    next_policy = next_state.init_policy_grouped_matrix(next_state.policy_group, tp_group_homo, policy_network)


    # optimizer
    optimizer = optim.SGD(policy_network.model.parameters(), lr=0.05)
    loss = policy_loss

    # train
    n_trajectories = 10
    trajectory_size = 10
    n_groups = tp_matrices.shape[0]

    kappa = 100

    epochs = 30
    for epoch in range(epochs):

        walks, trajectories, group_assignment_p, group, log_probs = explore.sample(n_trajectories, trajectory_size, n_groups, next_policy,
                                                                 keep_walking.init_fixed(trajectory_size), first_state.random)
        trajectories_np = np.array(trajectories)
        trajectories_tensor = torch.tensor(trajectories_np)
        group = group.long().detach().numpy()

        #make_dot(log_probs, params=dict(policy_network.model.named_parameters())).render("nn4", format="png")

        # calc evidence and loss
        group_assignment_probability = policy_network.get_group_assignment_p()      # curr output
        alpha = common.calc_mixed_hypothesis(group_assignment_probability, hyp)     # curr hypothesis


        group_assignment_p_data = np.zeros((transitions.shape[0], n_groups))
        for transition, i in zip(transitions, range(transitions.shape[0])):
            output = policy_network.get_group_prediction(transition)
            output = Categorical(logits=output)
            #output.probs
            output = output.probs.clone().detach().numpy()
            group_assignment_p_data[i, :] = output

        print(group_assignment_p_data)
        #sample_data = policy_network.transitions_to_one_hot(trajectories_np)
        #out = policy_network.model(sample_data)
        #make_dot(out, params=dict(policy_network.model.named_parameters())).render("nn2", format="png")
        #out = out.clone().detach().numpy()
        #select = np.c_[np.logical_not(group), group]
        #select = select == 1
        #out_selected = torch.tensor(out[select].reshape(n_trajectories, trajectory_size + 1)).requires_grad_()
        #make_dot(out_selected, params=dict(policy_network.model.named_parameters())).render("nn3", format="png")

        evidences = np.zeros((trajectories_np.shape[0]))
        for i, (trajectory, group_assignment) in enumerate(zip(trajectories_np, group_assignment_p)):
            evidence = deterministic.log_ml(trajectory, group_assignment, alpha * kappa)
            evidences[i] = evidence


        evidence_baseline = deterministic.log_ml(transitions, group_assignment_p_data, alpha * kappa)
        print(evidence_baseline)

        gt_evidence = deterministic.log_ml(transitions, group_assignment_p_gt,
                                           kappa * (common.calc_mixed_hypothesis(group_assignment_p_gt, hyp)))
        print(gt_evidence)

        reward = torch.as_tensor(evidences).requires_grad_()
        output = loss(log_probs, reward)
        print(output)
        optimizer.zero_grad()
        output.backward()
        #print(list(policy_network.model.parameters())[0].grad)
        #print(list(policy_network.model.parameters())[0])
        optimizer.step()
        #print(list(policy_network.model.parameters())[0])
        #print(list(policy_network.model.parameters())[0].grad)

