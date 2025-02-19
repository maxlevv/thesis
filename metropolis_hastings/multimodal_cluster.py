import numpy as np
import pandas as pd
import networkx as nx

from scipy.stats import dirichlet

from trails.mtmc.ml.direct.optimized import log_ml
from trails.mtmc.common import calc_mixed_hypothesis
import trails.transition_probabilities as tp
import trails.randomwalk.randomwalk as rw
import trails.randomwalk.next_state as next_state
import trails.randomwalk.keep_walking as keep_walking
import trails.randomwalk.first_state as first_state
import trails.group_assignment as group_assignment
import trails.utils as utils

from metropolis_hastings.mh_algo import metropolis_hastings

# graph
n_states = 100
p_state_classes = 3 * [1 / 3]
n_state_classes = len(p_state_classes)
graph = nx.barabasi_albert_graph(n_states, int(n_states * 0.10))
adjacency_matrix = np.asarray(nx.to_numpy_array(graph))
state_classes = np.zeros(n_states, dtype=int)
for i in range(n_states):
    state_classes[i] = np.arange(n_state_classes, dtype=int)[np.mod(i, n_state_classes)]

# hypotheses
tp_group_homo = tp.group_homo(adjacency_matrix, state_classes)
hyp_groups_homo = tp_group_homo
next_homo = next_state.init_grouped_matrix( \
    group_assignment.walker, \
    tp_group_homo)

hyp_teleport = utils.norm1_2d(np.ones((n_states, n_states)))
hyp_links = utils.norm1_2d(adjacency_matrix)
hyp_red = utils.norm1_2d(tp_group_homo[0])
hyp_blue = utils.norm1_2d(tp_group_homo[1])
hyp_green = utils.norm1_2d(tp_group_homo[2])

# hypotheses (mixed colours)

hyp_rb = utils.norm1_2d(hyp_red + hyp_blue)
hyp_rg = utils.norm1_2d(hyp_red + hyp_green)

hyp4 = np.array([hyp_rb, hyp_rg, hyp_blue, hyp_green])

# walker data
n_random_walkers = 100
n_steps = 10
p_dist = 3 * [1/3]
n_random_walker_classes = len(p_dist)
#random_walker_class_counts = np.random.multinomial(n_random_walkers, p_dist)
random_walker_class_counts = np.array(np.array(p_dist) * n_random_walkers).astype(int)
random_walker_classes = np.repeat(range(n_random_walker_classes), random_walker_class_counts)

r = rw.RandomWalk(adjacency_matrix, state_classes)
walks = r.walk(random_walker_classes, next_homo, keep_walking.init_fixed(n_steps), first_state.random)
transitions = np.concatenate([list(zip(walk[:-1], walk[1:])) for walker, walk in walks])
print(random_walker_class_counts)

# init prior distribution
a = np.array(4*[0.1])
prior = dirichlet(a)

# mixed trails specific function to calculate the values for the current state needed to calculate the acceptance ratio
def calc_values(curr_state, kappa=10000, smoothing=0, n_samples=10):
    group_assignment_p_curr = np.repeat(np.array([curr_state]), n_random_walkers * n_steps, axis=0)
    alpha_curr = calc_mixed_hypothesis(group_assignment_p_curr, hyp4)
    log_likelihood_curr = log_ml(transitions, group_assignment_p_curr, alpha_curr * kappa, smoothing=smoothing, n_samples=n_samples)

    return log_likelihood_curr, prior.pdf(curr_state)

# dirichlet proposal distribution
def dirichlet_proposal(curr_state, alpha_prop=0.8):
    proposed_state = dirichlet.rvs(alpha=4 * [alpha_prop], size=1).reshape(len(curr_state))
    # proposed_state = dirichlet.rvs(alpha=curr_state*concentration_factor, size=1).reshape(len(curr_state))
    # problem: dirichlet samples with zero entries cannot be used as new parameters for the dirichlet proposal dist
    mask = (proposed_state == 0)
    proposed_state[mask] = 0.001
    counts = mask.sum()
    proposed_state[np.argmax(proposed_state)] -= counts * 0.001
    proposal_ratio = dirichlet.pdf(curr_state, 4 * [alpha_prop]) / dirichlet.pdf(proposed_state, 4 * [alpha_prop])
    return proposed_state, proposal_ratio

init_state = np.random.dirichlet(np.ones(4))
samples, burnin_samples, accepted_samples, map_estimate, map_estimate_list = metropolis_hastings(init_state, calc_values, 5000000, 0.5, dirichlet_proposal)

samples_array = np.array(samples)
burnin_samples_array = np.array(burnin_samples)
all_samples = np.concatenate((burnin_samples_array, samples_array), axis=0)

# save as csv
np.savetxt("samples.csv", all_samples, delimiter=",")
pd.DataFrame(accepted_samples).to_csv("accepted_samples.csv")
pd.DataFrame(map_estimate_list).to_csv("map_estimate_list.csv")
