import trails.hyptrails as hyptrails
from trails.mtmc.common import *
from scipy.sparse import csr_matrix
import scipy.misc
import scipy.special
import math




import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display
from matplotlib.widgets import Button, Slider
from ipywidgets import interact, IntSlider

from ipywidgets import interact
from mpl_interactions import ipyplot as iplt

import math
from scipy.stats import dirichlet, lognorm, multivariate_normal, beta

from collections import OrderedDict

from trails.mtmc.ml.direct.optimized import log_ml
from trails.mtmc.common import calc_mixed_hypothesis
import trails.transition_probabilities as tp
import trails.randomwalk.randomwalk as rw
import trails.randomwalk.next_state as next_state
import trails.randomwalk.keep_walking as keep_walking
import trails.randomwalk.first_state as first_state
import trails.group_assignment as group_assignment
import trails.mtmc.common as common
import trails.utils as utils
import trails.plot as pl
import trails.mtmc.ml.deterministic.default as deterministic

from metropolis_hastings.mh_algo import metropolis_hastings


def log_ml_2(
        transitions_walk, group_assignment_function, hypothesis, kappa, smoothing=0, n_samples=100):
    """
    Directly samples the marginal likelihood of the MTMC model
    from the corresponding, analytically derived formula.
    In particular the group assignments are sampled.

    This particular implementation takes advantage of the "logsumexp trick",
    which allows to work with log-probabilities hopefully reducing numeric errors.

    Parameters
    ----------
    transitions: ndarray
        Transitions betweens states described by their source and destination state.
        Thus, the shape is: ``(n,2)``,
        where ``n`` is the number of states and
        ``transitions[i,] = [source_state_i, destination_state_i]``.
    group_assignment_p: ndarray
        Group assignment probabilities, i.e.,
        for each transition it holds a probability distribution over groups.
        Thus, the shape is ``(n,g)``,
        where ``n`` is the number of transitions and ``g`` is the number of groups.
    alpha: ndarray
        The dirichlet prior parameters of the model.
        They have the same dimension as the transition probabilities.
        Thus the shape is ``(g,m,m)``,
        where ``g`` is the number of groups and ``m`` is the number of states.
    n_samples: int
        Number of Gibbs runs.
        (Default: 100)
    smoothing: float
        Adds a constant to alpha during calculations.
        Usually, this is used with sparse alpha matrices, i.e.,
        we add the "proto-prior" by setting ``smoothing=1``.
    """

    # derive variables
    n_walker = transitions_walk.shape[0]
    n_groups = hypothesis.shape[0]
    n_states = hypothesis[0].shape[0]

    # initialize samples
    samples = np.zeros(n_samples)

    # transitions
    transitions = np.concatenate(transitions_walk)

    # run sampler
    for i in range(0, n_samples):
        # initialize list for current group assignment
        group_assignments_list = []
        group_assignment_p_list = []

        # sample group assignments
        for j in range(0, n_walker):
            len_walker = len(transitions_walk[j])
            for k in range(0, len_walker):
                # Utilize the cdf of the passed beta distributed random variable
                prob = group_assignment_function.cdf(x=k/(len_walker-1))
                group_assignment_p_list.append([1-prob, 0, prob])    # TODO: trash
                t = np.random.uniform(0, 1)  # threshold
                if t < prob:
                    group = n_groups - 1    # the last group
                else:
                    group = 0           # TODO: this only works for 2 groups!
                group_assignments_list.append(group)

        group_assignments = np.array(group_assignments_list)
        group_assignment_p = np.array(group_assignment_p_list)
        alpha = calc_mixed_hypothesis(group_assignment_p, hypothesis)
        # prepare alpha
        #if len(alpha.shape) > 1:
        alpha = np.array([group_alpha * kappa for group_alpha in alpha])

        # calculate transition counts
        for g in range(0, n_groups):
            selected_transitions = transitions[group_assignments == g, :]
            group_counts = csr_matrix(
                (np.ones(selected_transitions.shape[0]), (selected_transitions[:, 0], selected_transitions[:, 1])),
                (n_states, n_states))
            samples[i] += hyptrails.evidence_markov_matrix(
                n_states,
                group_counts,
                alpha[g],
                smoothing=smoothing)

    return scipy.special.logsumexp(samples) - math.log(n_samples)


if __name__ == "__main__":
    # graph
    n_states = 100
    p_state_classes = 3 * [1 / 3]
    n_state_classes = len(p_state_classes)
    graph = nx.barabasi_albert_graph(n_states, int(n_states * 0.10))
    adjacency_matrix = np.asarray(nx.to_numpy_array(graph))
    state_classes = np.zeros(n_states, dtype=int)
    for i in range(n_states):
        state_classes[i] = np.arange(n_state_classes, dtype=int)[np.mod(i, n_state_classes)]

    plt.close()
    plt.rcParams['figure.figsize'] = 12, 8


    # hypotheses
    tp_group_homo = tp.group_homo(adjacency_matrix, state_classes)
    hyp_groups_homo = tp_group_homo
    next_homo = next_state.init_grouped_matrix( \
        group_assignment.walker, \
        tp_group_homo)

    # transition matrices for walkers

    tp_random = tp.random(adjacency_matrix, state_classes)
    tp_links = tp.links(adjacency_matrix, state_classes)
    tp_group_homo = tp.group_homo(adjacency_matrix, state_classes)
    tp_group_homo_weighted = tp.group_homo_weighted(4, adjacency_matrix, state_classes)

    # hypotheses (single)

    hyp_teleport = utils.norm1_2d(np.ones((n_states, n_states)))
    hyp_links = utils.norm1_2d(adjacency_matrix)
    hyp_red = utils.norm1_2d(tp_group_homo[0])
    hyp_blue = utils.norm1_2d(tp_group_homo[1])

    # hypotheses (groups)

    hyp_groups_homo = tp_group_homo
    hyp_groups_homo_weighted = tp_group_homo_weighted

    hyp_groups_memory = np.array([hyp_links, hyp_groups_homo[0], hyp_groups_homo[1]])
    hyp_groups_memory_weighted = np.array([hyp_links, hyp_groups_homo_weighted[0], hyp_groups_homo_weighted[1]])

    # hypotheses (direct)

    number_of_groups = [2, 3]
    hyp_cart_homo_weighted = tp.expand(hyp_groups_homo_weighted, number_of_groups, 0)
    hyp_cart_memory_weighted = tp.expand(hyp_groups_memory_weighted, number_of_groups, 1)

    hyp_cart_homo = tp.expand(hyp_groups_homo, number_of_groups, 0)
    hyp_cart_memory = tp.expand(hyp_groups_memory, number_of_groups, 1)

    # next_step functions for different walker types to determine their next move

    next_random = next_state.init_matrix( \
        tp_random)

    next_links = next_state.init_matrix( \
        tp_links)

    next_homo = next_state.init_grouped_matrix( \
        group_assignment.walker, \
        tp_group_homo)

    next_memory = next_state.init_grouped_matrix( \
        group_assignment.memory, \
        [tp_links, tp_group_homo[0], tp_group_homo[1]])

    next_homo_weighted = next_state.init_grouped_matrix( \
        group_assignment.walker, \
        tp_group_homo_weighted)

    next_memory_weighted = next_state.init_grouped_matrix( \
        group_assignment.memory, \
        [tp_links, tp_group_homo_weighted[0], tp_group_homo_weighted[1], tp_group_homo_weighted[2]])


    # violet walkers: we consider violet walkers, where each transition based on the walkers inconsistency
    def violet_group(walk, walker, adjacency_matrix, state_properties):
        group = np.random.choice(len(walker), p=walker)
        return group


    next_violet = next_state.init_grouped_matrix(violet_group, tp_group_homo_weighted)


    def time_group(walk, walker, adjacency_matrix, state_properties, n_steps, n_groups):
        k = 2  # exponent
        i = len(walk)
        t = np.random.uniform(0, 1)  # threshold
        if t < np.power((i - 1) / (n_steps - 1), k):
            group = n_groups
        else:
            group = walker
        return group


    n_steps = 10
    n_groups = 3

    tp_time = np.array([tp_group_homo[0], tp_group_homo[1], tp_group_homo[2], tp_links])
    hyp_time = tp_time

    next_time = next_state.init_grouped_matrix_time(time_group, tp_time, n_steps, n_groups)

    # walker data
    n_random_walkers = 100
    n_steps = 10
    p_dist = [1, 0, 0]
    n_random_walker_classes = len(p_dist)
    random_walker_class_counts = np.random.multinomial(n_random_walkers, p_dist)
    random_walker_class_counts = np.array(np.array(p_dist) * n_random_walkers).astype(int)
    random_walker_classes = np.repeat(range(n_random_walker_classes), random_walker_class_counts)

    r = rw.RandomWalk(adjacency_matrix, state_classes)
    walks = r.walk(random_walker_classes, next_time, keep_walking.init_fixed(n_steps), first_state.random)
    transitions_walk = np.array([list(zip(walk[:-1], walk[1:])) for walker, walk in walks])
    transitions = np.concatenate([list(zip(walk[:-1], walk[1:])) for walker, walk in walks])
    print(random_walker_class_counts)


    def multinorm_proposal(curr_state, sigma=np.diag(np.ones(2))):
        proposed_state = multivariate_normal.rvs(mean=curr_state, cov=sigma, size=1).reshape(len(curr_state))
        proposal_ratio = multivariate_normal.pdf(x=curr_state, mean=proposed_state,
                                                 cov=sigma) / multivariate_normal.pdf(x=proposed_state, mean=curr_state,
                                                                                      cov=sigma)
        return proposed_state, proposal_ratio


    def calc_valueZ(curr_state, kappa=10000, smoothing=1, n_samples=10):
        # beta function to be passed to the sample based evaluation of the evidence
        group_assignment_function = beta(a=curr_state[0], b=curr_state[1])
        # pass transitions without collapsing the different walkers -> transitions_walk
        log_likelihood_curr = log_ml_2(transitions_walk, group_assignment_function, hyp_time, kappa,
                                       smoothing=smoothing, n_samples=n_samples)
        return log_likelihood_curr, prior.pdf(curr_state)


    # init prior distribution
    mu = np.array([1, 1])
    prior = multivariate_normal(mean=mu, cov=1)


    init_state = multivariate_normal.rvs(mean=np.ones(2), cov=1, size=1)
    init_state = np.array([1, 1])
    print(init_state)

    samples, burnin_samples, accepted_samples, map_estimate, map_estimate_list = metropolis_hastings(init_state,
                                                                                                     calc_valueZ, 10,
                                                                                                     0.5,
                                                                                                     multinorm_proposal)
    print(map_estimate)
    print(len(accepted_samples))

    samples_array = np.array(samples)
    burnin_samples_array = np.array(burnin_samples)
