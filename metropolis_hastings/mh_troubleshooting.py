import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from scipy.stats import dirichlet, norm, truncnorm, uniform

from trails.mtmc.ml.direct.optimized import log_ml
from trails.mtmc.common import calc_mixed_hypothesis
import trails.transition_probabilities as tp
import trails.randomwalk.randomwalk as rw
import trails.randomwalk.next_state as next_state
import trails.randomwalk.keep_walking as keep_walking
import trails.randomwalk.first_state as first_state
import trails.randomwalk.transition_matrix as tm
import trails.group_assignment as group_assignment
import trails.mtmc.common as common
import trails.utils as utils



def metropolis_hastings_ts(initial_state, value_func, n_samples, burnin, proposal_dist, concentrations):
    n_variables = len(initial_state)
    curr_state = initial_state

    log_likelihood_curr, prior_pdf_value_curr = value_func(curr_state)
    map_estimate = [curr_state, log_likelihood_curr]
    map_estimate_list = [map_estimate]      # for plotting purposes

    samples = []
    accepted_samples = []
    burnin_samples = []
    burnin_idx = int(burnin * n_samples)
    states = []
    log_likelihoods_prop = []
    likelihood_ratios = []
    prior_ratios = []
    acceptance_ratios = []
    ratio_log = []

    for i in range(n_samples):
        # propose state
        prop_state, proposal_ratio = proposal_dist(curr_state)
        states.append([np.round(curr_state, 2), np.round(prop_state, 2)])
        ratios = []
        for concentration in concentrations:
            ratio = ratio_calculator(curr_state, prop_state, concentration)
            ratios.append(np.round(ratio, 2))
        ratio_log.append(ratios)

        log_likelihood_prop, prior_pdf_value_prop = value_func(prop_state)
        log_likelihoods_prop.append(log_likelihood_prop)

        diff = log_likelihood_prop - log_likelihood_curr
        likelihood_ratio = np.exp(diff)
        likelihood_ratios.append(likelihood_ratio)

        prior_ratio = prior_pdf_value_prop / prior_pdf_value_curr
        prior_ratios.append(np.round(prior_ratio,2))

        acceptance_ratio = likelihood_ratio * prior_ratio * proposal_ratio
        acceptance_threshold = np.random.uniform(0, 1)
        acceptance_ratios.append(np.round(acceptance_ratio, 2))

        if acceptance_ratio > acceptance_threshold:
            curr_state = prop_state
            accepted_samples.append([curr_state, i])

            # keep track of state with highest log-likelihood
            if diff > 0:
                map_estimate = [prop_state, log_likelihood_prop]
            log_likelihood_curr, prior_pdf_value_curr = value_func(curr_state)

        map_estimate_list.append(map_estimate)
        if i >= burnin_idx:
            samples.append(curr_state)
        else:
            burnin_samples.append(curr_state)

    return samples, burnin_samples, accepted_samples, map_estimate, map_estimate_list, states, log_likelihoods_prop, likelihood_ratios, prior_ratios, ratio_log, acceptance_ratios


def dirichlet_proposal(curr_state, concentration_factor=100, c2=100):
    proposed_state = dirichlet.rvs(alpha=curr_state*concentration_factor, size=1).reshape(len(curr_state))
    # problem: dirichlet samples with zero entries cannot be used as new parameters for the dirichlet proposal dist
    mask = (proposed_state == 0)
    proposed_state[mask] = 0.001
    counts = mask.sum()
    proposed_state[np.argmax(proposed_state)] -= counts * 0.001

    proposal_ratio = dirichlet.pdf(curr_state, proposed_state * c2) / dirichlet.pdf(proposed_state, curr_state * c2)
    return proposed_state, proposal_ratio


def ratio_calculator(curr_state, proposed_state, concentration_factor):
    proposal_ratio = dirichlet.pdf(curr_state, proposed_state * concentration_factor) / dirichlet.pdf(
        proposed_state, curr_state * concentration_factor)
    return proposal_ratio

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
    hyp_green = utils.norm1_2d(tp_group_homo[2])

    # hypotheses (mixed colours)
    hyp_rb = utils.norm1_2d(hyp_red + hyp_blue)
    hyp_rg = utils.norm1_2d(hyp_red + hyp_green)

    hyp4 = np.array([hyp_rb, hyp_rg, hyp_blue, hyp_green])

    # hypotheses (groups)
    hyp_groups_homo = tp_group_homo
    hyp_groups_homo_weighted = tp_group_homo_weighted

    # next_step functions for different walker types to determine their next move
    next_random = next_state.init_matrix(tp_random)
    next_links = next_state.init_matrix(tp_links)
    next_homo = next_state.init_grouped_matrix(group_assignment.walker, tp_group_homo)
    next_memory = next_state.init_grouped_matrix(group_assignment.memory,
                                                 [tp_links, tp_group_homo[0], tp_group_homo[1]])
    next_homo_weighted = next_state.init_grouped_matrix(group_assignment.walker, tp_group_homo_weighted)
    next_memory_weighted = next_state.init_grouped_matrix(group_assignment.memory, [tp_links, tp_group_homo_weighted[0],
                                                                                    tp_group_homo_weighted[1]])


    # violet walkers: we consider violet walkers, where each transition based on the walkers inconsistency
    def violet_group(walk, walker, adjacency_matrix, state_properties):
        group = np.random.choice(len(walker), p=walker)
        return group


    next_violet = next_state.init_grouped_matrix(violet_group, tp_group_homo_weighted)

    # walker data
    n_random_walkers = 100
    n_steps = 10
    p_dist = 3 * [1 / 3]
    n_random_walker_classes = len(p_dist)
    random_walker_class_counts = np.random.multinomial(n_random_walkers, p_dist)
    random_walker_class_counts = np.array(np.array(p_dist) * n_random_walkers).astype(int)
    random_walker_classes = np.repeat(range(n_random_walker_classes), random_walker_class_counts)

    r = rw.RandomWalk(adjacency_matrix, state_classes)
    walks = r.walk(random_walker_classes, next_homo, keep_walking.init_fixed(n_steps), first_state.random)
    transitions = np.concatenate([list(zip(walk[:-1], walk[1:])) for walker, walk in walks])


    # mixed trails specific function to calculate the values for the current state needed to calculate the acceptance ratio
    def calc_values(curr_state, kappa=10000, smoothing=1, n_samples=10):
        group_assignment_p_curr = np.repeat(np.array([curr_state]), n_random_walkers * n_steps, axis=0)
        alpha_curr = calc_mixed_hypothesis(group_assignment_p_curr, hyp4)
        log_likelihood_curr = log_ml(transitions, group_assignment_p_curr, alpha_curr * kappa, smoothing=smoothing,
                                     n_samples=n_samples)
        return log_likelihood_curr, prior.pdf(curr_state)


    # dirichlet proposal distribution
    def dirichlet_proposal(curr_state, concentration_factor=1, c2=1):
        proposed_state = dirichlet.rvs(alpha=4 * [0.5], size=1).reshape(len(curr_state))
        # problem: dirichlet samples with zero entries cannot be used as new parameters for the dirichlet proposal dist
        mask = (proposed_state == 0)
        proposed_state[mask] = 0.001
        counts = mask.sum()
        proposed_state[np.argmax(proposed_state)] -= counts * 0.001
        proposal_ratio = dirichlet.pdf(curr_state, proposed_state * c2) / dirichlet.pdf(proposed_state, curr_state * c2)
        return proposed_state, proposal_ratio



    ratios = [1, 10, 20, 30, 50, 80, 100]
    # init prior distribution
    a = np.array(4 * [0.3])
    prior = dirichlet(a)

    # Run Metropolis Hastings
    init_state = np.random.dirichlet(np.ones(4))
    init_state = np.array([0.63, 0.01, 0.01, 0.35])
    print(init_state)
    samples, burnin_samples, accepted_samples, map_estimate, map_estimate_list, log, ratio_log = metropolis_hastings(init_state,
                                                                                                          calc_values,
                                                                                                          100, 0.5,
                                                                                                          dirichlet_proposal,ratios)
    print(map_estimate)
    print(len(accepted_samples))
    print(ratio_log)
    samples_array = np.array(samples)
    burnin_samples_array = np.array(burnin_samples)

    print(log)

    group_assignment_p_curr = np.repeat(np.array([[0.66, 0, 0, 0.34]]), n_random_walkers * n_steps, axis=0)
    alpha_curr = calc_mixed_hypothesis(group_assignment_p_curr, hyp4)
    log_likelihood_maximum = log_ml(transitions, group_assignment_p_curr, alpha_curr * 10000, smoothing=1, n_samples=10)
    print(log_likelihood_maximum)

    for i in range(len(np.concatenate((samples_array, burnin_samples_array)))):
        diff = log_likelihood_maximum - log[i][2]
        print(diff)
        # log[i][2] = diff