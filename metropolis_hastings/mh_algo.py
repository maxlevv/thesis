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



def metropolis_hastings(initial_state, value_func, n_samples, burnin, proposal_dist):
    n_variables = len(initial_state)
    curr_state = initial_state

    log_likelihood_curr, prior_pdf_value_curr = value_func(curr_state)
    map_estimate = [curr_state, log_likelihood_curr]
    map_estimate_list = [map_estimate]      # for plotting purposes

    samples = []
    accepted_samples = []
    burnin_samples = []
    burnin_idx = int(burnin * n_samples)
    #log = []

    for i in range(n_samples):
        # propose state
        prop_state, proposal_ratio = proposal_dist(curr_state)
        log_likelihood_prop, prior_pdf_value_prop = value_func(prop_state)

        diff = log_likelihood_prop - log_likelihood_curr
        likelihood_ratio = np.exp(diff)

        prior_ratio = prior_pdf_value_prop / prior_pdf_value_curr

        acceptance_ratio = likelihood_ratio * prior_ratio * proposal_ratio
        acceptance_threshold = np.random.uniform(0, 1)

        #log.append([curr_state, prop_state, log_likelihood_prop, proposal_ratio, likelihood_ratio, prior_ratio, acceptance_ratio])

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

    return samples, burnin_samples, accepted_samples, map_estimate, map_estimate_list #, log


def dirichlet_proposal(curr_state, concentration_factor=100, c2=100):
    proposed_state = dirichlet.rvs(alpha=curr_state*concentration_factor, size=1).reshape(len(curr_state))
    # problem: dirichlet samples with zero entries cannot be used as new parameters for the dirichlet proposal dist
    mask = (proposed_state == 0)
    proposed_state[mask] = 0.001
    counts = mask.sum()
    proposed_state[np.argmax(proposed_state)] -= counts * 0.001

    proposal_ratio = dirichlet.pdf(curr_state, proposed_state * c2) / dirichlet.pdf(proposed_state, curr_state * c2)
    return proposed_state, proposal_ratio


def dirichlet_proposal2(curr_state, concentration_factor=100, c2=100):
    curr_kappa = curr_state[0]
    curr_state = curr_state[1:]

    scale = 500
    proposed_kappa = truncnorm.rvs(a=-curr_kappa / scale, b=np.inf, loc=curr_kappa, scale=scale)
    # print(proposed_kappa)
    # proposed_kappa = norm.rvs(loc=curr_kappa, scale=scale)
    # proposed_kappa = uniform.rvs(loc=0, scale=scale)

    proposed_state = dirichlet.rvs(alpha=curr_state * concentration_factor, size=1).reshape(len(curr_state))

    # problem: dirichlet samples with zero entries cannot be used as new parameters for the dirichlet proposal dist
    mask = (proposed_state == 0)
    proposed_state[mask] = 0.001
    counts = mask.sum()
    proposed_state[np.argmax(proposed_state)] -= counts * 0.001

    proposal_ratio = (dirichlet.pdf(curr_state, proposed_state * c2) * truncnorm.pdf(curr_kappa,
                                                                                     a=-proposed_kappa / scale,
                                                                                     b=np.inf, loc=proposed_kappa,
                                                                                     scale=scale)) / (
                                 dirichlet.pdf(proposed_state, curr_state * c2) * truncnorm.pdf(proposed_kappa,
                                                                                                a=-curr_kappa / scale,
                                                                                                b=np.inf,
                                                                                                loc=curr_kappa,
                                                                                                scale=scale))
    # proposal_ratio = (dirichlet.pdf(curr_state, proposed_state * c2) * norm.pdf(curr_kappa, loc=proposed_kappa, scale=scale)) / (dirichlet.pdf(proposed_state, curr_state * c2) * norm.pdf(proposed_kappa, loc=curr_kappa, scale=scale))
    # proposal_ratio = dirichlet.pdf(curr_state, proposed_state * c2) / dirichlet.pdf(proposed_state, curr_state * c2)
    proposed_state = np.concatenate((np.array([proposed_kappa]), proposed_state))
    # print(proposal_ratio)
    # print(proposed_state)
    return proposed_state, proposal_ratio

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
    hyp_links = utils.norm1_2d(adjacency_matrix)
    hyp_groups_homo = tp_group_homo
    next_homo = next_state.init_grouped_matrix( \
        group_assignment.walker, \
        tp_group_homo)

    hyp = np.array([hyp_groups_homo[0], hyp_groups_homo[1], hyp_links])

    # walker data
    n_random_walkers = 100
    n_steps = 10
    p_dist = [0.6, 0.3, 0.1]
    n_random_walker_classes = len(p_dist)
    random_walker_class_counts = np.random.multinomial(n_random_walkers, p_dist)
    random_walker_class_counts = np.array(np.array(p_dist) * n_random_walkers).astype(int)
    random_walker_classes = np.repeat(range(n_random_walker_classes), random_walker_class_counts)

    r = rw.RandomWalk(adjacency_matrix, state_classes)
    walks = r.walk(random_walker_classes, next_homo, keep_walking.init_fixed(n_steps), first_state.random)
    transitions = np.concatenate([list(zip(walk[:-1], walk[1:])) for walker, walk in walks])

    # init prior distribution
    a = np.array([1, 1, 1])
    prior = dirichlet(a)

    # mixed trails specific function to calculate the values for the current state needed to calculate the acceptance ratio
    def calc_values(curr_state, kappa=10000, smoothing=0, n_samples=10):
        group_assignment_p_curr = np.repeat(np.array([curr_state]), n_random_walkers * n_steps, axis=0)
        alpha_curr = calc_mixed_hypothesis(group_assignment_p_curr, hyp_groups_homo)
        log_likelihood_curr = log_ml(transitions, group_assignment_p_curr, alpha_curr * kappa, smoothing=smoothing, n_samples=n_samples)

        return log_likelihood_curr, prior.pdf(curr_state)


    def calc_values2(curr_state, smoothing=0, n_samples=10):
        kappa = curr_state[0]
        print(kappa)
        curr_state = curr_state[1:]
        print(curr_state)
        group_assignment_p_curr = np.repeat(np.array([curr_state]), n_random_walkers * n_steps, axis=0)
        mixed_hyp = calc_mixed_hypothesis(group_assignment_p_curr, hyp_groups_homo)
        alpha_curr = np.array([a * kappa for a in mixed_hyp])
        log_likelihood_curr = log_ml(transitions, group_assignment_p_curr, alpha_curr, smoothing=smoothing,
                                     n_samples=n_samples)
        print(log_likelihood_curr)
        return log_likelihood_curr, prior1.pdf(kappa) * prior2.pdf(curr_state)

    # mixed trails specific likelihood function
    def calc_ratio(curr_state, prop_state, kappa=10000, smoothing=0, n_samples=10):
        group_assignment_p_curr = np.repeat(np.array([curr_state]), n_random_walkers * n_steps, axis=0)
        group_assignment_p_prop = np.repeat(np.array([prop_state]), n_random_walkers * n_steps, axis=0)
        alpha_curr = calc_mixed_hypothesis(np.repeat(group_assignment_p_curr, n_random_walkers * n_steps, axis=0), hyp_groups_homo)
        alpha_prop = calc_mixed_hypothesis(np.repeat(group_assignment_p_prop, n_random_walkers * n_steps, axis=0), hyp_groups_homo)
        log_likelihood_curr = log_ml(transitions, group_assignment_p_curr, alpha_curr*kappa, smoothing=smoothing, n_samples=n_samples)
        log_likelihood_prop = log_ml(transitions, group_assignment_p_prop, alpha_prop*kappa, smoothing=smoothing, n_samples=n_samples)

        diff = log_likelihood_prop - log_likelihood_curr
        likelihood_ratio = math.exp(diff)
        prior_ratio = prior.pdf(prop_state) / prior.pdf(curr_state)
        proposal_ratio = dirichlet.pdf(curr_state, prop_state*100)/dirichlet.pdf(prop_state, curr_state*100)

        acceptance_ratio = likelihood_ratio * prior_ratio * proposal_ratio

        return acceptance_ratio


    loc = 1000
    scale = 500
    prior1 = truncnorm(a=-loc / scale, b=np.inf, loc=loc, scale=scale)
    a = np.array([1, 1, 1])
    prior2 = dirichlet(a)

    init_kappa = np.array([100])
    init_state = np.random.dirichlet(np.ones(3))
    init_state = np.concatenate((init_kappa, init_state))

    samples, burnin_samples, accepted_samples, map_estimate, map_estimate_list = metropolis_hastings(init_state, calc_values2, 100, 0.5, dirichlet_proposal2)



    samples_array = np.array(samples)
    burnin_samples_array = np.array(burnin_samples)

    # Plot the samples
    plt.figure(figsize=(8, 6))
    plt.scatter(samples_array[:, 0], samples_array[:, 1], s=10, c='blue', alpha=0.3)
    plt.scatter(burnin_samples_array[:, 0], burnin_samples_array[:, 1], s=10, c='red', alpha=0.3)
    plt.title('Samples from Metropolis-Hastings')
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.grid(True)
    plt.show()

    plt.xlim(0, 1)
    plt.ylim(0, 1)
