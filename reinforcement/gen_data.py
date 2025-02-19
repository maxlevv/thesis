import numpy as np
import torch
import torch.nn as nn
import networkx as nx

from reinforcement.sample import Explore
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



def gen_data(n_random_walkers, n_steps, adjacency_matrix, state_classes, p_dist, next_step):
    n_random_walker_classes = len(p_dist)

    random_walker_class_counts = np.random.multinomial(n_random_walkers, p_dist)
    random_walker_classes = np.repeat(range(n_random_walker_classes), random_walker_class_counts)

    r = rw.RandomWalk(adjacency_matrix, state_classes)
    return r.walk(random_walker_classes, next_step, keep_walking.init_fixed(n_steps), first_state.random), random_walker_classes


def gen_violet_data():
    return None
