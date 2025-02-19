import numpy as np
import torch
import torch.nn as nn
import networkx as nx

from reinforcement.sample import Explore
from reinforcement.gen_data import gen_data
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


def policy_loss(output: torch.tensor, reward: torch.tensor):
    #sum = torch.log(output).sum(axis=1)
    sum = output.sum(axis=1)
    return (torch.mean(sum * reward))
