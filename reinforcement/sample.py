import numpy as np
import torch
from trails.randomwalk.randomwalk import RandomWalk as RandomWalk
import trails.randomwalk.first_state as first_state
import trails.randomwalk.keep_walking as keep_walking
import trails.randomwalk.next_state as next_state
import trails.utils as utils

class Explore(RandomWalk):
    def __init__(self, adjacency_matrix, state_properties, policy):
        super().__init__(adjacency_matrix, state_properties)

        self.policy = policy

    def sample(self,
             n_samples,
             n_steps,
             n_groups,
             f_next_state=next_state.random,
             f_keep_walking=keep_walking.init_fixed(10),
             f_first_state=first_state.random):

        # initialize container for walks
        walks = []
        trajectories = []
        group_assignment_p = np.zeros((n_samples, n_steps + 1, n_groups))
        groups = torch.empty((n_samples, n_steps + 1, 1))
        log_probs = torch.empty((n_samples, n_steps + 1))

        # do the walking
        for walker in range(n_samples):

            # set the first state for the walker
            first_state = f_first_state(walker, self.adjacency_matrix, self.state_properties)
            walk = [first_state]
            second_state = next_state.random(walk, walker, self.adjacency_matrix, self.state_properties)
            walk = [second_state]           # overwrite the walk to begin at "first" transition, first state then only known to the trajectory
            trajectory = [np.array([first_state, second_state])]

            j = 0
            # walk the walk
            while f_keep_walking(walk, walker, self.adjacency_matrix, self.state_properties):

                # get next state
                next_state1, output, group, log_prob = f_next_state(trajectory, walk, walker, self.adjacency_matrix, self.state_properties)

                # quit if we could not determine the next state
                if next_state1 is None:
                    break
                else:
                    # append the new state to the walk
                    curr_state = walk[-1]
                    trajectory.append(np.array([curr_state, next_state1]))
                    walk.append(next_state1)

                    group_assignment_p[walker, j, :] = output
                    groups[walker, j] = int(group)
                    log_probs[walker, j] = log_prob

                    j += 1

            # prob for last transition
            _, output, group, log_prob = f_next_state(trajectory, walk, walker, self.adjacency_matrix,
                                                      self.state_properties)
            #output = self.policy.get_group_prediction(trajectory[-1]).detach().numpy()
            group_assignment_p[walker, j, :] = output
            groups[walker, j] = int(group)
            log_probs[walker, j] = log_prob
            # log walk
            walks.append((walker, walk))
            #trajectories.append((walker, trajectory))
            trajectories.append(trajectory)

        # return walks
        return walks, trajectories, group_assignment_p, groups, log_probs
