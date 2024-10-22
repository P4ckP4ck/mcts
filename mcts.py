import collections
import math

import numpy as np


class UCTNode:
    def __init__(self, state, move, parent=None, action_size=3):
        self.state = state
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = {}  # Dict[[move, prob], UCTNode]
        self.child_priors = np.zeros([action_size], dtype=np.float32)
        self.child_total_value = np.zeros([action_size], dtype=np.float32)
        self.child_number_visits = np.zeros([action_size], dtype=np.float32)

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move[0]]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move[0]] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move[0]]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move[0]] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * (
                self.child_priors / (1 + self.child_number_visits))

    def best_child(self):
        return np.argmax(self.child_Q() + self.child_U())

    def select_leaf(self, forecast):
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            prob = np.random.choice(range(11), p=forecast[current.state.time])
                                    #p=forecaster.predict(np.expand_dims(calc_time_waves(current.state.time),
                                    #                                    axis=0))[0])
            current = current.maybe_add_child((best_move, prob))
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def maybe_add_child(self, move):
        if move not in self.children:
            self.children[move] = UCTNode(
                self.state.play(move), move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value_estimate
            current = current.parent


def calc_dirichlet(child_priors, use_dirichlet, dir_x=0.75, dir_alpha=1):
    if use_dirichlet:
        priors = dir_x * child_priors[0] + (1 - dir_x) * np.random.dirichlet([dir_alpha * child_priors[0]])
        return priors
    else:
        return child_priors[0]


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.value_estimates = np.zeros([3])


def uct_search(state, num_reads, forecast_timeseries, evaluator_network=None, use_dirichlet=False):
    root = UCTNode(state, move=(0, 0), parent=DummyNode())
    for _ in range(num_reads):
        leaf = root.select_leaf(forecast_timeseries)
        child_priors, value_estimate = evaluator_network.predict(np.expand_dims(leaf.state.state[6:], axis=0))
        leaf.value_estimates = value_estimate[0]
        leaf.expand(calc_dirichlet(child_priors, use_dirichlet))
        leaf.backup(leaf.parent.value_estimates[leaf.move[0]])
    return np.argmax(root.child_number_visits), root


def calc_time_waves(env_time):
    sin_day = np.sin(2 * np.pi * env_time / (24 * 4))
    cos_day = np.cos(2 * np.pi * env_time / (24 * 4))
    sin_week = np.sin(2 * np.pi * env_time / (24 * 4 * 7))
    cos_week = np.cos(2 * np.pi * env_time / (24 * 4 * 7))
    sin_year = np.sin(2 * np.pi * env_time / (24 * 4 * 365))
    cos_year = np.cos(2 * np.pi * env_time / (24 * 4 * 365))
    return np.array([sin_day, cos_day, sin_week, cos_week, sin_year, cos_year])


class StateNode:
    def __init__(self, state, env_time, max_c=25):
        self.state = state
        self.time = env_time
        self.max_c = max_c

    def play(self, move):
        env_time = self.time + 1
        action = move[0]
        state_transition = move[1]
        next_residual = (state_transition - 5) / 10
        if action == 0:
            state_of_charge = self.state[4]
        elif action == 1:
            state_of_charge = (self.state[4] * self.max_c + abs(self.state[5]) * 0.9) / self.max_c
        elif action == 2:
            state_of_charge = (self.state[4] * self.max_c - abs(self.state[5]) * 0.9) / self.max_c
        else:
            raise LookupError("Performed action not found!")

        if state_of_charge < 0:
            state_of_charge = 0
        if state_of_charge > 1:
            state_of_charge = 1

        time_waves = calc_time_waves(env_time)
        state = np.hstack([time_waves, state_of_charge, next_residual])
        return StateNode(state, env_time)
