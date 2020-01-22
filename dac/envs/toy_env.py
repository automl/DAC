"""
Gym environments containing the White-Box benchmarks discussed in

.. moduleauthor:: André Biedenkapp, H. Furkan Bozkurt
"""
import csv
import itertools
import logging
import os
import sys
from typing import List, Tuple

import gym
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from gym import Env, spaces, wrappers
from scipy.stats import truncnorm


class ADPBench(Env):
    """
    Abstract class as both toy environments implement mostly the same behavior
    """

    # ####################### Not needed Methods ##################### #
    def step(self, action) -> Tuple[List[int], float, bool, None]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def __init__(self, n_steps: int, n_actions: int, seed: int=0) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)
        # self.observation_space = spaces.Discrete(n_steps)
        self.observation_space = spaces.Box(low=np.array([-1 for _ in range(n_steps)]),
                                            high=np.array([n_actions for _ in range(n_steps)]),
                                            dtype=np.float32)
        self.rng = np.random.RandomState(seed)
        self._c_step = 0
        self.logger = None

    def close(self) -> bool:
        return True

    def render(self, mode: str='human', close: bool=True) -> None:
        if mode != 'human':
            raise NotImplementedError
        pass

# Instance IDEA 1: shift luby seq -> feat is sum of skipped action values
# Instance IDEA 2: "Wiggle" luby i.e. luby(t + N(0, 0.1)) -> feat is sampled value
class LubyTime(ADPBench):
    """
    Luby "cyclic" benchmark
    """

    def __init__(self,
                 min_steps: int=2**3,
                 max_steps: int=2**6,
                 seed: int=0,
                 hist_len: int=5,
                 fuzzy: bool=False,
                 instance_mode: int=0,
                 instance_feats: str=None,
                 noise_sig: float=1.5
                 ) -> None:
        n_actions = int(np.log2(max_steps))
        super().__init__(n_steps=min_steps, n_actions=n_actions, seed=seed)
        self.reward_range = (-1, 0)
        self._hist_len = hist_len
        self._ms = max_steps
        self._mi = min_steps
        self._state = np.array([-1 for _ in range(self._hist_len + 1)])
        self._r = 0
        self._genny = luby_gen(1)
        self._next_goal = next(self._genny)
        # Generate luby sequence up to 2*max_steps + 2 as mode 1 could potentially shift up to max_steps
        self.__seq = np.log2([next(luby_gen(i)) for i in range(1, 2*max_steps + 2)])
        self._jenny_i = 1
        self._fuzz = fuzzy
        self.__mode = instance_mode  # 0 -> No instances, 1, 1 -> IDEA 1, 2 -> IDEA 2, 3 -> 1+2
        additional_feats = 1  # time-step is default additional feature
        if instance_mode in [1, 2]:  # each mode adds one instance feature
            additional_feats += 1
        elif instance_mode == 3:
            additional_feats += 2
        self.__n_feats = additional_feats
        self.observation_space = spaces.Box(
            low=np.array([-1 for _ in range(self._hist_len + additional_feats)]),
            high=np.array([2**max(self.__seq + 1) for _ in range(self._hist_len + additional_feats)]),
            dtype=np.float32)
        self.logger = logging.getLogger(self.__str__())
        self.noise_sig = noise_sig

        self._start_dist = None
        self._sticky_dis = None
        self._sticky_shif = 0
        self._start_shift = 0
        self.__lower, self.__upper = 0, 0
        self.__error = 0
        self._inst_feat_dict = {}
        self._inst_id = None
        if instance_mode and instance_feats:  # we have a training set which we can load here
            with open(instance_feats, 'r') as fh:
                reader = csv.DictReader(fh)
                if instance_mode == 3:
                    for row in reader:
                        self._inst_feat_dict[int(row['ID'])] = [float(row['start']), float(row['sticky'])]
                elif instance_mode == 2:
                    for row in reader:
                        self._inst_feat_dict[int(row['ID'])] = [0, float(row['sticky'])]
                elif instance_mode == 1:
                    for row in reader:
                        self._inst_feat_dict[int(row['ID'])] = [float(row['start']), 0]
                self._inst_id = -1
        elif instance_mode > 0:  # we don't have an instance set so we sample from a distribution
            sig = .25 * max_steps
            self._start_dist = truncnorm((0 - min_steps) / sig, (max_steps - min_steps) / sig, loc=min_steps, scale=sig)
            mu, sigma = 0, .15
            self.__lower, self.__upper = -.49999, .49999
            self._sticky_dis = truncnorm((self.__lower - mu) / sigma, (self.__upper - mu) / sigma, loc=mu, scale=sigma)

    def step(self, action: int):
        """Function to interact with the environment.
            Args:
            action (int): one of [1, 2, 4, 8, 16, 32, 64, 128]/[0, 1, 2, 3, 4, 5, 6, 7]

        Returns:
            next_state (List[int]):  Next state observed from the environment.
            reward (float):
            done (bool):  Specifies if environment is solved.
            info (None):
        """
        self._c_step += 1
        prev_state = self._state.copy()
        if action == self._next_goal:
            self._r = 0  # we don't want to allow for exploiting large rewards by tending towards long sequences
        else:  # mean and var chosen s.t. ~1/4 of rewards are positive
            self._r = -1 if not self._fuzz else self.rng.normal(-1, self.noise_sig)
            diff = abs(self._next_goal - action)
            self.n_steps = min(self._ms, self.n_steps + diff)
        done = self._c_step >= self.n_steps

        if self.__error < self.__lower:  # needed to avoid too long sequences of sticky actions
            self.__error += np.abs(self.__lower)
        elif self.__error > self.__upper:
            self.__error -= np.abs(self.__upper)
        self._jenny_i += 1
        self.__error += self._sticky_shif

        # next target in sequence at step luby_t is determined by the current time step (jenny_i), the start_shift
        # value and the sticky error. Additive sticky error leads to sometimes rounding to the next time_step and
        # thereby repeated actions. With check against lower/upper we reset the sequence to the correct timestep in
        # the t+1 timestep.
        luby_t = max(1, int(np.round(self._jenny_i + self._start_shift + self.__error)))
        self._next_goal = self.__seq[luby_t - 1]
        if self._c_step - 1 < self._hist_len:
            self._state[(self._c_step-1)] = action
        else:
            self._state[:-self.__n_feats - 1] = self._state[1:-self.__n_feats]
            self._state[-self.__n_feats - 1] = action
        self._state[-self.__n_feats] = self._c_step - 1
        next_state = self._state if not done else prev_state
        self.logger.debug("i: (s, a, r, s') / %+5d: (%s, %d, %5.2f, %2s)     g: %3d  l: %3d", self._c_step-1,
                          str(prev_state),
                          action, self._r, str(next_state),
                          int(self._next_goal), self.n_steps)
        return np.array(next_state), self._r, done, {}

    def reset(self) -> List[int]:
        """
        Returns:
            next_state (int):  Next state observed from the environment.

        """
        self._c_step = 0
        self._r = 0
        self.n_steps = self._mi

        if self.__mode > 0:
            if self._inst_feat_dict:
                self._inst_id = (self._inst_id + 1) % len(self._inst_feat_dict)
                self._start_shift = self._inst_feat_dict[self._inst_id][0]
                self._sticky_shif = self._inst_feat_dict[self._inst_id][1]
            else:
                if self.__mode == 1:  # extremely heterogeneous
                    self._start_shift = int(np.round(self._start_dist.rvs()))
                    self._sticky_shif = 0
                elif self.__mode == 2:  # very homogeneous
                    self._sticky_shif = np.around(self._sticky_dis.rvs(), decimals=2)
                    self._start_shift = 0
                else:  # heterogeneous sets with homogeneous subsets
                    self._start_shift = int(np.round(self._start_dist.rvs()))
                    self._sticky_shif = np.around(self._sticky_dis.rvs(), decimals=2)

        self.__error = 0 + self._sticky_shif
        self._jenny_i = 1
        luby_t = max(1, int(np.round(self._jenny_i + self._start_shift + self.__error)))
        self._next_goal = self.__seq[luby_t - 1]
        self.logger.debug("i: (s, a, r, s') / %+5d: (%2d, %d, %5.2f, %2d)     g: %3d  l: %3d", -1, -1, -1, -1, -1,
                          int(self._next_goal), self.n_steps)
        self._state = [-1 for _ in range(self._hist_len + self.__n_feats)]
        if self.__mode == 1:
            self._state[-1] = self._start_shift
        elif self.__mode == 2:
            self._state[-1] = self._sticky_shif
        elif self.__mode == 3:
            self._state[-2] = self._start_shift
            self._state[-1] = self._sticky_shif
        return np.array(self._state)


def luby_gen(i):
    for k in range(1, 33):
        if i == ((1 << k) - 1):
            yield 1 << (k-1)
    for k in range(1, 9999):
        if 1 << (k - 1) <= i < (1 << k) - 1:
            for x in luby_gen(i - (1 << (k-1)) + 1):
                yield x


class SigmoidMultiActMultiValAction(ADPBench):
    """
    Sigmoid reward
    """

    def _sig(self, x, scaling, inflection):
        """ Simple sigmoid """
        return 1 / (1 + np.exp(-scaling * (x - inflection)))

    def __init__(self,
                 n_steps: int=10,
                 n_actions: int=2,
                 action_vals: tuple=(5, 10),
                 seed: bool=0,
                 noise: float=0.0,
                 instance_feats: str=None,
                 slope_multiplier: float=2
                 ) -> None:
        super().__init__(n_steps=n_steps, n_actions=n_actions, seed=seed)
        assert self.n_actions == len(action_vals), (
            f'action_vals should be of length {self.n_actions}.')
        self.shifts = [self.n_steps / 2 for _ in action_vals]
        self.slopes = [-1 for _ in action_vals]
        self.reward_range = (0, 1)
        self._c_step = 0
        self.noise = noise
        self.slope_multiplier = slope_multiplier
        self.action_vals = action_vals
        # budget spent, inst_feat_1, inst_feat_2
        # self._state = [-1 for _ in range(3)]
        # self.action_space = spaces.MultiDiscrete(action_vals)
        self.action_space = spaces.Discrete(int(np.prod(action_vals)))
        self.action_mapper = {}
        for idx, prod_idx in zip(range(np.prod(action_vals)),
                                       itertools.product(*[np.arange(val) for val in action_vals])):
            self.action_mapper[idx] = prod_idx
        self.observation_space = spaces.Box(
            low=np.array([-np.inf for _ in range(1 + n_actions * 3)]),
            high=np.array([np.inf for _ in range(1 + n_actions * 3)]))
        self.logger = logging.getLogger(self.__str__())
        self._prev_state = None
        self._inst_feat_dict = {}
        self._inst_id = None
        if instance_feats:
            with open(instance_feats, 'r') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    self._inst_feat_dict[int(row['ID'])] = [float(shift) for shift in row['shifts'].split(",")] + \
                                                           [float(slope) for slope in row['slopes'].split(",")]
                self._inst_id = -1

    def step(self, action: int):
        action = self.action_mapper[action]
        assert self.n_actions == len(action), (
            f'action should be of length {self.n_actions}.')

        val = self._c_step
        r = [1 - np.abs(self._sig(val, slope, shift) - (act / (max_act - 1)))
            for slope, shift, act, max_act in zip(
                self.slopes, self.shifts, action, self.action_vals
            )]
        r = np.clip(np.prod(r), 0.0, 1.0)
        remaining_budget = self.n_steps - self._c_step

        next_state = [remaining_budget]
        for shift, slope in zip(self.shifts, self.slopes):
            next_state.append(shift)
            next_state.append(slope)
        next_state += action
        prev_state = self._prev_state

        self.logger.debug("i: (s, a, r, s') / %d: (%s, %d, %5.2f, %2s)", self._c_step-1, str(prev_state),
                          action, r, str(next_state))
        self._c_step += 1
        self._prev_state = next_state
        return np.array(next_state), r, self._c_step >= self.n_steps, {}

    def reset(self) -> List[int]:
        if self._inst_feat_dict:
            self._inst_id = (self._inst_id + 1) % len(self._inst_feat_dict)
            self.shifts = self._inst_feat_dict[self._inst_id][:self.n_actions]
            self.slopes = self._inst_feat_dict[self._inst_id][self.n_actions:]
        else:
            self.shifts = self.rng.normal(self.n_steps/2, self.n_steps/4, self.n_actions)
            self.slopes = self.rng.choice([-1, 1], self.n_actions) * self.rng.uniform(size=self.n_actions) * self.slope_multiplier
        self._c_step = 0
        remaining_budget = self.n_steps - self._c_step
        next_state = [remaining_budget]
        for shift, slope in zip(self.shifts, self.slopes):
            next_state.append(shift)
            next_state.append(slope)
        next_state += [-1 for _ in range(self.n_actions)]
        self._prev_state = None
        self.logger.debug("i: (s, a, r, s') / %d: (%2d, %d, %5.2f, %2d)", -1, -1, -1, -1, -1)
        return np.array(next_state)

    def render(self, mode: str, close: bool=True) -> None:
        if mode == 'human' and self.n_actions == 2:
            plt.ion()
            plt.show()
            plt.cla()
            steps = np.arange(self.n_steps)
            self.data = self._sig(steps, self.slopes[0], self.shifts[0]) * \
                        self._sig(steps, self.slopes[1], self.shifts[1]).reshape(-1, 1)

            plt.imshow(
                self.data,
                extent=(0, self.n_steps - 1, 0, self.n_steps - 1),
                interpolation='nearest', cmap=cm.plasma)
            plt.axvline(x=self._c_step, color='r', linestyle='-', linewidth=2)
            plt.axhline(y=self._c_step, color='r', linestyle='-', linewidth=2)

            plt.draw()
            plt.pause(0.005)
