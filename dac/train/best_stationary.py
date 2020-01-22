"""
Simple code to brute-force determine the performance of the best static configurations.
"""

from functools import partial
from dac.envs import SigmoidMultiActMultiValAction as SigMV, LubyTime
import json as j
from collections import defaultdict
import os
import argparse
import numpy as np

env_choice = {
    # Luby Benchmark without fuzzy rewards
    'lubyt': LubyTime,                                # no instances
    'lubytho': partial(LubyTime, instance_mode=2),    # homogeneous instances (only some actions differ)
    'lubythe': partial(LubyTime, instance_mode=1),    # heterogeneous instances (sequence is shifted -> less overlap)
    'lubytheho': partial(LubyTime, instance_mode=3),  # combination of both instance sampling strategies (not used)

    # Luby Benchmark with fuzzy rewards              Version used for experiments in paper
    'lubytfuzz': partial(LubyTime, fuzzy=True),
    'lubytfuzzho': partial(LubyTime, fuzzy=True, instance_mode=2),   # used to generate Table 1 & 2 in the paper
    'lubytfuzzhe': partial(LubyTime, fuzzy=True, instance_mode=1),   # used to generate Table 1 & 2 in the paper
    'lubytfuzzheho': partial(LubyTime, fuzzy=True, instance_mode=3),

    # Sigmoid Benchmark
    '1D': partial(SigMV, n_actions=1, action_vals=(2, )),              # Used to generate Figure 2(a)-(c)
    '1D3M': partial(SigMV, n_actions=1, action_vals=(3, )),            # Used to generate Figure 3(a)
    '2D3M': partial(SigMV, n_actions=2, action_vals=(3, 3)),           # Used to generate Figure 3(b)
    '3D3M': partial(SigMV, n_actions=3, action_vals=(3, 3, 3)),        # Used to generate Figure 3(c)
    '5D3M': partial(SigMV, n_actions=5, action_vals=(3, 3, 3, 3, 3)),  # Used to generate Figure 3(d)
}


parser = argparse.ArgumentParser('Determine best stationary config on the benchmarks')
parser.add_argument('--env',
                    choices=list(env_choice.keys()),
                    default='lubyt',
                    help='Which environment to run.')
parser.add_argument('--instance_feature_file', dest='inst_feats',
                    default=None,
                    help='Instance feature file to use for sigmoid environment',
                    type=os.path.abspath)
parser.add_argument('-s', '--seed',
                    default=0,
                    type=int)
parser.add_argument('-S', default=[],
                    nargs='+', dest='seeds',
                    type=int)
parser.add_argument('--reward_variance',
                    default=1.5,
                    type=float,
                    help='Variance of noisy reward signal for lubyt*')
parser.add_argument('--cutoff',
                    default=None,
                    type=int,
                    help='Env max steps')
parser.add_argument('--min_steps',
                    default=None,
                    type=int,
                    help='Env min steps. Only active for the lubyt* environments')

args = parser.parse_args()

if args.reward_variance >= 0 and args.env.startswith('lubyt'):
    env_choice[args.env] = partial(env_choice[args.env], noise_sig=args.reward_variance)
if args.cutoff and not args.env.startswith('lubyt'):
    env_choice[args.env] = partial(env_choice[args.env], n_steps=args.cutoff)
elif args.cutoff:
    env_choice[args.env] = partial(env_choice[args.env], max_steps=args.cutoff)
if args.min_steps and args.env.startswith('lubyt'):
    env_choice[args.env] = partial(env_choice[args.env], min_steps=args.min_steps)

env = env_choice[args.env](seed=args.seed)

# Brute force determine the best stationary configuration possible. Configuration spaces are so small that brute-force
# is the simplest way

try:
    range_ = env._ms  # luby
except AttributeError:
    range_ = env.n_steps  # sigmoid
if args.seeds:
    acts = defaultdict(lambda: list())
    for seed in args.seeds:
        print(seed)
        for action in range(env.action_space.n):
            print('\t', action)
            if args.inst_feats and (args.env.startswith('sigmoid') or args.env.endswith('D') or args.env.endswith('M')):
                env = env_choice[args.env](seed=seed, instance_feats=args.inst_feats)
            elif args.inst_feats and args.env.startswith('lubyt'):
                env = env_choice[args.env](seed=seed, instance_feats=args.inst_feats)
            else:
                env = env_choice[args.env](seed=seed)
            reps = 10
            rewards = []
            if args.inst_feats:
                reps = 100
            for i in range(reps):
                done = False
                _ = env.reset()
                at = 0
                cumulative_reard = 0
                while not done:
                    a = action
                    _, r, done, _ = env.step(a)
                    cumulative_reard += r
                    at += 1
                rewards.append(cumulative_reard)
            acts[action].append(np.mean(rewards))
            env.close()
    vals = []
    stds = []
    for act in acts:
        v = np.mean(acts[act])
        s = np.std(acts[act])
        vals.append(v)
        stds.append(s)
        print(act, v, s)
    print()
    print(np.argmax(vals))
    print('Max: ', np.max(vals), stds[np.argmax(vals)])
else:
    for action in range(env.action_space.n):
        if args.inst_feats and args.env == '1D':
            env = env_choice[args.env](seed=args.seed, instance_feats=args.inst_feats)
        else:
            env = env_choice[args.env](seed=args.seed)
        reps = 10
        rewards = []
        if args.inst_feats:
            reps = 100
        for i in range(reps):
            done = False
            _ = env.reset()
            at = 0
            cumulative_reard = 0
            while not done:
                a = action
                _, r, done, _ = env.step(a)
                cumulative_reard += r
                at += 1
            rewards.append(cumulative_reard)
        print(action, np.mean(rewards))
        env.close()
