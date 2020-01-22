"""
Adaptation of the chainerRL DQN example
"""
import argparse
import os
import sys
import shutil

from chainer import optimizers
from gym import spaces
import numpy as np

import chainerrl
from chainer.optimizer_hooks import GradientClipping
from chainerrl.agents.double_dqn import DoubleDQN as DDQN
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl import q_functions
from chainerrl import replay_buffer
from dac.envs import SigmoidMultiActMultiValAction as SigMV


def save_agent_and_replay_buffer(agent, t, outdir, logger, suffix='', chckptfrq=-1):
    """
    Replaces chainers default save function and allows to also save the replay buffer.
    """
    dirname = os.path.join(outdir, '{}{}'.format(t, suffix))
    agent.save(dirname)                 # save the agent to "dirname"
    filename = os.path.join(dirname, 'replay_buffer.pkl')
    agent.replay_buffer.save(filename)  # save the replay buffer in "filename"
    logger.info('Saved the agent and replay buffer to %s', dirname)
    with open(os.path.join(dirname, 't.txt'), 'w') as fh:  # also dump the value of the current time step
        fh.writelines(str(t))
    if chckptfrq > 0 and os.path.exists(os.path.join(outdir, '{}{}'.format(t - chckptfrq, suffix))):
        # Delete old checkpoints to not flood the file server with too many frequent checkpoints
        shutil.rmtree(os.path.join(outdir, '{}{}'.format(t-chckptfrq, suffix)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='/tmp/chainerRL_results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 4)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help="Run evaluation mode")
    parser.add_argument('--load', type=str, default=None,
                        help="Load saved_model")
    parser.add_argument('--steps', type=int, default=10 ** 6)
    parser.add_argument('--prioritized-replay', action='store_true')
    parser.add_argument('--replay-start-size', type=int, default=1000)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 2)
    parser.add_argument('--target-update-method', type=str, default='hard')
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=11)
    parser.add_argument('--n-hidden-channels', type=int, default=50)
    parser.add_argument('--n-hidden-layers', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--minibatch-size', type=int, default=None)
    parser.add_argument('--reward-scale-factor', type=float, default=1)
    parser.add_argument('--outdir-time-suffix', choices=['empty', 'none', 'time'], default='empty', type=str.lower)
    parser.add_argument('--checkpoint_frequency', type=int, default=1e3,
                        help="Nuber of steps to checkpoint after")
    parser.add_argument('--verbose', '-v', action='store_true', help='Use debug log-level')
    parser.add_argument('--scenario', choices=['1D-INST', '1D-DIST',
                                               '1DM', '2DM', '3DM', '5DM',
                                               '1D3M', '2D3M', '3D3M', '5D3M'],
                        default='1D-INST', type=str.upper,
                        help='Which scenario to use.')
    if __name__ != '__main__':
        print(__name__)
        parser.add_argument('--timeout', type=int, default=0,
                            help='Wallclock timeout in sec')  # Has no effect in this file!
        # can only be used in conjunction with "train_with_wallclock_limit.py"!
    args = parser.parse_args()
    import logging
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG)

    # Set a random seed used in ChainerRL ALSO SETS NUMPY SEED!
    misc.set_random_seed(args.seed)

    if args.outdir and not args.load:
        outdir_suffix_dict = {'none': '', 'empty': '', 'time': '%Y%m%dT%H%M%S.%f'}
        args.outdir = experiments.prepare_output_dir(
            args, args.outdir, argv=sys.argv, time_format=outdir_suffix_dict[args.outdir_time_suffix])
    elif args.load:
        if args.load.endswith(os.path.sep):
            args.load = args.load[:-1]
        args.outdir = os.path.dirname(args.load)
        count = 0
        fn = os.path.join(args.outdir.format(count), 'scores_{:>03d}')
        while os.path.exists(fn.format(count)):
            count += 1
        os.rename(os.path.join(args.outdir, 'scores.txt'), fn.format(count))
        if os.path.exists(os.path.join(args.outdir, 'best')):
            os.rename(os.path.join(args.outdir, 'best'), os.path.join(args.outdir, 'best_{:>03d}'.format(count)))

    logging.info('Output files are saved in {}'.format(args.outdir))

    def make_env(test):
        if args.scenario == '1D-INST':  # Used to create Figures 2(b)&(c)
            env = SigMV(instance_feats=os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'envs',
                                                    'feats.csv' if not test else 'test_feats.csv'),
                        seed=args.seed, n_actions=1, action_vals=(2, ))
        elif args.scenario == '1D-DIST':  # Used to create Figure 2(a)
            env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
            env = SigMV(seed=env_seed, n_actions=1, action_vals=(2, ))
        elif args.scenario == '1D3M':  # Used to create Figure 3(a)
            env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
            env = SigMV(n_actions=1, action_vals=(3, ), seed=env_seed)
        elif args.scenario == '2D3M':  # Used to create Figure 3(b)
            env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
            env = SigMV(n_actions=2, action_vals=(3, 3), seed=env_seed)
        elif args.scenario == '3D3M':  # Used to create Figure 3(c)
            env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
            env = SigMV(n_actions=3, action_vals=(3, 3, 3), seed=env_seed)
        elif args.scenario == '5D3M':  # Used to create Figure 3(d)
            env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
            env = SigMV(n_actions=5, action_vals=(3, 3, 3, 3, 3), seed=env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        return env

    env = make_env(test=False)
    timestep_limit = 10 ** 3  # TODO don't hardcode env params
    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space

    n_actions = action_space.n
    q_func = q_functions.FCStateQFunctionWithDiscreteAction(
        obs_size, n_actions,
        n_hidden_channels=args.n_hidden_channels,
        n_hidden_layers=args.n_hidden_layers)
    explorer = explorers.LinearDecayEpsilonGreedy(
        args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
        action_space.sample)

    if args.noisy_net_sigma is not None:
        links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        # Turn off explorer
        explorer = explorers.Greedy()

    # Draw the computational graph and save it in the output directory.
    if not args.load:
        chainerrl.misc.draw_computational_graph(
            [q_func(np.zeros_like(obs_space.low, dtype=np.float32)[None])],
            os.path.join(args.outdir, 'model'))

    opt = optimizers.Adam(eps=1e-2)
    opt.setup(q_func)
    opt.add_hook(GradientClipping(5))

    rbuf_capacity = 5 * 10 ** 5
    if args.minibatch_size is None:
        args.minibatch_size = 32
    if args.prioritized_replay:
        betasteps = (args.steps - args.replay_start_size) \
                    // args.update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            rbuf_capacity, betasteps=betasteps)
    else:
        rbuf = replay_buffer.ReplayBuffer(rbuf_capacity)

    agent = DDQN(q_func, opt, rbuf, gamma=args.gamma,
                 explorer=explorer, replay_start_size=args.replay_start_size,
                 target_update_interval=args.target_update_interval,
                 update_interval=args.update_interval,
                 minibatch_size=args.minibatch_size,
                 target_update_method=args.target_update_method,
                 soft_update_tau=args.soft_update_tau,
                 )
    t_offset = 0
    if args.load:  # Continue training model or load for evaluation
        agent.load(args.load)
        rbuf.load(os.path.join(args.load, 'replay_buffer.pkl'))
        try:
            t_offset = int(os.path.basename(args.load).split('_')[0])
        except TypeError:
            with open(os.path.join(args.load, 't.txt'), 'r') as fh:
                data = fh.readlines()
            t_offset = int(data[0])
        except ValueError:
            t_offset = 0

    eval_env = make_env(test=True)

    if args.evaluate:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        criterion = 'steps'  # can be made an argument if we support any other form of checkpointing
        l = logging.getLogger('Checkpoint_Hook')

        def checkpoint(env, agent, step):
            if criterion == 'steps':
                if step % args.checkpoint_frequency == 0:
                    save_agent_and_replay_buffer(agent, step, args.outdir, suffix='_chkpt', logger=l,
                                                 chckptfrq=args.checkpoint_frequency)
            else:
                # TODO seems to checkpoint given wall_time we would have to modify the environment such that it tracks
                # time or number of episodes
                raise NotImplementedError

        def eval_hook(env, agent, step):
            """
            Necessary hook to evaluate the DDQN on all 100 Training instances.
            :param env: The training environment
            :param agent: (Partially) Trained agent
            :param step: Number of observed training steps.
            :return:
            """
            if step % 10 == 0:  #
                train_reward = 0
                for _ in range(100):
                    obs = env.reset()
                    done = False
                    rews = 0
                    while not done:
                        obs, r, done, _ = env.step(agent.act(obs))
                        rews += r
                    train_reward += rews
                train_reward = train_reward / 100
                with open(os.path.join(args.outdir, 'train_reward.txt'), 'a') as fh:
                    fh.writelines(str(train_reward) + '\t' + str(step) + '\n')

        hooks = [checkpoint]
        if args.scenario=='1D-INST':
            hooks.append(eval_hook)
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=None,  # unlimited number of steps per evaluation rollout
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            eval_env=eval_env,
            train_max_episode_len=timestep_limit,
            step_hooks=hooks,
            step_offset=t_offset
        )


if __name__ == '__main__':
    main()
