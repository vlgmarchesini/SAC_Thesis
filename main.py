import argparse
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from DynEnv import make_custom_env #,DynEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.exploration import OUNoise
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Tests')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--noise_std', type=float, default=1.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--auto_alpha', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=12000)
    parser.add_argument('--step-per-collect', type=int, default=5)
    parser.add_argument('--update-per-step', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=str, default='128, 128')
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='/mundus/vgomesma005/rl_corrector/log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', type=bool, default=False) #Reward Normalization
    parser.add_argument('--path', type=str,
                        default='/mundus/vgomesma005/rl_corrector/Xu_Hydrofoil_Useed1236_Train117000s.csv')

    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return parser.parse_args()




def test_sac(args=get_args()):
    env = make_custom_env(args.path)
    activation = nn.Tanh
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
        [lambda: make_custom_env(args.path) for _ in range(args.training_num)]
    )

    test_envs = DummyVectorEnv(
        [lambda: make_custom_env(args.path) for _ in range(args.test_num)]
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #train_envs.seed(args.seed)
    #test_envs.seed(args.seed)

    # model
    # check if hidden size has the size of the hidden layers and the depth
    hidden_sizes = [int(var) for var in args.hidden_sizes.split(',')]
    print(f"Hidden Size={hidden_sizes}", flush=True)
    net = Net(args.state_shape, hidden_sizes=hidden_sizes, activation=activation, device=args.device)
    print()
    actor = ActorProb(
        net,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True
    ).to(args.device)
    print(f"Actor Ready", flush=True)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=hidden_sizes,
        activation=activation,
        concat=True,
        device=args.device
    )
    print(f"Critic1 Ready", flush=True)
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=hidden_sizes,
        activation=activation,
        concat=True,
        device=args.device
    )
    print(f"Critic2 Ready", flush=True)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)
    print(f"alpha optimizer ready", flush=True)
    
    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        reward_normalization=args.rew_norm,
        exploration_noise=OUNoise(0.0, args.noise_std),
        action_space=env.action_space
    )
    print(f"SAC Policy ready", flush=True)
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    print(f"train_collector ready", flush=True)
    test_collector = Collector(policy, test_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)))
    print(f"test_collector ready", flush=True)
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, 'sac')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        is_end = mean_rewards >= env.reward_threshold
        if is_end:
            print(f"::: CRITERIA REACHED, Mean Rewards:{mean_rewards} ================================================",flush)
        return is_end

    # trainer
    print(f"Starting Training", flush=True)
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
    )
    print(f"Finished Training", flush=True)

    assert stop_fn(result['best_reward'])
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        policy.eval()
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    test_sac()