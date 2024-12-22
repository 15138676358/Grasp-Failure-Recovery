# 基于stable-baselines3的RL算法，实现graspenv的智能体，训练代码如下：
#
import argparse
import graspenvs
import gymnasium
import os
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import graspagents



# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='GraspEnv_v3', help='environment name')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--algorithm', type=str, default='PPO', help='algorithm type')
args = parser.parse_args()
log_dir = f'./checkpoint/{args.env}/rl/{args.algorithm}/lr_{args.lr}_bs_{args.batch_size}'
os.makedirs(log_dir, exist_ok=True)

env = gymnasium.make(id=args.env, render_mode='rgb_array')
env = DummyVecEnv([lambda: env])

model = graspagents.GraspPPO(graspagents.GraspPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
model.save(os.path.join(log_dir, "model_final"))