# 基于stable-baselines3的RL算法，实现graspenv的智能体，训练代码如下：
#
import argparse
import graspenvs
import gymnasium
import os
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
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
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

train_env = gymnasium.make(id=args.env, render_mode='rgb_array')
train_env = Monitor(train_env, log_dir)
train_env = DummyVecEnv([lambda: train_env])
eval_env = gymnasium.make(id=args.env, render_mode='rgb_array')
eval_env = Monitor(eval_env, log_dir)
eval_env = DummyVecEnv([lambda: eval_env])
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=100,
                             n_eval_episodes=10, deterministic=True, render=False)

# model = graspagents.GraspSAC(graspagents.GraspSACPolicy, train_env, verbose=1, learning_starts=100)
model = graspagents.GraspPPO(graspagents.GraspPPOPolicy, train_env, verbose=1)
model.set_logger(new_logger)
model.learn(total_timesteps=10000, callback=eval_callback)
model.save(os.path.join(log_dir, "model_final"))