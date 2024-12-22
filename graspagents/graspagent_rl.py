# 基于stable-baselines3的RL算法，实现graspenv的智能体，训练代码如下：
#
import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th


class GraspAgent_rl:
    def __init__(self, config):
        self.env = gymnasium.make(id=config['env'], render_mode='rgb_array')
        self.model = load_model(self.env, config)
    
    def choose_action(self):
        observation = self.env.get_observation()
        action = self.model.predict(observation, deterministic=False)

        return action

    def update(self, action, force):
        self.env.state['history'][self.env.state['attempt']] = np.array([action[0], action[1], action[2], force])
        self.env.state['attempt'] += 1

    def reset(self, contour, convex):
        self.env.reset()
        self.env.initialize_state(contour, convex)

class GraspPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(GraspPolicy, self).__init__(*args, **kwargs)
        self.action_dist = DiagGaussianDistribution(self.action_space.shape[0])
        self.candidate_actions = candidate_actions

    def forward(self, obs: th.Tensor, candidate_actions):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        candidate_actions = th.tensor(candidate_actions, dtype=th.float32).to(device='cuda')
        scores = th.exp(distribution.log_prob(candidate_actions - distribution.mode()))
        topk = th.argsort(scores)[-100:]
        topk_probs = scores[topk] / th.sum(scores[topk])
        action = candidate_actions[topk[th.multinomial(topk_probs, 1).item()]]

        return action

class GraspPPO(PPO):
    def __init__(self, policy, env, **kwargs):
        super(GraspPPO, self).__init__(policy, env, **kwargs)

    def predict(self, observation, deterministic=False):
        obs = th.tensor(observation, dtype=th.float32).unsqueeze(0).to(device='cuda')  # 添加 batch 维度
        action = self.policy.forward(obs, self.env.get_attr('state')[0]['candidate_actions'])

        return action.cpu().detach().numpy()  # 移除 batch 维度并转换为 numpy 数组
    

def load_model(env, config):
    env_id, model_id = config['env'], config['model']
    if model_id == 'PPO':
        model = GraspPPO(GraspPolicy, env, verbose=1)
    elif model_id == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1)
    model.load(f'./checkpoint/{env_id}/rl/{model_id}/model_best')
    
    return model

