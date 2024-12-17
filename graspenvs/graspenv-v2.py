"""
GraspEnv_v2.py
Created by Yue Wang on 2024-09-10
Version 2.1
动作空间为3维, 历史为15*4维, 视觉为8*2维, 观测为8*2+15*4=76维
"""

import cv2
import matplotlib.path as mplPath
import numpy as np
import gymnasium
from gymnasium import spaces
import utils

class GraspEnv_v2(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, render_mode='human'):
        super(GraspEnv_v2, self).__init__()
        self.max_steps = 15
        self.render_mode = render_mode
        self.state_space = {'contour': spaces.Box(low=0, high=1, shape=(50, 2)), 'convex': spaces.Box(low=0, high=1, shape=(8, 2)), 'mass': spaces.Box(low=0, high=1, shape=(1, )), 'com': spaces.Box(low=0, high=1, shape=(2, )), 'attempt': spaces.Discrete(1), 'history': spaces.Box(low=-10, high=10, shape=(self.max_steps, 4))}
        self.observation_space = spaces.Box(low=0, high=1, shape=(76, ))
        self.action_space = spaces.Box(low=0, high=1, shape=(3, ))
        self.reset()
    
    def get_observation(self):
        return np.concatenate((self.state['convex'].reshape(-1), self.state['history'].reshape(-1)))
    
    def get_info(self):
        return {'contour': self.state['contour'], 'convex': self.state['convex'], 'mass': self.state['mass'], 'com': self.state['com']}

    def is_done(self, force):
        return ((np.abs(force / self.state['mass'][0]) > 0.9) or (self.state['attempt'] >= self.max_steps - 1))

    def is_truncated(self, force):
        return ((self.state['attempt'] >= self.max_steps - 1) and (np.abs(force / self.state['mass'][0]) < 0.9))

    def compute_force(self, action):
        # calculate the force
        noise = np.random.normal(0, 0.0001)
        t_vec = np.array([np.cos(action[2]), np.sin(action[2])])
        norm_points = np.dot(self.state['contour'] - action[0:2], t_vec)
        norm_com = np.dot(self.state['com'] - action[0:2], t_vec)
        max_norm_point = np.max(norm_points / norm_com) * norm_com
        force = self.state['mass'][0] * (max_norm_point - norm_com) / max_norm_point + noise

        return force
    
    def compute_reward(self, force):
        # calculate the reward
        return np.abs(force / self.state['mass'][0]) * np.abs(force / self.state['mass'][0]) + 1 * (np.abs(force - self.state['mass'][0]) < 0.05) - 1 - 0 * (force < 0.1)
    
    def step(self, action):
        force = self.compute_force(action)
        self.state['history'][self.state['attempt']] = np.array([action[0], force])
        self.state['attempt'] += 1

        return self.get_observation(), self.compute_reward(force), self.is_done(force), self.is_truncated(force), self.get_info()
    
    def reset(self, seed=None):
        # initialize the env.state
        while True:
            contour, convex, principal_pts = utils.generate_contour()
            if contour.min() > 0 and contour.max() < 50:
                break
        mass = np.array([np.random.uniform(5, 25)])
        # mass = np.array([15])
        while True:
            com = np.random.uniform(0, 50, 2)
            # com = np.array([30, 30])
            if mplPath.Path(convex).contains_point(com) and np.linalg.norm(contour - com, axis=1).min() > 1:
                break
        state = {'contour': contour / 50, 'convex': convex / 50, 'principal_pts': principal_pts / 50, 'mass': mass / 25, 'com': com / 50, 'attempt': 0}

        self.state = {'contour': contour, 'convex': convex, 'mass': mass, 'com': com, 'attempt': 0, 'history': np.zeros((self.max_steps, 2))}
        utils.generate_state()

        return self.get_observation(), self.get_info()
    
    def render(self):
        frame = np.ones((500, 500, 3), dtype=np.uint8) * 255
        # draw the contour
        contour = self.state['contour']
        for point in contour:
            cv2.circle(frame, (int(500 * point[0]), int(250 * 1)), 3, (0, 255, 0), -1)  # Black point
        # draw the com
        com = self.state['com']
        mass = self.state['mass']
        cv2.circle(frame, (int(500 * com[0]), int(250 * 1)), 3, (0, 0, 255), -1)  # Red point
        cv2.putText(frame, f'{25 * mass[0]:.1f}', (int(500 * com[0]) + 5, int(250 * 1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, f"Attempts: {self.state['attempt']}", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # draw the history
        for i in range(self.state['attempt']):
            action_x, force = self.history[i]
            action_y = 1
            if i == self.state['attempt'] - 1:
                cv2.circle(frame, (int(500 * action_x), int(250 * action_y)), 3, (255, 0, 0), -1)  # If the last action, red point
            else:
                cv2.circle(frame, (int(500 * action_x), int(250 * action_y)), 3, (0, 0, 0), -1)  # Black point
            cv2.putText(frame, f'{25 * force:.1f}', (int(500 * action_x) + 5, int(250 * action_y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        # show the frame
        if self.render_mode == 'human':
            cv2.imshow('Environment State', frame)
            cv2.waitKey(500)  # wait for 1ms
            return frame
        elif self.render_mode == 'rgb_array':
            return frame
        
