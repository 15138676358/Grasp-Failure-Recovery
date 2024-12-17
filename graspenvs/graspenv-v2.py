"""
GraspEnv_v2.py
Created by Yue Wang on 2024-09-10
Version 2.1
动作空间为3维, 历史为15*4维, 视觉为8*2维, 观测为8*2+15*4=76维
self.state = {'contour': contour, 'convex': convex, 'candidate_actions': candidate_actions, 'mass': mass, 'com': com, 'attempt': 0, 'history': np.zeros((self.max_steps, 4))}
Note: 
状态空间增加候选抓取
轮廓采用等比例缩放法
"""

import cv2
import matplotlib.path as mplPath
import numpy as np
import gymnasium
from gymnasium import spaces
from scipy.interpolate import interp1d

class GraspEnv_v2(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, render_mode='human'):
        super(GraspEnv_v2, self).__init__()
        self.max_steps = 15
        self.render_mode = render_mode
        self.state_space = {'contour': spaces.Box(low=0, high=1, shape=(100, 2)), 'convex': spaces.Box(low=0, high=1, shape=(8, 2)), 'mass': spaces.Box(low=0, high=1, shape=(1, )), 'com': spaces.Box(low=0, high=1, shape=(2, )), 'attempt': spaces.Discrete(1), 'history': spaces.Box(low=-10, high=10, shape=(self.max_steps, 4))}
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
        # initialize the contour and candidate actions
        contour, convex, candidate_actions = generate_contour()
        # initialize the mass and com
        mass = np.array([np.random.uniform(5, 25)])
        while True:
            com = np.random.uniform(0, 50, 2)
            if mplPath.Path(convex).contains_point(com) and np.linalg.norm(contour - com, axis=1).min() > 1:
                break
        self.state = {'contour': contour, 'convex': convex, 'candidate_actions': candidate_actions, 'mass': mass, 'com': com, 'attempt': 0, 'history': np.zeros((self.max_steps, 4))}

        return self.get_observation(), self.get_info()
    
    def render(self):
        frame = np.ones((500, 500, 3), dtype=np.uint8) * 255
        # draw the contour
        for point in self.state['contour']:
            cv2.circle(frame, (int(500 * point[0]), int(500 * point[1])), 3, (0, 255, 0), -1)  # Black point
        # draw the convex
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(500 * self.state['convex'], dtype=np.int32)], (0, 0, 0))
        alpha = 0.1
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # draw the com
        com = self.state['com']
        mass = self.state['mass']
        cv2.circle(frame, (int(500 * com[0]), int(500 * com[1])), 3, (0, 0, 255), -1)  # Red point
        cv2.putText(frame, f'{25 * mass[0]:.1f}', (int(500 * com[0]) + 5, int(500 * com[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # draw the history
        cv2.putText(frame, f"Attempts: {self.state['attempt']}", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        for i in range(self.state['attempt']):
            action_x, action_y, _, force = self.history[i]
            if i == self.state['attempt'] - 1:
                cv2.circle(frame, (int(500 * action_x), int(500 * action_y)), 3, (255, 0, 0), -1)  # If the last action, red point
            else:
                cv2.circle(frame, (int(500 * action_x), int(500 * action_y)), 3, (0, 0, 0), -1)  # Black point
            cv2.putText(frame, f'{25 * force:.1f}', (int(500 * action_x) + 5, int(500 * action_y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        if self.render_mode == 'human':
            cv2.imshow('Environment State', frame)
            cv2.waitKey(500)  # wait for 1ms
            return frame
        elif self.render_mode == 'rgb_array':
            return frame
        

# 生成随机轮廓, 包括两个相互垂直的矩形。详见论文agent5
def generate_contour():
    # Create a block comsists of two rectangles, whose size is length x width
    # initialize the center of the main rectangle
    center = np.random.uniform(5, 45, 2) / 50
    theta = np.random.uniform(0, 2*np.pi)
    length = np.random.uniform(10, 30) / 50
    width = np.random.uniform(5, 10) / 50

    t_vec = np.array([np.cos(theta), np.sin(theta)])
    n_vec = np.array([-np.sin(theta), np.cos(theta)])
    pt_top = center + length / 2 * t_vec
    pt_bottom = center - length / 2 * t_vec
    pt1 = center + length / 2 * t_vec + width / 2 * n_vec
    pt2 = center + length / 2 * t_vec - width / 2 * n_vec
    pt3 = center - length / 2 * t_vec - width / 2 * n_vec
    pt4 = center - length / 2 * t_vec + width / 2 * n_vec

    # initialize the size of the co-rectangles
    co_length = np.random.uniform(0, 30, 2) / 50
    co_width = np.random.uniform(5, 10) / 50
    offset = np.random.uniform(0, length / 2 - co_width / 2)

    pt_left = center + offset * t_vec + co_length[0] / 2 * n_vec
    pt_right = center + offset * t_vec - co_length[1] / 2 * n_vec
    pt5 = center + offset * t_vec + co_width / 2 * t_vec + co_length[0] / 2 * n_vec
    pt6 = center + offset * t_vec + co_width / 2 * t_vec + width / 2 * n_vec
    pt7 = center + offset * t_vec + co_width / 2 * t_vec - width / 2 * n_vec
    pt8 = center + offset * t_vec + co_width / 2 * t_vec - co_length[1] / 2 * n_vec
    pt9 = center + offset * t_vec - co_width / 2 * t_vec + co_length[0] / 2 * n_vec
    pt10 = center + offset * t_vec - co_width / 2 * t_vec + width / 2 * n_vec
    pt11 = center + offset * t_vec - co_width / 2 * t_vec - width / 2 * n_vec
    pt12 = center + offset * t_vec - co_width / 2 * t_vec - co_length[1] / 2 * n_vec

    # merge the points
    if co_length[0] / 2 < width / 2:
        left_pts = [pt2, pt1, pt4]
        left_convex = [pt1, pt6, pt10, pt4]
    else:
        left_pts = [pt2, pt1, pt6, pt5, pt9, pt10, pt4]
        left_convex = [pt1, pt5, pt9, pt4]

    if co_length[1] / 2 < width / 2:
        right_pts = [pt4, pt3, pt2]
        right_convex = [pt3, pt11, pt7, pt2]
    else:
        right_pts = [pt4, pt3, pt11, pt12, pt8, pt7, pt2]
        right_convex = [pt3, pt12, pt8, pt2]

    loop_outline = np.array(right_pts + left_pts)

    # rearrange the points to 50 points
    # Compute the cumulative distance along the outline
    distance = np.cumsum(np.sqrt(np.sum(np.diff(loop_outline, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)

    # Create an interpolator for the x and y coordinates
    fx = interp1d(distance, loop_outline[:, 0])
    fy = interp1d(distance, loop_outline[:, 1])

    # Create an array of evenly spaced distance values
    sample_distances = np.linspace(0, distance[-1], 100)

    # Use the interpolators to compute the x and y coordinates of the samples
    sample_x = fx(sample_distances)
    sample_y = fy(sample_distances)
    contour = np.column_stack((sample_x, sample_y))
    convex = np.array(right_convex + left_convex)

    # calculate the candidate actions
    grid_size = 50
    candidate_actions = []
    for a_x in range(0, 1, 1 / grid_size):
        for a_y in range(0, 1, 1 / grid_size):
            if mplPath.Path([pt1, pt2, pt3, pt4]).contains_point([a_x, a_y]) and not mplPath.Path([pt6, pt7, pt11, pt10]).contains_point([a_x, a_y]):
                a_theta = theta
                candidate_actions.append([a_x, a_y, a_theta])
            if mplPath.Path([pt5, pt8, pt12, pt9]).contains_point([a_x, a_y]) and not mplPath.Path([pt6, pt7, pt11, pt10]).contains_point([a_x, a_y]):
                a_theta = theta + np.pi / 2
                candidate_actions.append([a_x, a_y, a_theta])

    return contour, convex, candidate_actions     
