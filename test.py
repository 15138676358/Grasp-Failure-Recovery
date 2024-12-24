"""
Note:
在调用step和render前, 需要先调用reset初始化环境
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import matplotlib.pyplot as plt
import numpy as np
import graspenvs
import gymnasium
import graspenvs.utils
import graspagents


def test_env_v1():
    env = gymnasium.make(id='GraspEnv_v1', render_mode='rgb_array')
    attempts, returns, success = np.zeros((100)), np.zeros((100)), np.zeros((100))
    video_writer = cv2.VideoWriter(filename='video.mp4', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=5, frameSize=(500, 500))
    for i in range(10):
        env.reset()
        while True:
            # 随机选择一个动作
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            frame = env.render()  # 获取渲染帧
            # 将帧写入视频文件
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if done or truncated:
                attempts[i] = env.state['attempt']
                success[i] = done and not truncated
                break
    
    # 释放视频写入对象并关闭环境
    video_writer.release()
    cv2.destroyAllWindows()

    print(f'Average Attempts: {attempts.mean()}')
    print(f'Success Rate: {np.sum(success) / 100}')

def test_env_v2_and_v3():
    # agent = graspagents.GraspAgent_dl({'env': 'GraspEnv_v3', 'model': 'Transnet'})
    # agent = graspagents.GraspAgent_bayes({'env': 'GraspEnv_v3'})
    agent = graspagents.GraspAgent_rl({'env': 'GraspEnv_v3', 'model': 'SAC'})
    env = gymnasium.make(id='GraspEnv_v3', render_mode='rgb_array')
    num_tasks, num_trials = 100, 1
    attempts, returns, success = np.zeros((num_tasks * num_trials)), np.zeros((num_tasks * num_trials)), np.zeros((num_tasks * num_trials))
    w = []
    video_writer = cv2.VideoWriter(filename='video.mp4', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=5, frameSize=(500, 500))
    for i in range(num_tasks):
        while True:
            env.reset()
            contour, convex = env.state['contour'].copy(), env.state['convex'].copy()
            agent.reset(contour, convex)
            if agent.env.state['candidate_actions'].shape[0] > 0:
                break
        for j in range(num_trials):
            env.state['attempt'], env.state['history'] = 0, np.zeros((env.max_steps, 4))
            agent.env.state['attempt'], agent.env.state['history'] = 0, np.zeros((agent.env.max_steps, 4))
            while True:
                # 随机选择一个动作
                # action = env.state['candidate_actions'][np.random.randint(0, len(env.state['candidate_actions']))]
                action = agent.choose_action()[0]
                next_observation, reward, done, truncated, info = env.step(action)
                agent.update(action, env.state['history'][env.state['attempt'] - 1, -1])
                frame = env.render()  # 获取渲染帧
                # 将帧写入视频文件
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if done or truncated:
                    attempts[num_trials * i + j] = env.state['attempt']
                    success[num_trials * i + j] = done and not truncated
                    print(f'Task {i + 1}, Trial {j + 1}, Attempts: {env.state["attempt"]}, Success: {done and not truncated}')
                    break
            # if done and not truncated:
            #     w.append(agent.scores)    
    
    # 释放视频写入对象并关闭环境
    video_writer.release()
    cv2.destroyAllWindows()
    # w = np.array(w)
    print(f'Average Attempts: {attempts.mean()}')
    print(f'Average Returns: {returns.mean()}')
    print(f'Success Rate: {100 * np.sum(success) / (num_tasks * num_trials)}%')
    plt.plot(attempts)
    plt.savefig('attempts.png')


if __name__ == '__main__':
    test_env_v2_and_v3()
