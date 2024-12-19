"""
Note:
在调用step和render前, 需要先调用reset初始化环境
"""


import cv2
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
    agent = graspagents.GraspAgent({'env': 'GraspEnv_v3', 'model': 'Dirnet'})
    env = gymnasium.make(id='GraspEnv_v2', render_mode='human')
    num_trials = 10
    attempts, returns, success = np.zeros((num_trials)), np.zeros((num_trials)), np.zeros((num_trials))
    video_writer = cv2.VideoWriter(filename='video.mp4', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=5, frameSize=(500, 500))
    for i in range(num_trials):
        contour, convex = graspenvs.utils.generate_contour()
        agent.reset(contour, convex)
        env.initialize_state(contour, convex)
        env.reset()
        while True:
            # 随机选择一个动作
            # action = env.state['candidate_actions'][np.random.randint(0, len(env.state['candidate_actions']))]
            action = agent.choose_action()
            state, reward, done, truncated, info = env.step(action)
            agent.update(action, env.state['history'][env.state['attempt'] - 1, -1])
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
    print(f'Average Returns: {returns.mean()}')
    print(f'Success Rate: {np.sum(success) / num_trials}')


if __name__ == '__main__':
    test_env_v2_and_v3()
