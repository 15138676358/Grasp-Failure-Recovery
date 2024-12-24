import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np


# 绘制训练曲线
def sb3_draw_train_curve(log_dir):
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    results = load_results(log_dir)
    x, y = ts2xy(results, 'timesteps')
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    # 绘制 mean reward
    axs[0].plot(x[199::1], np.convolve(np.array(y), np.ones(200)/200, mode='valid'))
    axs[0].set_title('Mean Reward')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Reward')
    # 绘制 mean episode length
    mean_ep_length = results['l']
    axs[1].plot(x[199::1], np.convolve(np.array(mean_ep_length), np.ones(200)/200, mode='valid'))
    axs[1].set_title('Mean Episode Length')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Episode Length')
    # 绘制 success rate
    success_rate = []
    for i in range(len(mean_ep_length) // 100):
        sample_list = mean_ep_length[100 * i : 100 * (i + 1)]
        success_rate.append(len([x for x in sample_list if x < 10]) / 100)
    axs[2].plot(success_rate)
    axs[2].set_title('Success Rate')
    axs[2].set_xlabel('100*Episodes')
    axs[2].set_ylabel('Success Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'train_curve.png'))
    plt.close()

# 绘制测试曲线
def sb3_draw_eval_curve(log_dir):
    data = np.load(os.path.join(log_dir, 'evaluations.npz'))
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    # 绘制 mean reward
    y = data['results'][:, 0]
    axs[0].plot(np.convolve(y, np.ones(100)/100, mode='valid'), color='b')
    axs[0].set_xlim(0, 5250)
    axs[0].set_title('Mean Reward', fontsize=14)
    axs[0].set_xlabel('Episodes', fontsize=12)
    # axs[0].set_ylabel('Reward', fontsize=12)
    # 绘制 mean episode length
    y = data['ep_lengths'][:, 0]
    axs[1].plot(np.convolve(y, np.ones(100)/100, mode='valid'), color='b')
    axs[1].set_xlim(0, 5250)
    axs[1].set_title('Mean Episode Length', fontsize=14)
    axs[1].set_xlabel('Episodes', fontsize=12)
    # axs[1].set_ylabel('Episode Length', fontsize=12)
    # 绘制 success rate
    success_rate = []
    for i in range(y.shape[0] // 100):
        sample_list = y[100 * i : 100 * (i + 1)]
        success_rate.append(len([x for x in sample_list if x < 9]) / 100)
    axs[2].plot(success_rate, color='b')
    axs[2].set_xlim(0, 53)
    axs[2].set_title('Success Rate', fontsize=14)
    axs[2].set_xlabel('100*Episodes', fontsize=12)
    # axs[2].set_ylabel('Success Rate', fontsize=12)

    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'eval_curve.png'))
    plt.close()


if __name__ == '__main__':
    log_dir = './checkpoint/GraspEnv_v3/rl/SAC/lr_0.0001_bs_512'
    sb3_draw_train_curve(log_dir)
    # sb3_draw_eval_curve(log_dir)