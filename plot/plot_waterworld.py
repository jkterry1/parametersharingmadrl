import numpy as np
import pandas as pd
import os
import scipy
from scipy import signal
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

plt.figure(figsize=(6.5, 3.5))
data_path = "data/waterworld/"

data = np.genfromtxt(os.path.join(data_path,'gupta_trpo.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], '--', label='TRPO')

df = pd.read_csv(os.path.join(data_path,'a2c.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='A2C')

df = pd.read_csv(os.path.join(data_path, 'sac2.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50),5)
plt.plot(data[:, 0], filtered, label='SAC')

df = pd.read_csv(os.path.join(data_path,'apex_ddpg_2.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], '--', label='ApeX-DDPG')

df = pd.read_csv(os.path.join(data_path,'impala.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='IMPALA')

df = pd.read_csv(os.path.join(data_path, 'ppo.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50)+1,5)
plt.plot(data[:, 0], data[:, 1], label='PPO')

df = pd.read_csv(os.path.join(data_path,'td3_3.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50),5)
plt.plot(data[:, 0], filtered, label='TD3')

plt.plot(np.array([0,60000]),np.array([-6.82,-6.82]), label='Random')

plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.title('Waterworld')
plt.xticks(ticks=[0,10000,20000,30000,40000,50000,60000],labels=['0','10k','20k','30k','40k','50k','60k'])
plt.tight_layout()
plt.legend(loc='upper right', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25)
plt.margins(x=0)
plt.savefig("waterworldGraph_camera.png", bbox_inches = 'tight',pad_inches = .025)
