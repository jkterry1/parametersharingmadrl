import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy
from scipy import signal

plt.rcParams["font.family"] = "serif"

plt.figure(figsize=(6.5, 3.5)) 
data_path = "data/multi_walker/"

data = np.genfromtxt(os.path.join(data_path,'gupta_ddpg.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], label='DDPG')

data = np.genfromtxt(os.path.join(data_path,'gupta_trpo.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], label='TRPO')

df = pd.read_csv(os.path.join(data_path, 'SAC.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50),4)
plt.plot(data[:, 0], data[:, 1], label='SAC')

df = pd.read_csv(os.path.join(data_path,'A2C.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='A2C')

df = pd.read_csv(os.path.join(data_path,'APEX_DDPG.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='ApeX-DDPG')

df = pd.read_csv(os.path.join(data_path,'IMPALA.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='IMPALA')

df = pd.read_csv(os.path.join(data_path, 'PPO1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='PPO')

df = pd.read_csv(os.path.join(data_path,'TD3.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50),5)
plt.plot(data[:, 0], data[:, 1], label='TD3')

plt.plot(np.array([0,60000]),np.array([-102.05,-102.05]), label='Random')

plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.title('Multiwalker')
plt.xticks(ticks=[0,10000,20000,30000,40000,50000,60000],labels=['0','10k','20k','30k','40k','50k','60k'])
plt.tight_layout()
plt.legend(loc='lower right', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25)
plt.margins(x=0)
plt.savefig("multiwalkerGraph_camera.png", bbox_inches = 'tight',pad_inches = .025)
