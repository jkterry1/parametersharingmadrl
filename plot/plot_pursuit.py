import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.rcParams["font.family"] = "serif"

plt.figure(figsize=(6.5, 3.5)) 
data_path = "data/pursuit/"

data = np.genfromtxt(os.path.join(data_path,'gupta_trpo.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], label='TRPO')

data = np.genfromtxt(os.path.join(data_path,'gupta_dqn.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], label='DQN')

df = pd.read_csv(os.path.join(data_path,'a2c_2.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1]/8, label='A2C')

df = pd.read_csv(os.path.join(data_path, 'dqn_1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1]/8, label='DD-DQN')

df = pd.read_csv(os.path.join(data_path,'impala_1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='IMPALA')

df = pd.read_csv(os.path.join(data_path, 'ppo_1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='PPO')

plt.plot(np.array([0,60000]),np.array([31.03,31.03]), label='Random')

plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.title('Pursuit')
plt.xticks(ticks=[0,10000,20000,30000,40000,50000,60000],labels=['0','10k','20k','30k','40k','50k','60k'])
plt.tight_layout()
plt.legend(loc='lower right', ncol=1, labelspacing=.2, columnspacing=.25, borderpad=.25)
plt.margins(x=0)
plt.savefig("pursuitGraph_camera.png", bbox_inches = 'tight',pad_inches = .025)
