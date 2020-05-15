import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy
from scipy import signal

matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": 6,
    "legend.fontsize": 5,
    "ytick.labelsize": 4,
    "text.usetex": True,
    "pgf.rcfonts": False
});

plt.figure(figsize=(2.65, 1.5))
data_path = "data/multi_walker/"

data = np.genfromtxt(os.path.join(data_path, 'gupta_ddpg.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], '--', label='DDPG', linewidth=0.75)

data = np.genfromtxt(os.path.join(data_path, 'gupta_trpo.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], '--', label='TRPO', linewidth=0.75)

df = pd.read_csv(os.path.join(data_path, 'SAC.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1], int(len(data[:, 1])/50), 4)
plt.plot(data[:, 0], data[:, 1], label='SAC', linewidth=0.75)

df = pd.read_csv(os.path.join(data_path,'A2C.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], '--', label='A2C', linewidth=0.75)

df = pd.read_csv(os.path.join(data_path,'APEX_DDPG.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='ApeX-DDPG', linewidth=0.75)

df = pd.read_csv(os.path.join(data_path,'IMPALA.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='IMPALA', linewidth=0.75)

df = pd.read_csv(os.path.join(data_path, 'PPO1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='PPO', linewidth=0.75)

df = pd.read_csv(os.path.join(data_path,'TD3.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50),5)
plt.plot(data[:, 0], data[:, 1], label='TD3', linewidth=0.75)

plt.plot(np.array([0,60000]),np.array([-102.05,-102.05]), label='Random', linewidth=0.75)

plt.xlabel('Episode', labelpad=1)
plt.ylabel('Average Total Reward', labelpad=1)
plt.title('Multiwalker')
plt.xticks(ticks=[0,10000,20000,30000,40000,50000,60000],labels=['0','10k','20k','30k','40k','50k','60k'])
plt.tight_layout()
plt.legend(loc='lower right', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25)
plt.margins(x=0)
plt.savefig("multiwalkerGraph_camera.pgf", bbox_inches = 'tight',pad_inches = .025)
