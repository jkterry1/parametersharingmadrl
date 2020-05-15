import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": 6,
    "legend.fontsize": 5,
    "text.usetex": True,
    "pgf.rcfonts": False
});

plt.figure(figsize=(2.65, 1.5))
data_path = "data/pursuit/"

data = np.genfromtxt(os.path.join(data_path,'gupta_trpo.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], '--', label='TRPO', linewidth=0.75)

data = np.genfromtxt(os.path.join(data_path,'gupta_dqn.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], label='DQN', linewidth=0.75)

df = pd.read_csv(os.path.join(data_path,'a2c_2.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1]/8, label='A2C', linewidth=0.75)

df = pd.read_csv(os.path.join(data_path, 'dqn_1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1]/8, '--', label='DD-DQN', linewidth=0.75)

df = pd.read_csv(os.path.join(data_path,'impala_1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='IMPALA', linewidth=0.75)

df = pd.read_csv(os.path.join(data_path, 'ppo_1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='PPO', linewidth=0.75)

plt.plot(np.array([0,60000]),np.array([31.03,31.03]), label='Random', linewidth=0.75)

plt.xlabel('Episode', labelpad=1) 
plt.ylabel('Average Total Reward', labelpad=1)
plt.title('Pursuit')
plt.xticks(ticks=[0,10000,20000,30000,40000,50000,60000],labels=['0','10k','20k','30k','40k','50k','60k'])
plt.tight_layout()
plt.legend(loc='lower right', ncol=1, labelspacing=.2, columnspacing=.25, borderpad=.25)
plt.margins(x=0)
plt.savefig("pursuitGraph_camera.pgf", bbox_inches = 'tight',pad_inches = .025)
