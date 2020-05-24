import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy
from scipy import signal
import sys

game = sys.argv[1].lower()
games = ["pursuit", "multiwalker", "waterworld"]
assert game in games, "Game is not in {}".format(games)

algorithm = sys.argv[2].upper()

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(6.5, 3.5))

if game == 'waterworld':
    # plt.plot(np.array([0,60000]),np.array([-6.82,-6.82]), label='Random') # waterworld
    # data = np.genfromtxt(os.path.join('./plot/data/waterworld', 'gupta_trpo.csv'), delimiter=',')
    # plt.plot(data[:, 0], data[:, 1], 'k:', label='Gupta_TRPO')
    methods = ["A2C", "APEX_DDPG", "DDPG", "IMPALA", "PPO", "SAC", "TD3"] # waterworld
elif game == 'multiwalker':
    plt.plot(np.array([0,60000]),np.array([-102.05,-102.05]), label='Random') # multiwalker
    data = np.genfromtxt(os.path.join('./plot/data/multi_walker', 'gupta_ddpg.csv'), delimiter=',')
    plt.plot(data[:, 0], data[:, 1], 'k:', label='Gupta_DDPG')
    data = np.genfromtxt(os.path.join('./plot/data/multi_walker', 'gupta_trpo.csv'), delimiter=',')
    plt.plot(data[:, 0], data[:, 1], '--',color=[0.6,0.6,0.6,1], label='Gupta_TRPO')
    methods = ["A2C", "APEX_DDPG", "DDPG", "IMPALA", "PPO", "SAC", "TD3"] # multiwalker
elif game == 'pursuit':
    # plt.plot(np.array([0,60000]),np.array([31.03,31.03]), label='Random') # pursuit
    # data = np.genfromtxt(os.path.join('./plot/data/pursuit', 'gupta_dqn.csv'), delimiter=',')
    # plt.plot(data[:, 0], data[:, 1], 'k:', label='Gupta_DQN')
    # data = np.genfromtxt(os.path.join('./plot/data/pursuit', 'gupta_trpo.csv'), delimiter=',')
    # plt.plot(data[:, 0], data[:, 1], '--',color=[0.6,0.6,0.6,1], label='Gupta_TRPO')
    methods = ["A2C", "ADQN", "DQN", "IMPALA", "PPO", "RDQN"] # pursuit

data_path = "../ray_results/"+game

# plt.figure(figsize=(6.5, 3.5))

for x in os.listdir(data_path):
    '''
    if x not in methods:
        continue
    sub_folder = data_path + "/" + x
    counter = 0
    if game == 'waterworld':
        plt.plot(np.array([0,60000]),np.array([-6.82,-6.82]), label='Random') # waterworld
        data = np.genfromtxt(os.path.join('./plot/data/waterworld', 'gupta_trpo.csv'), delimiter=',')
        plt.plot(data[:, 0], data[:, 1], 'k:', label='Gupta_TRPO')
        methods = ["A2C", "APEX_DDPG", "DDPG", "IMPALA", "PPO", "SAC", "TD3"] # waterworld
    elif game == 'multiwalker':
        plt.plot(np.array([0,60000]),np.array([-102.05,-102.05]), label='Random') # multiwalker
        data = np.genfromtxt(os.path.join('./plot/data/multi_walker', 'gupta_ddpg.csv'), delimiter=',')
        plt.plot(data[:, 0], data[:, 1], 'k:', label='Gupta_DDPG')
        data = np.genfromtxt(os.path.join('./plot/data/multi_walker', 'gupta_trpo.csv'), delimiter=',')
        plt.plot(data[:, 0], data[:, 1], '--',color=[0.6,0.6,0.6,1], label='Gupta_TRPO')
        methods = ["A2C", "APEX_DDPG", "DDPG", "IMPALA", "PPO", "SAC", "TD3"] # multiwalker
    elif game == 'pursuit':
        plt.plot(np.array([0,60000]),np.array([31.03,31.03]), label='Random') # pursuit
        data = np.genfromtxt(os.path.join('./plot/data/pursuit', 'gupta_dqn.csv'), delimiter=',')
        plt.plot(data[:, 0], data[:, 1], 'k:', label='Gupta_DQN')
        data = np.genfromtxt(os.path.join('./plot/data/pursuit', 'gupta_trpo.csv'), delimiter=',')
        plt.plot(data[:, 0], data[:, 1], '--',color=[0.6,0.6,0.6,1], label='Gupta_TRPO')
        methods = ["A2C", "ADQN", "DQN", "IMPALA", "PPO", "RDQN"] # pursuit
    for y in os.listdir(sub_folder):
        if game not in y:
            continue
    '''
    if algorithm not in x:
        continue
    try:
        df = pd.read_csv(os.path.join(data_path+'/'+x,'progress.csv'))
        df = df[['episodes_total', "episode_reward_mean"]]
        data = df.to_numpy() 
        plt.plot(data[:, 0], data[:, 1], '--', label=algorithm+x[-1])       
    except:
        print("           ====== Exception")
        continue
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.title(game.capitalize())
    plt.xticks(ticks=[10000,20000,30000,40000,50000],labels=['10k','20k','30k','40k','50k'])
    plt.xlim(0,60000)
    plt.tight_layout()
    plt.legend(loc='lower right', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25)
    plt.margins(x=0)
    plt.savefig('SA_'+game+'_'+algorithm+'.png', bbox_inches = 'tight',pad_inches = .025)



# data = np.genfromtxt(os.path.join(data_path, 'gupta_ddpg.csv'), delimiter=',')
# plt.plot(data[:, 0], data[:, 1], '--', label='DDPG')

# data = np.genfromtxt(os.path.join(data_path, 'gupta_trpo.csv'), delimiter=',')
# plt.plot(data[:, 0], data[:, 1], '--', label='TRPO')

# df = pd.read_csv(os.path.join(data_path, 'SAC.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()
# filtered = scipy.signal.savgol_filter(data[:, 1], int(len(data[:, 1])/50), 4)
# plt.plot(data[:, 0], data[:, 1], label='SAC')

# df = pd.read_csv(os.path.join(data_path,'A2C.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()
# plt.plot(data[:, 0], data[:, 1], '--', label='A2C')

# df = pd.read_csv(os.path.join(data_path,'APEX_DDPG.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()
# plt.plot(data[:, 0], data[:, 1], label='ApeX-DDPG')

# df = pd.read_csv(os.path.join(data_path,'IMPALA.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()
# plt.plot(data[:, 0], data[:, 1], label='IMPALA')

# df = pd.read_csv(os.path.join(data_path, 'PPO1.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()
# plt.plot(data[:, 0], data[:, 1], label='PPO')

# df = pd.read_csv(os.path.join(data_path,'TD3.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()
# filtered = scipy.signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50),5)
# plt.plot(data[:, 0], data[:, 1], label='TD3')


# plt.xlabel('Episode')
# plt.ylabel('Average Total Reward')
# plt.title(game.capitalize())
# plt.xticks(ticks=[10000,20000,30000,40000,50000],labels=['10k','20k','30k','40k','50k'])
# plt.xlim(0,60000)
# plt.tight_layout()
# plt.legend(loc='lower right', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25)
# plt.margins(x=0)
# plt.savefig('all_single_'+game+'.png', bbox_inches = 'tight',pad_inches = .025)
