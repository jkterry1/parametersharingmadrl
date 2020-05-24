from sisl_games.waterworld import waterworld
from sisl_games.multiwalker import multiwalker
from sisl_games.pursuit import pursuit
import ray
from ray.tune.registry import register_trainable, register_env
import ray.rllib.agents.dqn as dqn  # DQNTrainer
import ray.rllib.agents.ppo as ppo  # PPOTrainer
import ray.rllib.agents.a3c.a2c as a2c  # A2CTrainer
import ray.rllib.agents.sac as sac  # SACTrainer
import ray.rllib.agents.ddpg as ddpg  # TD3Trainer
import ray.rllib.agents.ddpg.td3 as td3  # TD3Trainer
import ray.rllib.agents.ddpg.apex as apex  # ApexDDPGTrainer
import ray.rllib.agents.sac as sac  # SACTrainer
import ray.rllib.agents.impala as impala  # ImpalaTrainer
import os
import pickle
import numpy as np
import pandas as pd
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.utils import try_import_tf
# from parameterSharingMultiwalker import MLPModel
import sys
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "serif"

tf = try_import_tf()

env_name = "multiwalker"
algorithm = sys.argv[1].upper()
is_single_agent = bool(sys.argv[2])
trial_folder = sys.argv[3] 
iterss = int(sys.argv[4])

methods = ["A2C", "APEX_DDPG", "DDPG", "IMPALA", "PPO", "SAC", "TD3"]

assert algorithm in methods, "{} is not part of {}".format(algorithm, methods)


# path should end with checkpoint-<> data file

ray.init()

data_path = "../ray_results/multiwalker/"+trial_folder
# data_path = "./ray_results/"+env_name+'/'+algorithm

# checkpoint_path = "./ray_results/transfer/test"+str(i)+"/checkpoint_"+str(iters[i])+'/checkpoint-'+str(iters[i])

# checkpoint_path = data_path+"/checkpoint_"+str(iters[j])+'/checkpoint-'+str(iters[j])
checkpoint_path = data_path+"/checkpoint_"+str(iterss)+'/checkpoint-'+str(iterss)
print(checkpoint_path)
# TODO: see ray/rllib/rollout.py -- `run` method for checkpoint restoring

# register env -- For some reason, ray is unable to use already registered env in config
def env_creator(args):
    if env_name == 'waterworld':
        return waterworld.env()
    elif env_name == 'multiwalker':
        return multiwalker.env()
    elif env_name == 'pursuit':
        return pursuit.env()
    

env = env_creator(1)
register_env(env_name, env_creator)

# get the config file - params.pkl
config_path = os.path.dirname(checkpoint_path)
config_path = os.path.join(config_path, "../params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

class MLPModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        last_layer = tf.layers.dense(
                input_dict["obs"], 400, activation=tf.nn.relu, name="fc1")
        last_layer = tf.layers.dense(
            last_layer, 300, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")
        return output, last_layer

ModelCatalog.register_custom_model("MLPModel", MLPModel)

print(env.observation_space_dict)
# exit()

# RLAgent = ppo.PPOTrainer(env=env_name, config=config)
if algorithm == 'DQN':
    RLAgent = dqn.DQNTrainer(env=env_name, config=config)
elif algorithm == 'DDPG':
    RLAgent = ddpg.DDPGTrainer(env=env_name, config=config)
elif algorithm == 'PPO':
    RLAgent = ppo.PPOTrainer(env=env_name, config=config)
elif algorithm == 'A2C':
    RLAgent = a2c.A2CTrainer(env=env_name, config=config)
elif algorithm == 'APEX_DDPG':
    RLAgent = apex.ApexDDPGTrainer(env=env_name, config=config)
elif algorithm == 'IMPALA':
    RLAgent = impala.ImpalaTrainer(env=env_name, config=config)
elif algorithm == 'SAC':
    RLAgent = sac.SACTrainer(env=env_name, config=config)
elif algorithm == 'TD3':
    RLAgent = td3.TD3Trainer(env=env_name, config=config)
print('before restore')
RLAgent.restore(checkpoint_path)

num_runs = 10
totalRewards = np.empty((num_runs,))

for j in range(num_runs):

    # init obs, action, reward
    observations = env.reset()
    rewards, action_dict = {}, {}
    for agent_id in env.agent_ids:
        assert isinstance(agent_id, int), "Error: agent_ids are not ints."
        # action_dict = dict(zip(env.agent_ids, [np.array([0,1,0]) for _ in range(len(env.agent_ids))]))  # no action = [0,1,0]
        action_dict = dict(zip(env.agent_ids, [env.action_space_dict[i].sample() for i in env.agent_ids]))
        rewards[agent_id] = 0

    totalReward = 0
    done = False
    # action_space_len = 3 # for all agents

    # TODO: extra parameters : /home/miniconda3/envs/maddpg/lib/python3.7/site-packages/ray/rllib/policy/policy.py

    iteration = 0
    while not done:
        action_dict = {}
        # compute_action does not cut it. Go to the policy directly
        for agent_id in env.agent_ids:
            #print("id {}, obs {}, rew {}".format(agent_id, observations[agent_id], rewards[agent_id]))
            action, _, _ = RLAgent.get_policy("policy_{}".format(agent_id if is_single_agent else 0)).compute_single_action(observations[agent_id], prev_reward=rewards[agent_id]) # prev_action=action_dict[agent_id]
            #print("action: ", action)
            for i in range(len(action)):
                if action[i] < -1:
                    action[i] = -1
                elif action[i] > 1:
                    action[i] = 1
            action_dict[agent_id] = action

        observations, rewards, dones, info = env.step(action_dict)
        #env.render()
        totalReward += sum(rewards.values())
        done = any(list(dones.values()))
        # if sum(rewards.values()) > 0:
        #     print("rewards", rewards)
        # print("iter:", iteration, sum(rewards.values()))
        iteration += 1
    totalRewards[j] = totalReward

env.close()

print("\n\ndone: ", done, ', Mean Total Reward: ',np.mean(totalRewards), 'Total Reward: ', totalRewards)
print("\nMean Total Reward: ", np.mean(totalRewards))

df = pd.read_csv(os.path.join(data_path,'progress.csv'))
df = df[['training_iteration', "episode_reward_mean", "episodes_total"]]
rew = df.loc[df['training_iteration'] == iterss, ['episode_reward_mean']]
# print(rew)
rew = rew.to_numpy()[0][0]
epi = df.loc[df['training_iteration'] == iterss, ['episodes_total']]
rew_max = df['episode_reward_mean'].max()
epi_max = df.loc[df['episode_reward_mean'].idxmax(), ['episodes_total','training_iteration']]
epi = epi.to_numpy()[0][0]
print("Progress Report Reward: ", rew)
print("Reward Error Factor: ", np.mean(totalRewards)/rew, '\n')
print("Max of ",rew_max, " at ", int(epi_max[0]), " episodes (",int(epi_max[1]),' iterations)')
print("Episodes Total: ", epi, "\n\n")

df = pd.read_csv(os.path.join(data_path,'progress.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], '--', label=algorithm)       

plt.plot(np.array([0,60000]),np.array([-102.05,-102.05]), label='Random') # multiwalker
data = np.genfromtxt(os.path.join('./plot/data/multi_walker', 'gupta_ddpg.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], 'k:', label='Gupta_DDPG')
data = np.genfromtxt(os.path.join('./plot/data/multi_walker', 'gupta_trpo.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], '--',color=[0.6,0.6,0.6,1], label='Gupta_TRPO')

plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.title(env_name.capitalize())
plt.xticks(ticks=[10000,20000,30000,40000,50000],labels=['10k','20k','30k','40k','50k'])
plt.xlim(0,60000)
plt.tight_layout()
plt.legend(loc='lower right', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25)
plt.margins(x=0)
plt.savefig(env_name+'_'+algorithm+'.png', bbox_inches = 'tight',pad_inches = .025)


