import sys

filename = sys.argv[1]

reward_found = False
episodes_found = False

with open(filename,'r') as f:
    for line in reversed(list(f)):
        if "episode_reward_mean" in line:
            mean_reward = float(line.split()[1])
            reward_found = True
        if "episodes_total" in line:
            total_episodes = int(line.split()[1])
            episodes_found = True
        if reward_found and episodes_found:
            break

print("Average Total Reward: ", mean_reward, " after ", total_episodes, " episodes.")

