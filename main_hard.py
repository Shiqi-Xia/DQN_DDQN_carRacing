import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from utils import ReplayMemory, plot_durations
from DQN import DQN


def select_action(state):
    global steps_done
    # state = state.view(1, -1)
    # print("select_action state:",state.shape)
    sample = random.random()
    
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)




env = gym.make("RealDrive", render_mode='human', continuous=False)
#env = gym.make("CarRacing-v2", render_mode='human', continuous=False)


# env = gym.make("CarRacing-v2",render_mode='rgb_array',continuous=False)
# env = gym.make("ALE/Breakout-v5",render_mode='rgb_array',obs_type="grayscale")
# env = gym.make("ALE/Breakout-v5",render_mode='human',frameskip=3,obs_type="grayscale")
# set up matplotlib
def trans(array):
    #return array
    return np.mean(array, axis=2)


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
# if GPU is to be used
device = torch.device("cuda")
print("device:", device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))



BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2000
TAU = 0.05
LR = 9e-4
SEED = 3117
n_actions = env.action_space.n
state, info = env.reset()
n_frames_to_stack = 10
# n_observations = len(state)
n_observations = (96, 96)
print("n_actions:", n_actions)
print("n_observations:", n_observations)
print("observations:", state.shape)
policy_net = DQN(n_observations, n_actions, n_frames_to_stack).to(device)
target_net = DQN(n_observations, n_actions, n_frames_to_stack).to(device)
# if policy_net.load_state_dict(torch.load("modelcar10.pt")):
#     print("load modelcar10.pt")
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(capacity=2000)
steps_done = 0




episode_durations = []




def DQN_optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    if batch.next_state is not None:
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
    # print("batch.state:",batch.state)
    state_batch = torch.cat(batch.state)
    # print("state_batch:",state_batch.shape)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()



def DDQN_optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    if batch.next_state is not None:
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
    # print("batch.state:",batch.state)
    state_batch = torch.cat(batch.state)
    # print("state_batch:",state_batch.shape)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        max_action = policy_net(non_final_next_states).max(1)[1].view(-1, 1)
        next_state_values[non_final_mask] = torch.squeeze(target_net(non_final_next_states).gather(1, max_action))
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()



if torch.cuda.is_available():
    num_episodes = 2000  #600000
else:
    num_episodes = 50

total_reward = 0
max_reward = 0
total_t = 0
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset(seed=3117)
    state = trans(state)
    frames = []
    new_frame = torch.tensor(state, dtype=torch.float32, device=device)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames[9] = frames[9]  # 移除不必要的批次维度
    stacked_state = torch.stack(frames).unsqueeze(0)
    # print("stacked_state",stacked_state.shape)
    stacked_next_state = None
    for t in count():
        # print("stacked_state:",stacked_state.shape)
        action = select_action(stacked_state)
        # print("action",action.item())
        observation, reward, terminated, truncated, _ = env.step(action.item())
        observation = trans(observation)
        #print(observation)
        done = (terminated or truncated)
        if done:
            next_state = None
        else:
            # torch.stack(frames).unsqueeze(0).to(device)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device)
            frames.pop(0)
            frames.append(next_state)
        # Move all tensors to the same device (if necessary)
        reward_tensor = torch.tensor([reward], device=device)
        total_reward += reward
        frames[9] = frames[9].squeeze()  # 移除不必要的批次维度
        stacked_next_state = torch.stack(frames).unsqueeze(0)
        memory.push(Transition, stacked_state, action, stacked_next_state, reward_tensor)
        stacked_state = stacked_next_state
        #DQN_optimize_model()
        DDQN_optimize_model()
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)
        if done:
            total_t += t
            print(f'i_episode: {i_episode}, total_reward: {total_reward},   t: {t}  total_t: {total_t}  ')
            episode_durations.append(total_reward)
            if total_reward > max_reward:
                max_reward = total_reward
                print(f'i_episode: {i_episode}, total_reward: {total_reward},max_reward: {max_reward}  ')
                if max_reward > 10.0:
                    torch.save(policy_net.state_dict(), "modelcar10.pt")
                    print("save model")
            total_reward = 0
            plot_durations(episode_durations)
            break
print('Complete')
plt.savefig('real_drive_DDQN_hard_old.pdf', format="pdf")
plot_durations(episode_durations, show_result=True)
plt.ioff()
plt.show()