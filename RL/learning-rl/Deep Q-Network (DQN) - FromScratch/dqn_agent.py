import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

env = gym.make("ALE/Pong-v5", render_mode=None)

class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

input_size = 84
num_classes = 6
batch_size = 32
num_epochs = 100

def changing_to_grayscale(image):
    gray = 0.244 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return gray

def image_preprocess(image):
    image = changing_to_grayscale(image)
    preprocesse_image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
    return preprocesse_image

def stack_images(list_to_stack):
    stacked_images = np.stack((list_to_stack), axis=0)
    stacked_images = torch.tensor(stacked_images, dtype=torch.float32)
    return stacked_images

# FIX: Zawsze zwracaj coś albo None
def create_input_image(image, list_to_stack):
    if len(list_to_stack) >= 4:
        list_to_stack.pop(0)
    list_to_stack.append(image)
    
    if len(list_to_stack) == 4:
        stacking_images = stack_images(list_to_stack)
        return stacking_images
    return None

def create_replay_memory(state_list, replay_memory, action, reward):
    if len(state_list) == 2:
        replay_memory.extend([
            (state_list[0], torch.tensor(action), torch.tensor(reward), state_list[1])
        ])
    return replay_memory

def extract_states(batch):
    states_st = torch.stack([batch[i][0] for i in range(len(batch))])
    return states_st

def extract_actions(batch):
    actions_at = torch.stack([batch[i][1] for i in range(len(batch))])
    return actions_at

def extract_rewards(batch):
    reward_rt = torch.stack([batch[i][2] for i in range(len(batch))])
    return reward_rt

def extract_next_states(batch):
    next_states_st1 = torch.stack([batch[i][3] for i in range(len(batch))])
    return next_states_st1

def create_correct_pred(y_pred, actions):
    empty_list = []
    for j, i in enumerate(actions):
        empty_list.append(y_pred[j][i])
    return empty_list

def create_correct_pred_vectorized(y_pred, actions):
    row_indices = np.arange(len(actions))
    return y_pred[row_indices, actions]

def calculate_y_target(reward, gamma, q_max):
    y_target = reward + gamma * q_max
    return y_target

def e_greedy_policy(actions, epsilon):
    if np.random.random() > epsilon:
        actions = actions.detach().cpu().numpy()
        action = np.argmax(actions)
    else:
        action = env.action_space.sample()
    return action

def epsilon_decay(epsilon, decay_rate):
    epsilon = max(0.1, epsilon - decay_rate)
    return epsilon
