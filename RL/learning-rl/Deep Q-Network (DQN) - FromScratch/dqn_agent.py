import numpy as np
import matplotlib.pyplot as plt
import gym
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

env = gym.make("ALE/Pong-v5", render_mode="human")
obs = env.reset()


class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()

        # First convolutional layer: 4 input channels, 32 output channels, 8x8 kernel, stride 4, padding=0
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0)
        # Second convolutional layer: 32 input channels, 64 output channels, 4x4 kernel, stride=2, padding=0
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        # Third convolutional layer: 64 inputchannels, 64 output channels, 3x3 kernel, stride=1, padding=0
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        # FC layer
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
learning_rate = 0.0001
batch_size = 32
num_epochs = 100


model = DQN(in_channels=4, num_actions=6)

loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)




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
    stacked_images = torch.unsqueeze(stacked_images, 0)
    return stacked_images

def create_input_image(image, list_to_stack):
    if len(list_to_stack) % 4 != 0 or len(list_to_stack) == 0:
        list_to_stack.append(image)
    else:
        list_to_stack.pop(0)
        list_to_stack.append(image)
    if len(list_to_stack) % 4 == 0:
        stacking_images = stack_images(list_to_stack)
        return stacking_images

def create_replay_memory(state_list, replay_memory, action, reward):
    if len(state_list) == 2:
        replay_memory.extend([(state_list[0], torch.tensor(action), torch.tensor(reward), state_list[1])])
    return replay_memory

def convert_to_tensor(batch):
    states_st  = torch.tensor([np.stack(batch[i][0]) for i in range(len(batch))])
    #action_at  = torch.tensor([np.stack(batch[j][1]) for j in range(len(batch))])
    #reward_rt  = torch.tensor([np.stack(batch[k][2]) for k in range(len(batch))])
    #states_st1 = torch.tensor([np.stack(batch[l][3]) for l in range(len(batch))])
    return states_st
#env.render()






#plt.imshow(preprocessed_image, cmap="gray")  # Add cmap if grayscale
#plt.axis("off")
#plt.show()