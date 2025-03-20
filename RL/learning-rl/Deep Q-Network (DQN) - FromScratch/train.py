import numpy as np
import gym
import pdb
from collections import deque
import torch
from dqn_agent import image_preprocess, stack_images, create_input_image, DQN, create_replay_memory, convert_to_tensor
from random import sample


env = gym.make("ALE/Pong-v5", render_mode="human")
obs = env.reset()


print("Action Space:", env.action_space)
print("Possible actions:", list(range(env.action_space.n)))
print("Observation Space:", env.observation_space)

list_to_stack = []
et = []
st_prev_list = deque(maxlen=2)
model = DQN(in_channels=4, num_actions=6)
replay_memory = deque(maxlen=1000000)
d = 0

for i in range(1, 1000):
    env.render()
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    image_preprocessed= image_preprocess(obs)

    #stacked_images = create_input_image(image_preprocessed, list_to_stack)
    image_preprocessed= image_preprocess(obs)
    stacked_images = create_input_image(image_preprocessed, list_to_stack)

    if i >= 4:
        st_prev_list.append(stacked_images)
        


    replay_memory = create_replay_memory(st_prev_list, replay_memory, action, reward)
 
    if len(replay_memory) >= 32 and i % 4 == 0:
        batch = sample(replay_memory, 32)
        #pdb.set_trace()
        states_st = convert_to_tensor(batch)
        #pdb.set_trace()
        #y_pred = model(stacked_images)
        #pdb.set_trace

    if terminated or truncated:
        obs = env.reset()





#env.close()
#print(env.action_space)