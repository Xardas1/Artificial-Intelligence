import numpy as np
import gymnasium as gym
import pdb
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim
from random import sample
from dqn_agent import image_preprocess, stack_images, create_input_image, DQN, create_replay_memory, extract_states, extract_actions, create_correct_pred_vectorized, extract_rewards, extract_next_states, calculate_y_target, e_greedy_policy
from dqn_agent import epsilon_decay

env = gym.make("ALE/Pong-v5", render_mode=None)
print("Action Space:", env.action_space)
print("Possible actions:", list(range(env.action_space.n)))
print("Observation Space:", env.observation_space)

epochs = 5000000
learning_rate = 0.0001
gamma = 0.99
epsilon_start = 1.0
epsilon_final = 0.1
decay_steps = 1000000
epsilon = 1
decay_rate = (epsilon_start - epsilon_final) / decay_steps

list_to_stack = []
st_prev_list = deque(maxlen=2)

model = DQN(in_channels=4, num_actions=6)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

target_model = DQN(in_channels=4, num_actions=6)
target_model.load_state_dict(model.state_dict())

replay_memory = deque(maxlen=1000000)

obs, info = env.reset()
first_frame = image_preprocess(obs)
for _ in range(4):
    list_to_stack.append(first_frame)

terminated = False
truncated = False

for i in range(1, epochs):
    
    image_preprocessed = image_preprocess(obs)
    stacked_images = create_input_image(image_preprocessed, list_to_stack)
    
    if stacked_images is None:
        continue
    
    model_batch = torch.unsqueeze(stacked_images, 0)
    actions = model(model_batch)
    
    action = e_greedy_policy(actions, epsilon, env)
    
    epsilon = epsilon_decay(epsilon, decay_rate)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    st_prev_list.append(stacked_images)
    replay_memory = create_replay_memory(st_prev_list, replay_memory, action, reward)
    
    if terminated or truncated:
        obs, info = env.reset()
        list_to_stack = []
        first_frame = image_preprocess(obs)
        for _ in range(4):
            list_to_stack.append(first_frame)
        st_prev_list.clear()
    
    if i % 100 == 0:
        target_model.load_state_dict(model.state_dict())
        
    if len(replay_memory) >= 32 and i % 4 == 0:
        batch = sample(replay_memory, 32)
        
        model_states = extract_states(batch)
        model_actions = extract_actions(batch)
        model_rewards = extract_rewards(batch)
        model_next_states = extract_next_states(batch)
        
        y_pred_almost = model(model_states)
        y_pred = create_correct_pred_vectorized(y_pred_almost, model_actions)
        
        q_target = target_model(model_next_states)
        q_max = q_target.max(dim=1).values
        y_target = calculate_y_target(model_rewards, gamma, q_max)
        
        loss = loss_fn(y_pred, y_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if i % 10000 == 0:
        torch.save(model.state_dict(), "pong_model.pth")
        print(f"Step {i}, Epsilon: {epsilon:.3f}")
    



