import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import pdb
import json
import numpy as np
import torch
import torch.nn.functional as F
import os

env = gym.make("CarRacing-v3")
observation, info = env.reset()
one_rollout = []
training_data = []


def make_env():
    def _init():
        env = gym.make('CarRacing-v3')
        return env
    return _init

def collect_rollout_paralell(num_rollouts=200, num_parallel=16, max_steps=1000):
    envs = AsyncVectorEnv([make_env() for _ in range(num_parallel)])

    os.makedirs('data/batches', exist_ok=True)
    os.makedirs('data/actions', exist_ok=True)

    all_frames = []
    num_batches = num_rollouts // num_parallel

    for batch_idx in range(num_batches):

        observations, infos = envs.reset()
        batch_frames = []
        actions_list = []

        for step in range(max_steps):
            actions = np.array([envs.single_action_space.sample() for _ in range(num_parallel)])
            observations, rewards, terminated, truncated, infos = envs.step(actions)
            obs_tensor = torch.from_numpy(observations).permute(0, 3, 1, 2).float() / 255.0
            resized_obs = F.interpolate(obs_tensor, size=(64, 64), mode='bilinear', align_corners=False)

            actions_list.append(actions)
            batch_frames.append(resized_obs)

        batch_frames = torch.stack(batch_frames)
        batch_frames = batch_frames.numpy()
        actions_list = np.array(actions_list)

        batch_frames = batch_frames.reshape(-1, 3, 64, 64)

        np.save(f'data/batches/batch_{batch_idx+1}.npy', batch_frames)
        np.save(f'data/actions/action_{batch_idx+1}.npy', actions_list)
        print(f"Saved batch {batch_idx+1}/{num_batches}")

        del batch_frames


    envs.close()

if __name__ == '__main__':
    collect_rollout_paralell()




