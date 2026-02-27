import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import pdb
import json
import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F
import os

env = gym.make("CarRacing-v3")
observation, info = env.reset()
one_rollout = []
training_data = []

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def make_env():
    def _init():
        env = gym.make('CarRacing-v3')
        return env
    return _init

def collect_rollout_paralell(num_rollouts=10000, num_parallel=16, max_steps=1000):
    envs = AsyncVectorEnv([make_env() for _ in range(num_parallel)])

    os.makedirs('D:/WorldModels/batches', exist_ok=True)
    os.makedirs('D:/WorldModels/actions', exist_ok=True)

    all_frames = []
    num_batches = num_rollouts // num_parallel

    for batch_idx in range(num_batches):

        observations, infos = envs.reset()
        batch_frames = []
        actions_list = []

        for step in range(max_steps):
            actions = np.array([envs.single_action_space.sample() for _ in range(num_parallel)])
            observations, rewards, terminated, trunacted, infos = envs.step(actions)
            obs_tensor = torch.from_numpy(observations).permute(0, 3, 1, 2).float() / 255.0
            resized_obs = F.interpolate(obs_tensor, size=(64, 64), mode='bilinear', align_corners=False)

            actions_list.append(actions)
            batch_frames.append(resized_obs)

        batch_frames = np.array(batch_frames, dim=0)
        actions_list = np.array(actions_list)

        batch_frames = batch_frames.reshape(-1, 3, 64, 64)

        np.save(f'D:/WorldModels/batches/batch_{batch_idx+1}.npy', batch_frames)
        np.save(f'D:/WorldModels/actions/action_{batch_idx+1}.npy', actions_list)
        print(f"Saved batch {batch_idx+1}/{num_batches}")

        del batch_frames


    envs.close()

if __name__ == '__main__':
    collect_rollout_paralell()

    #num_batches = 10000 // 16
    #all_frames = []
    #for i in range(num_batches):
       # batch = np.load(f'batches/batch_{i}.npy')
        #all_frames.append(batch)
#
    #all_frames = np.concatenate(all_frames, axis=0)

    #np.save('frames.npy', all_frames)





#with open("training_data.json", "w") as f:
    # json.dump([item.tolist() for item in training_data], f)



