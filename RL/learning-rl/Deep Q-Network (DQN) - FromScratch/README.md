# Deep Q-Network (DQN) - From Scratch

A simple implementation of Deep Q-Network (DQN) reinforcement learning algorithm from scratch to play Atari Pong using PyTorch.

## Overview

This project implements the DQN algorithm as described in the original paper "Playing Atari with Deep Reinforcement Learning" by DeepMind. The agent learns to play Pong by processing raw pixel inputs and learning optimal actions through experience replay and target networks.

## Features

- **Image Preprocessing**: Converts raw Atari frames to grayscale 84x84 images
- **Experience Replay**: Stores and samples transitions for stable training
- **Target Network**: Separate network for stable Q-value estimation
- **Epsilon-Greedy Policy**: Balance between exploration and exploitation
- **Frame Stacking**: Uses 4 consecutive frames as input to capture motion

## Project Structure

```
Deep Q-Network (DQN) - FromScratch/
|-- dqn_agent.py          # DQN model and utility functions
|-- train.py              # Main training script
|-- pong_model.pth        # Pre-trained model weights
|-- Network Architecture.png  # Model architecture diagram
|-- requirements.txt      # Python dependencies
|-- README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Deep Q-Network (DQN) - FromScratch"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the DQN agent from scratch:

```bash
python train.py
```

The training will:
- Run for 5 million episodes
- Save model weights every 10,000 steps to `pong_model.pth`
- Print progress updates with current step and epsilon value
- Use experience replay with buffer size of 1,000,000 transitions

### Key Training Parameters

- **Learning Rate**: 0.0001
- **Discount Factor (Gamma)**: 0.99
- **Epsilon Start**: 1.0 (full exploration)
- **Epsilon Final**: 0.1 (minimal exploration)
- **Decay Steps**: 1,000,000
- **Batch Size**: 32
- **Target Network Update**: Every 100 steps

## Model Architecture

The DQN uses a convolutional neural network with:
- Conv2d(4, 32, kernel_size=8, stride=4) + ReLU
- Conv2d(32, 64, kernel_size=4, stride=2) + ReLU  
- Conv2d(64, 64, kernel_size=3, stride=1) + ReLU
- Flatten + Linear(64*7*7, 6)

Input: 4 stacked 84x84 grayscale frames
Output: Q-values for 6 possible actions in Pong

## Implementation Details

### Image Preprocessing
1. Convert RGB to grayscale using standard weights
2. Resize to 84x84 pixels using INTER_AREA interpolation
3. Stack 4 consecutive frames to capture motion dynamics

### Experience Replay
- Stores transitions as (state, action, reward, next_state)
- Uniform random sampling for training batches
- Maximum buffer size of 1,000,000 transitions

### Training Loop
1. Preprocess current observation and stack frames
2. Select action using epsilon-greedy policy
3. Execute action and observe reward/next state
4. Store transition in replay memory
5. Sample random batch and train network
6. Update target network periodically
7. Decay epsilon for less exploration over time

## Dependencies

- `numpy` - Numerical computations
- `gymnasium` - Atari environment
- `torch` - Deep learning framework
- `opencv-python` - Image processing
- `ale-py` - Atari Learning Environment
- `matplotlib` - Visualization (if needed)

## Notes

- This is a educational implementation focused on clarity over performance
- Training time can be significant (hours to days depending on hardware)
- GPU acceleration recommended for faster training
- Model weights are saved periodically and can be loaded for continued training

## Demo Status

This project is demo-ready and suitable for:
- Learning DQN fundamentals
- Educational purposes
- Experimentation with hyperparameters
- Understanding reinforcement learning concepts

**Not production-ready** for:
- Competitive performance
- Real-time applications
- Commercial deployment

## License

This project is for educational purposes. Please refer to the original DQN paper and OpenAI Gym licensing terms.
