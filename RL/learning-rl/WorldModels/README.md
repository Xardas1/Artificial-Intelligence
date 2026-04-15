# World Models Implementation

A clean, educational implementation of the World Models architecture by David Ha & Jürgen Schmidhuber. This project demonstrates the three-component approach to reinforcement learning in latent spaces.

## Overview

World Models decompose reinforcement learning into three main components:

1. **Vision Model (VAE)** - Compresses high-dimensional observations into compact latent representations
2. **Dynamics Model (RNN-MDN)** - Learns to predict future latent states
3. **Controller (Agent)** - Makes decisions based on latent state predictions

## Architecture

```
Observations (64x64x3) 
    |
    V
VAE (Encoder) -> Latent Space (32-dim) -> VAE (Decoder)
    |
    V
RNN-MDN (Predicts next latent state)
    |
    V
Controller (Outputs actions)
```

## Components

### 1. VAE (Vision Model) - `VAE.py`
- **Purpose**: Learn compressed representations of visual observations
- **Input**: 64x64x3 RGB images from CarRacing-v3
- **Output**: 32-dimensional latent vectors `z`
- **Architecture**: 4 convolutional layers (encoder) + 4 transpose conv layers (decoder)
- **Loss**: Reconstruction loss + KL divergence (VAE objective)

### 2. RNN-MDN (Dynamics Model) - `RNN-MDN.py`
- **Purpose**: Model transition dynamics in latent space
- **Input**: Current latent state `z_t` + action `a_t` (35-dim total)
- **Output**: Predicted next latent state `z_{t+1}` with uncertainty
- **Architecture**: Simple RNN (256 hidden units) + Mixture Density Network (5 Gaussians)
- **Key Feature**: Outputs probability distributions instead of point predictions

### 3. Controller (Agent) - `Controler.py`
- **Purpose**: Policy that maps latent states to actions
- **Input**: Sequence of latent states (288-dim = 9×32)
- **Output**: 3-dimensional action for CarRacing
- **Training**: Designed for CMA-ES evolution (simplified in this implementation)

### 4. Data Collection - `DataCollection.py`
- **Purpose**: Generate training data from random rollouts
- **Environment**: CarRacing-v3
- **Processing**: Resize observations to 64x64, save as numpy arrays

## Key Concepts

### Latent Space Learning
Instead of learning directly from high-dimensional pixels, the VAE learns to compress observations into a compact 32-dimensional latent space. This makes learning dynamics much more efficient.

### Mixture Density Networks
The RNN doesn't output single predictions but probability distributions (mixture of Gaussians). This captures uncertainty in future predictions, which is crucial for robust decision-making.

### Evolutionary Training
The controller is optimized using evolutionary strategies (CMA-ES) rather than gradient-based methods, allowing for non-differentiable decision-making.

## Installation

### Quick Start (Demo Only)
```bash
pip install -r requirements.txt
python simple_demo.py
```

### Full Training Setup
```bash
# Install core dependencies (always works)
pip install -r requirements.txt

# For CarRacing-v3 environment (optional, may fail)
pip install swig
pip install "gymnasium[box2d]"
```

**Note**: Box2D installation may fail on some systems due to compilation issues. If you get Box2D errors:
- **Use `simple_demo.py`** - Works without environment dependencies
- **Skip `DataCollection.py`** - Requires working Box2D installation
- **Architecture demonstration** - Demo shows all components without environment requirements

### Manual Installation
```bash
pip install torch>=1.9.0
pip install gymnasium>=0.26.0
pip install "gymnasium[atari]>=0.26.0"
pip install torchvision>=0.10.0
pip install numpy>=1.21.0
pip install cma>=3.0.0
# Optional (may fail):
pip install swig>=4.0.0
pip install "gymnasium[box2d]>=0.26.0"
```

## Usage

### 1. Collect Training Data
```bash
python DataCollection.py
```

### 2. Train VAE
```bash
python VAE.py
```

### 3. Train RNN-MDN
```bash
python RNN-MDN.py
```

### 4. Test Controller
```bash
python Controler.py
```

## Implementation Notes

This is a **research prototype** focused on clarity and educational value:

- **Simplified RNN**: Uses basic RNN instead of LSTM for clarity
- **Fixed dimensions**: Uses constant batch sizes for easier understanding
- **Minimal dependencies**: Core implementation without production complexity
- **Educational focus**: Clean, readable code that explains the concepts

## Differences from Original Paper

- Simplified RNN architecture (basic RNN vs LSTM)
- Reduced hyperparameter complexity
- Focus on single-environment demonstration
- Streamlined for educational purposes

## File Structure

```
World Models/
|
|-- VAE.py           # Vision model implementation
|-- RNN-MDN.py        # Dynamics model with mixture density
|-- Controler.py     # Agent controller
|-- DataCollection.py # Training data generation
|-- README.md        # This file
|-- vae_weights.pth  # Pre-trained VAE weights
|-- frames.npy       # Training data
```

## Research Context

World Models represent a paradigm shift in RL by:

1. **Separating concerns**: Vision, dynamics, and control are learned independently
2. **Learning in latent space**: More efficient than pixel-based learning
3. **Model-based planning**: Can imagine future scenarios before acting
4. **Uncertainty modeling**: Mixture density networks capture prediction confidence

This implementation captures these core concepts while maintaining code clarity.

## References

- [World Models Paper](https://worldmodels.github.io/) - Ha & Schmidhuber, 2018
- [NIPS 2018 Proceedings](https://papers.nips.cc/paper/2018/hash/2f5d1ecdf3289b2197dacc0c09f78760-Abstract.html)

---

**Note**: This implementation prioritizes educational clarity over production readiness. Perfect for understanding the World Models concept!
