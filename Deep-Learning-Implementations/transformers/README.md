# Transformers Implementation from Scratch

This is a complete implementation of the Transformer architecture from scratch using PyTorch. The implementation follows the original "Attention Is All You Need" paper and includes a character-level language model.

## Architecture Overview

The model consists of:

### Core Components

1. **DawidGPT**: Main transformer model
   - Character-level embeddings (512 dimensions)
   - 4 transformer blocks
   - 8 attention heads per block
   - Positional encodings
   - Layer normalization and residual connections

2. **SelfAttention**: Multi-head self-attention mechanism
   - 8 parallel attention heads
   - 64 dimensions per head
   - Scaled dot-product attention
   - Softmax normalization

3. **FeedForwardNet**: Position-wise feed-forward network
   - Linear expansion: 512 -> 2048 dimensions
   - ReLU activation
   - Linear projection: 2048 -> 512 dimensions

## Key Features

- **Multi-Head Attention**: 8 heads allow the model to focus on different positions
- **Positional Encodings**: Sinusoidal encodings to capture sequence order
- **Residual Connections**: 0.1 scaling factor to prevent gradient explosion
- **Layer Normalization**: Pre-normalization for stable training
- **Character-level Tokenization**: Simple character-to-index mapping

## Training Details

- **Dataset**: Sample text data (can be replaced with custom dataset)
- **Batch Size**: 32 sequences
- **Sequence Length**: 128 characters
- **Learning Rate**: 0.0003 (Adam optimizer)
- **Loss Function**: Cross-entropy loss
- **Device**: CUDA-enabled (falls back to CPU)

## Usage

### Training

```bash
# Train the model
python train.py
```

This will:
- Create sample training data
- Train the transformer for 1000 epochs
- Save model weights as `dawid_gpt_model.pt`
- Save vocabulary as `vocab.pt`

### Text Generation

```bash
# Generate text with trained model
python demo.py --prompt "The quick brown fox" --length 200 --temperature 0.8
```

Options:
- `--prompt`: Starting text for generation
- `--length`: Number of characters to generate
- `--temperature`: Sampling temperature (0.1-2.0, lower = more deterministic)
- `--model_path`: Path to trained model (default: dawid_gpt_model.pt)
- `--vocab_path`: Path to vocabulary file (default: vocab.pt)

## Implementation Notes

### What Works Well
- Complete transformer architecture from scratch
- Proper multi-head attention with correct dimensions
- Positional encodings following the original paper
- Stable training with residual connections and normalization

### Areas for Improvement
- Character-level tokenization limits vocabulary size
- No dropout regularization
- Fixed sequence length (128 characters)
- Simple training loop without validation

### Key Fixes Applied
- Fixed attention score scaling (using head_dim instead of 128)
- Removed hardcoded Windows file paths
- Added proper batch size handling
- Removed debug code (pdb.set_trace())
- Improved parameter initialization

## Files

- `model.py`: Core transformer implementation (DawidGPT, SelfAttention, FeedForwardNet)
- `train.py`: Training script to train the model
- `demo.py`: Demo script for text generation with trained model
- `dawid_gpt_model.pt`: Trained model weights (generated after training)
- `vocab.pt`: Vocabulary mappings (generated after training)

## Dependencies

- PyTorch
- NumPy

## Future Enhancements

- Add dropout regularization
- Implement subword tokenization (BPE)
- Add validation and early stopping
- Implement learning rate scheduling
- Add more sophisticated text generation strategies

## Understanding the Code

This implementation demonstrates understanding of:

1. **Self-Attention Mechanism**: How queries, keys, and values interact
2. **Multi-Head Attention**: Parallel attention computation
3. **Positional Encodings**: How transformers handle sequence order
4. **Residual Connections**: How gradients flow through deep networks
5. **Layer Normalization**: Stabilizing deep network training

The code is intentionally kept simple to focus on the core transformer concepts rather than engineering optimizations.
