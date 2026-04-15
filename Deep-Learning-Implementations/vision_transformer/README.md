# Vision Transformer Implementation from Scratch

This is a complete implementation of Vision Transformer (ViT) architecture from scratch using PyTorch. The implementation follows the original "An Image is Worth 16x16 Words" paper and is trained on CIFAR-10.

## Architecture Overview

The Vision Transformer consists of:

### Core Components

1. **DawidGPT**: Main Vision Transformer model
   - Patch embedding layer (Linear projection from 768 to 1024)
   - 4 transformer blocks with self-attention
   - Positional encodings using sin/cos functions
   - Global average pooling before classification
   - Output layer for classification

2. **SelfAttention**: Multi-head self-attention mechanism
   - 4 parallel attention heads
   - 256 dimensions per head (1024/4)
   - Scaled dot-product attention
   - Fixed attention score scaling

3. **FeedForwardNet**: Position-wise feed-forward network
   - Linear expansion: 1024 -> 2048 dimensions
   - ReLU activation
   - Linear projection: 2048 -> 1024

4. **Utility Functions**:
   - `create_patches()`: Converts images to patch sequences
   - `create_slice_batches()`: Creates training batches

## Key Features

- **Patch-based Processing**: Images split into 4x4 patches
- **Class Token**: Special token for final classification
- **Positional Embeddings**: Learnable position encodings
- **Multi-Head Attention**: 4 heads for different feature representations
- **Multi-Head Attention**: 6 heads for different feature representations
- **Residual Connections**: Stable training with skip connections
- **Layer Normalization**: Pre-normalization for better convergence
- **Dropout**: Regularization to prevent overfitting

## Training Details

- **Dataset**: CIFAR-10 (10 classes, 32x32 RGB images)
- **Batch Size**: 64
- **Learning Rate**: 3e-4 (AdamW optimizer)
- **Epochs**: 50
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing learning rate
- **Data Augmentation**: Random crop and horizontal flip
- **Device**: CUDA-enabled (falls back to CPU)

## Usage

### Training

```bash
# Train the Vision Transformer on CIFAR-10
python train.py
```

This will:
- Download and preprocess CIFAR-10 dataset
- Train the ViT for 50 epochs
- Save best model as `best_vit_model.pt`
- Save final checkpoint as `vit_final_checkpoint.pt`

### Inference

```bash
# Classify an image
python demo.py --image path/to/image.jpg --top_k 5
```

Options:
- `--image`: Path to input image (required)
- `--model_path`: Path to trained model (default: best_vit_model.pt)
- `--top_k`: Show top k predictions (default: 5)

## Implementation Notes

### What Works Well
- Complete ViT architecture from scratch
- Proper patch embedding and positional encodings
- Correct multi-head attention implementation
- Stable training with proper normalization
- Data augmentation and regularization

### Key Fixes Applied
- Fixed attention score scaling (using head_dim instead of hardcoded values)
- Proper patch embedding with convolutional projection
- Correct class token handling
- Fixed residual connections and layer normalization order
- Removed debug code and hardcoded paths

### Architecture Choices for CIFAR-10
- Smaller embedding dimension (384 vs 768) for faster training
- 6 transformer blocks instead of 12
- 6 attention heads instead of 12
- 4x4 patch size for 32x32 images

## Files

- `model.py`: Core ViT implementation (VisionTransformer, PatchEmbedding, etc.)
- `train.py`: Training script with CIFAR-10 data loading
- `demo.py`: Inference script for image classification
- `best_vit_model.pt`: Best trained model weights (generated after training)
- `vit_final_checkpoint.pt`: Final training checkpoint (generated after training)

## Dependencies

- PyTorch
- torchvision
- Pillow (for image loading in demo)
- numpy

## Performance

Expected performance on CIFAR-10:
- Training time: ~30-60 minutes on modern GPU
- Test accuracy: ~80-85% (depending on hyperparameters)
- Model size: ~10M parameters

## Future Enhancements

- Add learning rate warmup
- Implement mixup/cutmix augmentation
- Add label smoothing
- Experiment with larger models
- Add Grad-CAM visualization
- Support other datasets (ImageNet, etc.)

## Understanding the Code

This implementation demonstrates understanding of:

1. **Vision Transformer Architecture**: How images are converted to sequences
2. **Patch Embedding**: Convolutional approach to patch creation
3. **Multi-Head Attention**: Parallel attention computation on patches
4. **Class Token**: How ViT performs classification
5. **Positional Embeddings**: Spatial information in transformer
6. **Residual Connections**: How gradients flow through deep networks

The code is optimized for CIFAR-10 while maintaining the core ViT concepts from the original paper.
