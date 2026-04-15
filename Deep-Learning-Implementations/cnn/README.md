# NumPy CNN Implementation from Scratch

This is a complete Convolutional Neural Network implementation from scratch using only NumPy. No deep learning frameworks are used - all operations are implemented manually to demonstrate understanding of CNN fundamentals.

## Architecture Overview

The CNN consists of:

### Core Functions

1. **create_kernels**: Creates random convolution kernels
   - Generates 3x3xdepth kernels
   - Used for both conv layers

2. **forward_pass**: Complete forward pass through network
   - Creates 32 and 64 filter kernels
   - Applies conv, ReLU, max pooling sequentially
   - Fully connected layers with bias
   - Final softmax classification

3. **convolve_one_step_vectorized**: Vectorized convolution
   - Uses stride tricks for efficiency
   - Matrix multiplication approach

4. **max_pooling**: Custom max pooling implementation
   - 2x2 pooling with stride 2
   - Handles edge cases

5. **ReLUv1/v2**: ReLU activation functions
   - Element-wise max(0, x)
   - Vectorized version

6. **fully_connected_vectorized**: Fully connected layer
   - Matrix multiplication + bias
   - Efficient vectorized implementation

7. **More_Robust_Softmax**: Numerically stable softmax
   - Temperature parameter for control
   - Prevents overflow

## Architecture Details

- **Input**: RGB images (H x W x 3)
- **Conv1**: 32 filters, 3x3 kernels
- **Pool1**: 2x2 max pooling
- **Conv2**: 64 filters, 3x3 kernels  
- **Pool2**: 2x2 max pooling
- **FC1**: 120 neurons
- **FC2**: 10 neurons (for classification)
- **Output**: Softmax probabilities

## Usage

### Demo (Forward Pass Only)

```bash
# Run demo with sample image
python demo.py

# Run demo with custom image
python demo.py --image path/to/image.jpg --size 32 --classes 10
```

Options:
- `--image`: Path to input image (optional, creates sample if not provided)
- `--size`: Image size for processing (default: 32)
- `--classes`: Number of output classes (default: 10)

### Example Output

```
NumPy CNN Demo
==============================
Initializing CNN with 10 output classes...
Model initialized successfully!
Creating sample gradient image...
Image shape: (32, 32, 3)

Running forward pass through CNN...
Prediction completed! Output shape: (10, 1)

==================================================
CNN PREDICTION RESULTS
==================================================
Top 5 predictions:
1. Class_3: 0.100123 (10.01%)
2. Class_7: 0.100089 (10.01%)
3. Class_1: 0.100076 (10.01%)
4. Class_9: 0.100045 (10.00%)
5. Class_5: 0.099987 (9.99%)
==================================================
```

## Implementation Notes

### What Works Well
- Complete CNN architecture from scratch
- Vectorized convolution operations
- Proper layer-by-layer forward pass
- Numerically stable softmax
- Modular, extensible design

### Key Technical Details

1. **Vectorized Convolution**: Uses `as_strided` for efficient patch extraction
2. **Stride Tricks**: Memory-efficient convolution implementation
3. **Layer Separation**: Each operation as independent class
4. **Shape Management**: Automatic dimension calculation throughout network

### Current Limitations
- Forward pass only (no backpropagation implemented)
- Random weight initialization (no training)
- Fixed architecture (not easily configurable)
- No batch processing (single image at a time)

### What Could Be Added
- Backpropagation implementation
- Training loop with gradient descent
- Batch processing support
- More layer types (BatchNorm, Dropout, etc.)
- Configurable architecture

## Files

- `model.py`: Core CNN implementation (all layer classes and NumpyCNN)
- `demo.py`: Demo script for forward pass inference
- `README.md`: This documentation

## Dependencies

- NumPy
- Pillow (for image loading)
- Standard library modules

## Understanding the Code

This implementation demonstrates understanding of:

1. **Convolution Operations**: How filters slide over images
2. **Stride Tricks**: Efficient memory access patterns
3. **Pooling Operations**: Spatial dimensionality reduction
4. **Activation Functions**: Non-linear transformations
5. **Forward Pass**: Data flow through neural networks
6. **Softmax**: Probability computation for classification

## Educational Value

This code is perfect for learning:
- How CNNs work under the hood
- Memory-efficient convolution implementation
- NumPy advanced features (stride_tricks)
- Neural network architecture design
- Forward pass mechanics

The implementation prioritizes clarity and educational value over performance optimizations, making it ideal for understanding CNN fundamentals.

## Future Enhancements

- Implement backpropagation for training
- Add gradient descent optimization
- Support for different architectures
- Batch processing capabilities
- Performance optimizations
- More advanced layers
