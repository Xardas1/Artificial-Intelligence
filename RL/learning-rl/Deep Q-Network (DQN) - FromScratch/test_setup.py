#!/usr/bin/env python3
"""
Simple test script to verify the DQN setup works without running infinite training
"""

import numpy as np
import gymnasium as gym
import torch
from dqn_agent import DQN, image_preprocess, create_input_image, stack_images

def test_imports():
    """Test all imports work correctly"""
    print("Testing imports...")
    try:
        import dqn_agent
        print("  dqn_agent imported successfully")
        return True
    except ImportError as e:
        print(f"  Import error: {e}")
        return False

def test_model_creation():
    """Test DQN model can be created"""
    print("Testing model creation...")
    try:
        model = DQN(in_channels=4, num_actions=6)
        print(f"  Model created: {model}")
        return True
    except Exception as e:
        print(f"  Model creation error: {e}")
        return False

def test_environment():
    """Test environment setup"""
    print("Testing environment...")
    try:
        env = gym.make("ALE/Pong-v5", render_mode=None)
        print(f"  Environment created: {env}")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        env.close()
        return True
    except Exception as e:
        print(f"  Environment error: {e}")
        return False

def test_image_processing():
    """Test image preprocessing functions"""
    print("Testing image processing...")
    try:
        # Create dummy observation (210, 160, 3)
        dummy_obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        
        # Test preprocessing
        processed = image_preprocess(dummy_obs)
        print(f"  Processed image shape: {processed.shape}")
        
        # Test stacking
        list_to_stack = [processed] * 4
        stacked = stack_images(list_to_stack)
        print(f"  Stacked images shape: {stacked.shape}")
        
        return True
    except Exception as e:
        print(f"  Image processing error: {e}")
        return False

def test_forward_pass():
    """Test model forward pass"""
    print("Testing forward pass...")
    try:
        model = DQN(in_channels=4, num_actions=6)
        
        # Create dummy input (batch_size=1, channels=4, height=84, width=84)
        dummy_input = torch.randn(1, 4, 84, 84)
        
        # Forward pass
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
    except Exception as e:
        print(f"  Forward pass error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== DQN Setup Test ===\n")
    
    tests = [
        test_imports,
        test_model_creation,
        test_environment,
        test_image_processing,
        test_forward_pass
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The DQN setup is working correctly.")
        print("You can now run 'python train.py' to start training.")
    else:
        print("Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
