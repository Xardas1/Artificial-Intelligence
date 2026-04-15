#!/usr/bin/env python3
"""
Training script for the Transformers implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import string

from model import DawidGPT

def create_sample_text():
    """Create sample training text"""
    return "The quick brown fox jumps over the lazy dog. " * 1000 + \
           "In a hole in the ground there lived a hobbit. " * 1000 + \
           "It was the best of times, it was the worst of times. " * 1000

def create_slice_batches(tokenized_text, batch_size=32, seq_len=128):
    """Create training batches"""
    max_start = max(0, len(tokenized_text) - seq_len - 1)
    start_pos = torch.randint(0, max_start, (batch_size,)).reshape(batch_size, 1)
    addition_value = torch.arange(0, seq_len + 1)
    indicies = start_pos + addition_value
    stacked_batch = tokenized_text[indicies]
    input_batches = stacked_batch[:, :seq_len]
    target_batches = stacked_batch[:, 1:]
    return input_batches, target_batches

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create training data
    print("Creating training data...")
    text = create_sample_text()
    
    # Create vocabulary
    vocabulary = sorted(set(text))
    vocab_size = len(vocabulary)
    char_to_idx = {vocabulary[i]: i for i in range(len(vocabulary))}
    idx_to_char = {i: vocabulary[i] for i in range(len(vocabulary))}
    tokenized_text = torch.tensor([char_to_idx[i] for i in text])
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training data length: {len(text)} characters")

    # Initialize model
    model = DawidGPT(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = 1000  # Quick training for demo
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        input_batch, target_batch = create_slice_batches(tokenized_text)
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        output = model(input_batch)
            
        output = output.view(-1, vocab_size)
        target_batch = target_batch.reshape(-1)
        
        loss = criterion(output, target_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    print("Training complete!")

    # Save model and vocabulary
    torch.save(model.state_dict(), 'dawid_gpt_model.pt')
    print("Model saved as 'dawid_gpt_model.pt'")

    torch.save({
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size
    }, 'vocab.pt')
    print("Vocabulary saved as 'vocab.pt'")

if __name__ == "__main__":
    main()
