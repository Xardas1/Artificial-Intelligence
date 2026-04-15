#!/usr/bin/env python3
"""
Demo script for the Transformers implementation
Run this script to quickly test text generation with a trained model
"""

import torch
import torch.nn.functional as F
import argparse

# Import the model classes
from model import DawidGPT

def generate_text(model, char_to_idx, idx_to_char, device, start_text, length=500, temperature=1.0):
    """Generate text using the trained model"""
    model.eval()
    
    # Convert start text to tokens
    tokens = [char_to_idx.get(ch, 0) for ch in start_text]
    
    for _ in range(length):
        # Take last 128 tokens (or all if less than 128)
        input_seq = tokens[-128:] if len(tokens) > 128 else tokens
        input_tensor = torch.tensor([input_seq]).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        # Get prediction for LAST position
        next_token_logits = output[0, -1, :] / temperature
        
        # Convert to probabilities and sample
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        
        # Add to sequence
        tokens.append(next_token)
    
    # Convert back to text
    generated_text = ''.join([idx_to_char.get(t, '<unk>') for t in tokens])
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Transformers Text Generation Demo')
    parser.add_argument('--prompt', type=str, default='The quick brown fox', 
                       help='Starting text for generation')
    parser.add_argument('--length', type=int, default=500, 
                       help='Length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0, 
                       help='Sampling temperature (lower = more deterministic)')
    parser.add_argument('--model_path', type=str, default='dawid_gpt_model.pt', 
                       help='Path to trained model')
    parser.add_argument('--vocab_path', type=str, default='vocab.pt', 
                       help='Path to vocabulary file')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load vocabulary
    try:
        vocab_data = torch.load(args.vocab_path)
        char_to_idx = vocab_data['char_to_idx']
        idx_to_char = vocab_data['idx_to_char']
        vocab_size = vocab_data['vocab_size']
        print(f"Loaded vocabulary with {vocab_size} characters")
    except FileNotFoundError:
        print("Vocabulary file not found! Please train the model first.")
        return
    
    # Load model
    try:
        model = DawidGPT(vocab_size)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file not found! Please train the model first.")
        return
    
    # Generate text
    print(f"\nGenerating text with prompt: '{args.prompt}'")
    print(f"Length: {args.length}, Temperature: {args.temperature}")
    print("-" * 50)
    
    generated = generate_text(
        model, char_to_idx, idx_to_char, device, 
        args.prompt, args.length, args.temperature
    )
    
    print(generated)
    print("-" * 50)

if __name__ == "__main__":
    main()
