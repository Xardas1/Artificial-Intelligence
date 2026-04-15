import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.query_weights = nn.Parameter(torch.rand(1024, 1024))
        self.key_weights = nn.Parameter(torch.rand(1024, 1024))
        self.value_weights = nn.Parameter(torch.rand(1024, 1024))

    def calculate_parameters(self):
        self.query = torch.matmul(self.input_vec, self.query_weights)
        self.key = torch.transpose(torch.matmul(self.input_vec, self.key_weights), 1, 2)
        self.value = torch.matmul(self.input_vec, self.value_weights)

    def multi_head_attention(self):
        batch_size = self.query.shape[0]
        seq_len = self.query.shape[1]
        self.query = torch.reshape(self.query, (batch_size, seq_len, 4, 256))
        self.query = torch.transpose(self.query, 1, 2)
        self.key = torch.reshape(self.key, (batch_size, seq_len, 4, 256))
        self.key = torch.transpose(self.key, 1, 2)
        self.key = torch.transpose(self.key, 2, 3)
        self.value = torch.reshape(self.value, (batch_size, seq_len, 4, 256))
        self.value = torch.transpose(self.value, 1, 2)

    def calculate_attention_score(self):
        self.attention_scores = torch.matmul(self.query, self.key).float() / math.sqrt(256)

    def normalize_softmax(self):
        self.normalized_values = torch.nn.functional.softmax(self.attention_scores, dim=-1)

    def create_representation(self):
        self.representations = torch.matmul(self.normalized_values, self.value)
        self.representations = torch.transpose(self.representations, 1, 2)
        batch_size = self.representations.shape[0]
        seq_len = self.representations.shape[1]
        self.representations = torch.reshape(self.representations, (batch_size, seq_len, 1024))

    def forward(self, x):
        self.input_vec = x
        self.calculate_parameters()
        self.multi_head_attention()
        self.calculate_attention_score()
        self.normalize_softmax()
        self.create_representation()
        return self.representations

class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        return output

class DawidGPT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.patch_embedding = nn.Linear(768, 1024)
        self.attention_blocks = nn.ModuleList([SelfAttention() for _ in range(4)])
        self.ffn_blocks = nn.ModuleList([FeedForwardNet() for _ in range(4)])
        self.output_layer = nn.Linear(1024, num_classes)
        self.norm1 = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(1024)
        
        frequency = self.calculate_frequencies()
        positional_encodings = self.calculate_positional_encodings(frequency)
        self.register_buffer('positional_encodings', positional_encodings)

    def calculate_positional_encodings(self, frequency):
        position_vector = torch.reshape(torch.arange(64), (64, 1)).float()
        frequency = torch.reshape(frequency, (1, 64))
        function_input = torch.matmul(position_vector, frequency)
        sin_values = torch.sin(function_input)
        cos_values = torch.cos(function_input)
        positional_encodings = torch.stack((sin_values, cos_values), dim=2).reshape(64, 1024)
        return positional_encodings

    def calculate_frequencies(self):
        pair_indices = torch.arange(64) * 2
        exponent = pair_indices / 1024
        denominator = 10000 ** exponent
        frequency = 1 / denominator
        return frequency

    def forward(self, x):
        x = self.patch_embedding(x)
        if x.shape[1] <= 64:
            x = x + self.positional_encodings[:x.shape[1], :]
        
        for i in range(4):
            residual = x
            x = self.norm1(x)
            attn_out = self.attention_blocks[i](x)
            x = residual + 0.1 * attn_out

            residual = x 
            x = self.norm2(x)
            ffn_out = self.ffn_blocks[i](x)
            x = residual + 0.1 * ffn_out

        x = torch.mean(x, dim=1)
        x = self.output_layer(x)

        return x

def create_patches(images, patch_size=4):
    batch_size, channels, height, width = images.shape
    
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 4, 1)
    patches = patches.contiguous().view(batch_size, -1, patch_size * patch_size * channels)
    
    return patches

def create_slice_batches(patches_list):
    batch_size = patches_list.shape[0]
    seq_len = patches_list.shape[1] - 1
    
    input_batches = patches_list[:, :seq_len, :]
    target_batches = patches_list[:, 1:, :]
    
    return input_batches, target_batches
