import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query_weights = nn.Parameter(torch.rand(d_model, d_model))
        self.key_weights = nn.Parameter(torch.rand(d_model, d_model))
        self.value_weights = nn.Parameter(torch.rand(d_model, d_model))
    
    def calculate_parameters(self):
        self.query = torch.matmul(self.input_vec, self.query_weights)
        self.key = torch.transpose(torch.matmul(self.input_vec, self.key_weights), 1, 2)
        self.value = torch.matmul(self.input_vec, self.value_weights)
    
    def multi_head_attention(self):
        batch_size = self.query.shape[0]
        seq_len = self.query.shape[1]
        self.query = torch.reshape(self.query, (batch_size, seq_len, self.n_heads, self.head_dim))
        self.query = torch.transpose(self.query, 1, 2)
        self.key = torch.reshape(self.key, (batch_size, seq_len, self.n_heads, self.head_dim))
        self.key = torch.transpose(self.key, 1, 2)
        self.key = torch.transpose(self.key, 2, 3)
        self.value = torch.reshape(self.value, (batch_size, seq_len, self.n_heads, self.head_dim))
        self.value = torch.transpose(self.value, 1, 2)
                
    def calculate_attention_score(self):
        self.attention_scores = torch.matmul(self.query, self.key).float() / math.sqrt(self.head_dim)
            
    def normalize_softmax(self):
        self.normalized_values = torch.nn.functional.softmax(self.attention_scores, dim=-1)
    
    def create_representation(self):
        self.representations = torch.matmul(self.normalized_values, self.value)
        self.representations = torch.transpose(self.representations, 1, 2)
        batch_size = self.representations.shape[0]
        seq_len = self.representations.shape[1]
        self.representations = torch.reshape(self.representations, (batch_size, seq_len, self.d_model))
            
    def forward(self, x):
        self.input_vec = x
        self.calculate_parameters()
        self.multi_head_attention()
        self.calculate_attention_score()
        self.normalize_softmax()
        self.create_representation()
        return self.representations

class FeedForwardNet(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        return output

class DawidGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=4, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.attention_blocks = nn.ModuleList([SelfAttention(d_model, n_heads) for _ in range(n_layers)])
        self.ffn_blocks = nn.ModuleList([FeedForwardNet(d_model) for _ in range(n_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        frequency = self.calculate_frequencies()
        positional_encodings = self.calculate_positional_encodings(frequency)
        self.register_buffer('positional_encodings', positional_encodings)
    
    def calculate_positional_encodings(self, frequency):
        position_vector = torch.reshape(torch.arange(self.max_seq_len), (self.max_seq_len, 1)).float()
        frequency = torch.reshape(frequency, (1, self.d_model // 2))
        function_input = torch.matmul(position_vector, frequency)
        sin_values = torch.sin(function_input)
        cos_values = torch.cos(function_input)
        positional_encodings = torch.stack((sin_values, cos_values), dim=2).reshape(self.max_seq_len, self.d_model)
        return positional_encodings
    
    def calculate_frequencies(self):
        pair_indices = torch.arange(self.d_model // 2) * 2
        exponent = pair_indices / self.d_model
        denominator = 10000 ** exponent
        frequency = 1 / denominator
        return frequency
    
    def forward(self, x):
        x = self.embeddings(x)
        x = x + self.positional_encodings[:x.shape[1], :]
        
        for i in range(len(self.attention_blocks)):
            residual = x
            x = self.norm1(x)
            attn_out = self.attention_blocks[i](x)
            x = residual + 0.1 * attn_out

            residual = x 
            x = self.norm2(x)
            ffn_out = self.ffn_blocks[i](x)
            x = residual + 0.1 * ffn_out
        
        x = self.output_layer(x)
        return x
