import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb

# World Models Dynamics Component
# This RNN-MDN learns transition dynamics in latent space
# Predicts next latent state z_{t+1} given current z_t and action a_t
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNModel(nn.Module):
    """
    World Models Dynamics Model - RNN with Mixture Density Network
    
    Predicts next latent state z_{t+1} given current state z_t and action a_t
    Uses mixture of Gaussians to capture uncertainty in predictions
    
    Input: [z_t, a_t] where z_t is 32-dim latent, a_t is 3-dim action (total 35)
    Output: Predicted z_{t+1} with uncertainty modeling
    
    Architecture:
    - RNN: 256 hidden units, processes sequences
    - MDN: 5 mixture components, each with mean and variance for 32 dimensions
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN processes sequences of (z_t, a_t) pairs
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          )
        
        # Mixture Density Network outputs
        # Each dimension has 5 mixture components
        self.mean = nn.Linear(hidden_size, 160)      # 32 dims * 5 components
        self.variance = nn.Linear(hidden_size, 160)  # 32 dims * 5 components
        self.weight = nn.Linear(hidden_size, 5)      # 5 mixture weights
        self.softmax = nn.Softmax(dim=2)
    

    def forward(self, x):
        # Process sequence through RNN
        output, h_n = self.rnn(x)

        batch, seq, hidden = output.shape

        # Generate mixture density parameters
        mean = self.mean(output)          # Shape: [batch, seq, 160]
        mean = mean.view(batch, seq, 32, 5)  # Reshape: [batch, seq, dims, components]

        variance = self.variance(output)    # Shape: [batch, seq, 160]
        variance = torch.reshape(variance, (100, 32, 32, 5))  # Hardcoded for demo

        weight = self.weight(output)        # Shape: [batch, seq, 5]
        probabilities = self.softmax(weight)  # Mixture component probabilities

        # Sample mixture components for each dimension
        probabilities_2d = probabilities.view(-1, 5)
        k = torch.multinomial(probabilities_2d, num_samples=1)
        k = k.view(100, 32, 1)

        # Select mean and variance for chosen components
        k_expanded = k.unsqueeze(2).expand(100, 32, 32, 1)
        selected_mean = torch.gather(mean, dim=-1, index=k_expanded).squeeze()
        selected_variance = torch.gather(variance, dim=-1, index=k_expanded).squeeze()
        
        # Convert log variance to variance and sample
        selected_variance = torch.exp(selected_variance)
        epsilon = torch.randn_like(selected_mean)
        std = torch.sqrt(selected_variance)
        
        # Sample from selected Gaussian: z = μ + σ * ε
        z = selected_mean + std * epsilon

        # Expand hidden state for return (hardcoded for demo)
        h_n = h_n.expand(100, 32, 256)

        return z, h_n


# Initialize dynamics model
# Input: 35-dim (32-dim latent + 3-dim action)
# Hidden: 256-dim RNN state
# Output: 32-dim next latent state
model = RNNModel(input_size=35, hidden_size=256, output_size=32, num_layers=1)

optimizer = optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.MSELoss()  # Predict next latent state

# Example training data (in real implementation, use actual trajectories)
example_data = torch.rand((100, 32, 35))  # Sequence of (z_t, a_t) pairs
z_t1 = torch.rand((100, 32, 32))        # Target next latent states

epochs = 50
# Training loop
for epoch in range(epochs):
    # Predict next latent states
    z_pred, h_n = model(example_data)
    
    # Compute prediction loss
    loss = criterion(z_t1, z_pred)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

print("RNN-MDN training completed!")