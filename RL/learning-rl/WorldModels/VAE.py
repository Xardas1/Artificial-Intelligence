import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# World Models Vision Component
# This VAE compresses 64x64x3 observations into 32-dimensional latent space
# The latent space is where the dynamics model and controller operate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Vision Model Encoder - Compresses observations to latent space
    
    Input: 64x64x3 RGB image
    Output: 32-dimensional latent vector z with reparameterization
    
    Architecture: 4 conv layers with stride 2 (downsampling)
    Final feature map: 256 channels x 2x2 = 1024 features
    """
    def __init__(self):
        super(Encoder, self).__init__()

        # Convolutional layers progressively downsample and extract features
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=0)  # 64x64 -> 31x31
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)  # 31x31 -> 14x14
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=0)  # 14x14 -> 6x6
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=0)  # 6x6 -> 2x2

        self.flatten = nn.Flatten(start_dim=1)

        # Latent space parameters (32-dimensional)
        self.fc_mu = nn.Linear(1024, 32)        # Mean of latent distribution
        self.fc_logvar = nn.Linear(1024, 32)   # Log variance of latent distribution


    def forward(self, x):    
        # Encode image through convolutional layers
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
    
        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.flatten(x)
 
        # Latent distribution parameters
        mu = self.fc_mu(x)           # Mean
        logvar = self.fc_logvar(x)   # Log variance
        
        # Reparameterization trick: z = μ + σ * ε
        std = torch.sqrt(torch.exp(logvar))
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon

        # KL divergence: D_KL(N(μ,σ) || N(0,1))
        # Encourages latent space to follow standard normal distribution
        summation = (1 + logvar - mu**2 - torch.exp(logvar))
        kl = (-0.5 * torch.mean(torch.sum(summation))) 

        return z, kl
    

class Decoder(nn.Module):
    """
    Vision Model Decoder - Reconstructs images from latent space
    
    Input: 32-dimensional latent vector z
    Output: 64x64x3 reconstructed image
    
    Architecture: 4 transpose conv layers (upsampling)
    """
    def __init__(self):
        super(Decoder, self).__init__()

        # Project latent vector back to feature map
        self.unflatten = nn.Linear(32, 1024)

        # Transpose conv layers progressively upsample
        self.convtranspose1 = nn.ConvTranspose2d(1024, 128, 5, stride=1, padding=0)  # 2x2 -> 6x6
        self.convtranspose2 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=0)      # 6x6 -> 14x14
        self.convtranspose3 = nn.ConvTranspose2d(64, 32, 6, stride=2, padding=0)      # 14x14 -> 31x31
        self.convtranspose4 = nn.ConvTranspose2d(32, 3, 6, stride=2, padding=0)       # 31x31 -> 64x64

    def forward(self, x):
        # Decode latent vector back to image
        x = self.unflatten(x)

        batch_size = x.size(0)
        x = x.view(batch_size, 1024, 1, 1)  # Reshape to 2x2 feature map

        # Upsample through transpose conv layers
        x = self.convtranspose1(x)
        x = F.relu(x)

        x = self.convtranspose2(x)
        x = F.relu(x)

        x = self.convtranspose3(x)
        x = F.relu(x)

        x = self.convtranspose4(x)
        x = torch.sigmoid(x)  # Normalize to [0,1] for image pixels

        return x


class VAE(nn.Module):
    """
    Complete Variational Autoencoder for World Models
    
    Combines encoder and decoder to learn compressed representations
    of observations that can be used by dynamics model and controller
    """
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # Encode to latent space
        z, kl_divergence = self.encoder(x)
        
        # Decode back to image space
        reconstructed_image = self.decoder(z)
        return reconstructed_image, kl_divergence


encoder = Encoder()
decoder = Decoder()

model = VAE(encoder, decoder).to(device)

# Training configuration
optimizer = optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.MSELoss()  # Reconstruction loss

num_epochs = 20  # Training epochs

def load_buffer(batch_indicies):
    """
    Load training batches from disk
    
    Args:
        batch_indicies: List of batch numbers to load
    
    Returns:
        Combined buffer of training frames
    """
    batches = []
    for idx in batch_indicies:
        batch = np.load(f'data/batches/batch_{idx+1}.npy')
        batches.append(batch)

    buffer = np.concatenate(batches, axis=0)
    return buffer


# Training loop
for epoch in range(num_epochs):
    # Randomly sample batches for training
    batch_indicies = np.random.choice(625, size=10, replace=False)
    buffer = load_buffer(batch_indicies)

    buffer_tensor = torch.from_numpy(buffer).float()

    dataset = TensorDataset(buffer_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    for batch in dataloader:

        frame = batch[0].to(device)

        output, kl_divergence = model(frame)

        # VAE Loss = Reconstruction Loss + β * KL Divergence
        # β controls the trade-off between reconstruction quality and latent space regularization
        reconstructed_loss = criterion(output, frame)
        total_loss = reconstructed_loss + 0.001 * kl_divergence


        # Training progress monitoring
        print(f"Input range: {frame.min():.3f} to {frame.max():.3f}")
        print(f"Output range: {output.min():.3f} to {output.max():.3f}")
        print(f"Reconstruction loss: {reconstructed_loss.item():.6f}")
        print(f"KL divergence: {kl_divergence.item():.6f}")
        print(f"Total loss: {total_loss.item():.6f}")


        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

        #if epoch % 10 == 0:
          #  print(f"Epoch {epoch}, Loss: {total_loss.item()}")



# Save trained VAE weights for use by other components
torch.save(model.state_dict(), 'vae_weights.pth')
print("VAE training completed and weights saved!")

