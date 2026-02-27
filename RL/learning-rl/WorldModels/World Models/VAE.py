import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)

        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=0)

        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=0)

        self.flatten = nn.Flatten(start_dim=1)

        self.fc_mu = nn.Linear(1024, 32)

        self.fc_logvar = nn.Linear(1024, 32)


    def forward(self, x):    
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
    
        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.flatten(x)
 
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.sqrt(torch.exp(logvar))
        epsilon = torch.randn_like(std)

        summation = (1 + logvar - mu**2 - torch.exp(logvar))
        kl = (-0.5 * torch.mean(torch.sum(summation))) 
        # for batch of images KL = -0.5 * mean(sum(1 + log(σ²) - μ² - σ²))
        z = mu + std * epsilon

        return z, kl
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.unflatten = nn.Linear(32, 1024)

        self.convtranspose1 = nn.ConvTranspose2d(1024, 128, 5, stride=1, padding=0)

        self.convtranspose2 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=0)

        self.convtranspose3 = nn.ConvTranspose2d(64, 32, 6, stride=2, padding=0)

        self.convtranspose4 = nn.ConvTranspose2d(32, 3, 6, stride=2, padding=0)

    def forward(self, x):
        
        x = self.unflatten(x)

        x = torch.reshape(x, (32, 1024, 1, 1))

        x = self.convtranspose1(x)
        x = F.relu(x)

        x = self.convtranspose2(x)
        x = F.relu(x)

        x = self.convtranspose3(x)
        x = F.relu(x)

        x = self.convtranspose4(x)
        x = torch.sigmoid(x)

        return x


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x, kl = self.encoder(x)

        reconstructed_image = self.decoder(x)
        return reconstructed_image, kl


encoder = Encoder()
decoder = Decoder()

model = VAE(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.MSELoss()

num_epochs = 300

def load_buffer(batch_indicies):
    batches = []
    for idx in batch_indicies:
        batch = np.load(f'D:/WorldModels/batches/batch_{idx+1}.npy')
        batches.append(batch)

    buffer = np.concatenate(batches, axis=0)
    return buffer


for epoch in range(num_epochs):
    # Randomly pick 50 batches
    batch_indicies = np.random.choice(625, size=10, replace=False)
    buffer = load_buffer(batch_indicies)

    buffer_tensor = torch.from_numpy(buffer).float()

    dataset = TensorDataset(buffer_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)




    for batch in dataloader:

        frame = batch[0].to(device)

        output, kl_divergence = model(frame)

        reconstructed_loss = criterion(output, frame)

        total_loss = reconstructed_loss + 0.001 * kl_divergence


        # Print actual values to see what's happening
        print(f"Input range: {frame.min()} to {frame.max()}")
        print(f"Output range: {output.min()} to {output.max()}")
        print(f"Reconstruction loss: {reconstructed_loss.item()}")
        print(f"KL divergence: {kl_divergence.item()}")
        print(f"Total loss: {total_loss.item()}")

        pdb.set_trace()

        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

        #if epoch % 10 == 0:
          #  print(f"Epoch {epoch}, Loss: {total_loss.item()}")



torch.save(model.state_dict(), 'vae_weights.pth')

