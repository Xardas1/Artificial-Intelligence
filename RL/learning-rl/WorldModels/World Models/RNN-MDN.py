import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          )
        
        self.mean = nn.Linear(hidden_size, 160)

        self.variance = nn.Linear(hidden_size, 160)

        self.weight = nn.Linear(hidden_size, 5)

        self.softmax = nn.Softmax(dim=2)
    

    def forward(self, x):

        output, h_n = self.rnn(x)

        #h_n = h_n.squeeze()

        mean = self.mean(output)

        mean = torch.reshape(mean, (100, 32, 32, 5))

        variance = self.variance(output)

        variance = torch.reshape(variance, (100, 32, 32, 5))

        weight = self.weight(output)

        probabilities = self.softmax(weight)

        probabilities_2d = probabilities.view(-1, 5)

        k = torch.multinomial(probabilities_2d, num_samples=1)

        k = k.view(100, 32, 1)

        k_expanded = k.unsqueeze(2).expand(100, 32, 32, 1)

        selected_mean = torch.gather(mean, dim=-1, index=k_expanded).squeeze()

        selected_variance = torch.gather(variance, dim=-1, index=k_expanded).squeeze()

        selected_variance = torch.exp(selected_variance)

        epsilon = torch.randn(32)

        std = torch.sqrt(selected_variance)

        z = selected_mean + std * epsilon

        h_n = h_n.expand(100, 32, 256)

        return z, h_n


model = RNNModel(input_size=35, hidden_size=256, output_size=32, num_layers=1)

optimizer = optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.MSELoss()

example_data = torch.rand((100, 32, 35))
z_t1 = torch.rand((100, 32, 32))


epochs = 10000
for epoch in range(epochs):

    z, h_n = model(example_data)
    
    pdb.set_trace()

    loss = criterion(z_t1, z)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")