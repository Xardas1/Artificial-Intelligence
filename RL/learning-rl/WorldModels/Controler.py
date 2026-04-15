import torch
import torch.nn as nn
import cma


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()

        self.linear = nn.Linear(288, 3)

    
    def forward(self, x):

        x = self.linear(x)

        return x
    


controller = Controller()

test = torch.rand(100, 32, 288)

actions = controller(test)

print(actions.shape )