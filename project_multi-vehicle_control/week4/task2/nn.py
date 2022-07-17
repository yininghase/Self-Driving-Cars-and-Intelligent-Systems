import torch
import torch.nn as nn

class NeuronNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        modules = []
        layer = len(hidden_sizes)
        for i in range(layer):
            if i == 0:
                modules.append(nn.Linear(input_size, hidden_sizes[i]))
                modules.append(nn.BatchNorm1d(hidden_sizes[i]))
                modules.append(nn.ReLU())
            elif i == layer-1:
                modules.append(nn.Linear(hidden_sizes[i-1], output_size))
                modules.append(nn.Tanh())
            else:
                modules.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                modules.append(nn.BatchNorm1d(hidden_sizes[i]))
                modules.append(nn.ReLU())
        
        self.net = nn.Sequential(*modules)
        self.net.apply(self.init_weights)
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        assert output_size % 2 == 0
        self.bound = torch.tensor([1, 0.8]).tile((int(output_size/2),)).to(device)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.net(x)
        x = x*self.bound
        return x     