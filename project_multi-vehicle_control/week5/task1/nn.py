import torch.nn as nn
import torch

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
            else:
                modules.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                modules.append(nn.BatchNorm1d(hidden_sizes[i]))
                modules.append(nn.ReLU())
        self.net = nn.Sequential(*modules)
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.net(x)
        return x        


class LargeNeuronNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        
        self.linear1 = nn.Linear(input_size, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.af1 = nn.ReLU()
        
        self.linear2 = nn.Linear(100, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.af2 = nn.ReLU()
        
        self.linear3 = nn.Linear(200, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.af3 = nn.ReLU()
        
        self.linear4 = nn.Linear(100, 200)
        self.bn4 = nn.BatchNorm1d(200)
        self.af4 = nn.ReLU()
        
        self.linear5 = nn.Linear(200, 100)
        self.bn5 = nn.BatchNorm1d(100)
        self.af5 = nn.ReLU()
        
        self.linear6 = nn.Linear(100, 200)
        self.bn6 = nn.BatchNorm1d(200)
        self.af6 = nn.ReLU()
        
        self.linear7 = nn.Linear(200, 100)
        self.bn7 = nn.BatchNorm1d(100)
        self.af7 = nn.ReLU()
        
        self.linear8 = nn.Linear(100, output_size)
        self.bn8 = nn.BatchNorm1d(output_size)
        self.tanh = nn.Tanh()
        
        self.linear1.apply(self.init_weights_relu)
        self.linear2.apply(self.init_weights_relu)
        self.linear3.apply(self.init_weights_relu)
        self.linear4.apply(self.init_weights_relu)
        self.linear5.apply(self.init_weights_relu)
        self.linear6.apply(self.init_weights_relu)
        self.linear7.apply(self.init_weights_relu)
        self.linear8.apply(self.init_weights_tanh)
        
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        assert output_size % 2 == 0
        self.bound = torch.tensor([1, 0.8]).tile((int(output_size/2),)).to(device)

    def init_weights_relu(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)
    
    def init_weights_tanh(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=5/3)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.af1(x)
        
        x1 = self.linear2(x)
        x1 = self.bn2(x1)
        x1 = self.af2(x1)
        
        x1 = self.linear3(x1)
        x1 = self.bn3(x1)
        x1 = self.af3(x1)
        
        x1 = x1+x
        
        x2 = self.linear4(x1)
        x2 = self.bn4(x2)
        x2 = self.af4(x2)
        
        x2 = self.linear5(x2)
        x2 = self.bn5(x2)
        x2 = self.af5(x2)
        
        x2 = x2+x1 
        
        x3 = self.linear6(x2)
        x3 = self.bn6(x3)
        x3 = self.af6(x3) 
        
        x3 = self.linear7(x3)
        x3 = self.bn7(x3)
        x3 = self.af7(x3)
        
        x3 = x3+x2
        
        x4 = self.linear8(x3)
        x4 = self.bn8(x4)
        x4 = self.tanh(x4)
        
        x4 = x4*self.bound
        return x4     