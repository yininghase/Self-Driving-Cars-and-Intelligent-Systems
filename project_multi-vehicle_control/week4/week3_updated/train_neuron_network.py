import os
import gc
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from loss import WeightedMeanSquaredError
from model import NeuronNetwork
from generate_data import load_data


out_folder = './week4/week3_updated/results/output/model'

epochs = 400
patience = 100
batch_size = 128
learning_rate = 0.1
weight_decay = 1e-6
lr_patience = 10
lr_factor = 0.2


def set_seed(seed = 1357):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    return

class VehicleControlDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, index):        
        return self.X[index], self.y[index]


def train():
    set_seed()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device available now:', device)
    
    '''Train the neuron network'''
    # load neuron network

    horizon =  20
    num_vehicles = 2

    model = NeuronNetwork(input_size=14, output_size=2*20*2, hidden_sizes=[30, 200, 100]).to(device)
    X, y = load_data() 

    # print information about the model and the data
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Trainable params: ', params)
    print('Data length: ', len(X))
    
    # split out the training data and validation data
    train_index, valid_index = train_test_split(np.arange(len(X)),test_size=0.2)
    
    try:
        os.makedirs(out_folder)
    except:
        pass
    f = open(os.path.join(out_folder, "logs.txt"), "w+")

    # prepare data loader for image loading
    train_X = X[train_index]
    train_y = y[train_index]
    valid_X = X[valid_index]
    valid_y = y[valid_index]

    train_dataset = VehicleControlDataset(train_X, train_y)
    valid_dataset = VehicleControlDataset(valid_X, valid_y)
    train_data_num = len(train_dataset)
    val_data_num = len(valid_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # set optimizer, scheduler and criterion
    optimizer = Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=lr_patience, verbose=True, factor=lr_factor)
    
    # model_name = "weights_even"
    # model_name ="weights_linear"
    model_name = "weights_exponential"

    criterion = WeightedMeanSquaredError(batch_size, horizon, num_vehicles)
    # weights = np.ones(horizon)
    # weights = np.flip(np.array(range(horizon)))
    weights = np.flip(np.exp(np.array(range(horizon))))
    normalized_weights = torch.tensor(weights / np.sum(weights))



    best_loss = np.inf

    LOGS = {
        "training_loss": [],
        "validation_loss": [],
    }
    
    for epoch in range(epochs):
        # === TRAIN ===
        # Sets the model in training mode
        train_loss = 0
        valid_loss = 0
        model.train()

        for k, (inputs, targets) in enumerate(train_loader):

            optimizer.zero_grad()
            
            inputs = inputs.type(torch.float32).to(device)
            targets = targets.type(torch.float32).to(device)
            
            out = model(inputs)
            
            loss = criterion(out, targets, normalized_weights)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*len(inputs)
            
        train_loss /= train_data_num
        LOGS["training_loss"].append(train_loss)

        # === EVAL ===
        # Sets the model in evaluation mode
        model.eval()

        with torch.no_grad():
            for k, (inputs, targets) in enumerate(valid_loader):
                
                inputs = inputs.type(torch.float32).to(device)
                targets = targets.type(torch.float32).to(device)

                out = model(inputs)
                loss = criterion(out, targets, normalized_weights)
                valid_loss += loss.item()*len(inputs)
            
            valid_loss /= val_data_num
            LOGS["validation_loss"].append(valid_loss)
                
        scheduler.step(valid_loss)

        msg = f'Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.6} | Valid Loss: {valid_loss:.6}'
        with open(os.path.join(out_folder, "logs.txt"), 'a+') as f:
            print(msg,file=f)
            print(msg)
    
        if valid_loss < best_loss:
            model_path = os.path.join(out_folder, model_name + '.pth')
            best_loss = valid_loss
            # Reset patience (because we have improvement)
            patience_f = patience
            torch.save(model.state_dict(), model_path)               
            print('Model Saved!')
                
        else:
            # Decrease patience (no improvement in ROC)
            patience_f -= 1
            if patience_f == 0:
                print(f'Early stopping (no improvement since {patience} models) | Best Valid Loss: {valid_loss:.6f}')
                break
    
    del train_dataset, valid_dataset, train_loader, valid_loader, inputs, targets
    gc.collect()
    # plot training and validation loss
    plt.plot(LOGS["training_loss"], label='Training Loss')
    plt.plot(LOGS["validation_loss"], label='Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(out_folder, model_name + "_learning_plot.png"))
    plt.show()


if __name__ == '__main__':
    
    train()


