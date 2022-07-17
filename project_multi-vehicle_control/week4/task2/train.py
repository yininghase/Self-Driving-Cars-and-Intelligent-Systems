import os
import random

from nn import NeuronNetwork
from generate_data import generate_data
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from loss import Loss

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

class Dataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]

if __name__ == "__main__":
    
    set_seed()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    out_folder = './results/model'
    
    try:
        os.makedirs(out_folder)
    except:
        pass
    
    # Hyperparameters 1
    NUM_VEHICLES = 1
    HORIZON = 20
    DATA_LENGTH = 100*1000
    BATCH_SIZE = 128
    EPOCHS = 400

    network_shape = [30, 200, 100]

    model = NeuronNetwork(input_size=7*NUM_VEHICLES, output_size=2*HORIZON*NUM_VEHICLES, hidden_sizes=network_shape).to(device)
    X = generate_data(DATA_LENGTH, NUM_VEHICLES, HORIZON)

    # split out the training data and validation data
    train_index, valid_index = train_test_split(np.arange(len(X)),test_size=0.2)

    train_X = X[train_index]
    valid_X = X[valid_index]

    train_dataset = Dataset(train_X)
    valid_dataset = Dataset(valid_X)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_data_num = len(train_dataset)
    val_data_num = len(valid_dataset)

    # Hyperparameters 2
    patience = 100
    learning_rate = 0.05
    weight_decay = 1e-6
    lr_patience = 10
    lr_factor = 0.2

    optimizer = Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=lr_patience, verbose=True, factor=lr_factor)


    criterion = Loss(horizon=HORIZON, num_vehicle=NUM_VEHICLES)

    LOGS = {
        "training_loss": [],
        "validation_loss": [],
    }
    
    best_loss = np.inf
    f = open(os.path.join(out_folder, "logs.txt"), "w+")

    for epoch in range(EPOCHS):
        train_loss = 0
        valid_loss = 0
        model.train()

        for k, (X) in enumerate(train_loader):
            current_batch_size = X.shape[0]
            optimizer.zero_grad()
            X = X.type(torch.float32).to(device)
            y_hat = model(X.reshape(current_batch_size, -1)).reshape(current_batch_size, HORIZON, NUM_VEHICLES, 2)
            loss = criterion(X[:,:,:4], X[:,:,4:], y_hat)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= train_data_num
        LOGS["training_loss"].append(train_loss)

        model.eval()
        with torch.no_grad():
            for k, (X) in enumerate(valid_loader):
                current_batch_size = X.shape[0]
                optimizer.zero_grad()
                X = X.type(torch.float32).to(device)
                y_hat = model(X.reshape(current_batch_size, -1)).reshape(current_batch_size, HORIZON, NUM_VEHICLES, 2)
                loss = criterion(X[:,:,:4], X[:,:,4:], y_hat)
                valid_loss += loss.item()    
            valid_loss /= val_data_num
            LOGS["validation_loss"].append(valid_loss)
        
        scheduler.step(train_loss)
        
        msg = f'Epoch: {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6} | Valid Loss: {valid_loss:.6}'
        with open(os.path.join(out_folder, "logs.txt"), 'a+') as f:
            print(msg,file=f)
            print(msg)
    
        if valid_loss < best_loss:
            model_path = os.path.join(out_folder, f'epoch_{epoch+1}_loss_{valid_loss:.4f}.pth')
            best_loss = valid_loss
            best_model_path = model_path 
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
    
    





