from statistics import variance
import numpy as np
from os import listdir
from os.path import isfile, join
import torch
import matplotlib.pyplot as plt
import scipy.stats as stats


def generate_data(data_length, num_vehicles, position_range=5):
    low_range = [-position_range, -position_range, -np.pi, -10]
    high_range = [position_range, position_range, np.pi, 10]
    starts = np.random.uniform(low=low_range, high=high_range, size=(data_length, num_vehicles, 4))
    targets = np.random.uniform(low=low_range[:3], high=high_range[:3], size=(data_length, num_vehicles, 3))
    
    pos_diff = np.linalg.norm(starts[:,:,:2]-targets[:,:,:2], axis=-1)
    factor = pos_diff/10
    factor[factor>=1] = 1
    
    # angle_diff = starts[:,:,2]-targets[:,:,2]
    # angle_diff *= factor
    # targets[:,:,2] = starts[:,:,2] - angle_diff
    
    starts[:,:,3] *= factor
    
    return np.append(starts, targets, axis=2)


def generate_data_with_collision(data_length, num_vehicles, position_range=5):
    low_range = [-position_range, -position_range, -np.pi, -10]
    high_range = [position_range, position_range, np.pi, 10]
    
    mean = [0, 0, 0]
    std_var = [3, 3, np.pi]
    
    starts = np.random.uniform(low=low_range, high=high_range, size=(data_length, num_vehicles, 4))
    shift = np.random.normal(loc=mean, scale=std_var, size=(data_length, num_vehicles, 3))
    
    targets = starts[:,:,:3] + shift
    
    org_arr = np.arange(num_vehicles)
    new_arr = np.random.permutation(org_arr)
    
    targets[:,[0,1],:] = targets[:,[1,0],:]
    
    pos_diff = np.linalg.norm(starts[:,:,:2]-targets[:,:,:2], axis=-1)
    factor = pos_diff/10
    factor[factor>=1] = 1
    
    # angle_diff = starts[:,:,2]-targets[:,:,2]
    # angle_diff *= factor
    # targets[:,:,2] = starts[:,:,2] - angle_diff
    
    starts[:,:,3] *= factor
    
    return np.append(starts, targets, axis=2)

def load_data(path):
    input_files = sorted([f for f in listdir(path) if isfile(join(path, f))and f.startswith('training')])
    label_files = sorted([f for f in listdir(path) if isfile(join(path, f))and f.startswith('label')])
    input = torch.ones(1, 14)
    label = torch.ones(1, 2*20*2)
    for file in input_files:
        input = torch.cat([input, torch.load(join(path, file))])
    for file in label_files:
        label = torch.cat([label, torch.load(join(path, file))])
    # two vehicles with each 7 values
    return input.reshape(-1, 2, 7), label.reshape(-1, 2, 20*2)

def compute_mean_std(data, visualize=True):
    mean, var = torch.var_mean(data[:,:,3])
    sigma = torch.sqrt(var) 
    x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
    if visualize:        
        plt.plot(x, stats.norm.pdf(x, mean, sigma))
        plt.show()
    return mean, sigma

def load_data(path):
    input_files = sorted([f for f in listdir(path) if isfile(join(path, f))and f.startswith('training')])
    label_files = sorted([f for f in listdir(path) if isfile(join(path, f))and f.startswith('label')])
    input = torch.ones(1, 14)
    label = torch.ones(1, 2*20*2)
    for file in input_files:
        input = torch.cat([input, torch.load(join(path, file))])
    for file in label_files:
        label = torch.cat([label, torch.load(join(path, file))])
    # two vehicles with each 7 values
    return input.reshape(-1, 2, 7), label.reshape(-1, 2, 20*2)


def split_data(input, label, split_distance=2):
    positions = input[:,:,:2]
    targets = input[:,:,4:6]
    indices = torch.linalg.norm(positions - targets, axis=2) > split_distance
    return input[indices], label[indices], input[~indices], label[~indices]


if __name__ == '__main__':
    path = "./week5/data/"
    input, label = load_data(path)
    far_input, far_label, close_input, close_label = split_data(input, label)
    mean, sigma = compute_mean_std(data, visualize=True)