from statistics import mean
import torch
from os.path import isfile, join
from os import listdir
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

def load_data(path, num_vehicles):
    input_files = sorted([f for f in listdir(path) if isfile(join(path, f))and f.startswith('training')])
    label_files = sorted([f for f in listdir(path) if isfile(join(path, f))and f.startswith('label')])
    input = torch.ones(1, num_vehicles*7)
    label = torch.ones(1, num_vehicles*20*2)
    for file in input_files:
        input = torch.cat([input, torch.load(join(path, file))])
    for file in label_files:
        label = torch.cat([label, torch.load(join(path, file))])
    # two vehicles with each 7 values
    return input.reshape(-1, num_vehicles, 7), label.reshape(-1, 20, num_vehicles, 2)

def compute_mean_std(vector, visualize=True):
    var, mean = torch.var_mean(vector)
    sigma = torch.sqrt(var) 
    x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
    if visualize:        
        plt.plot(x, stats.norm.pdf(x, mean, sigma))
        plt.show()
    return mean, sigma

def data_nearer_than_distance(data, label, distance):
    smaller = torch.linalg.norm(data[:,:,:2] - data[:,:,4:6], axis=2, ord=2) < (distance + 0.1)
    if label is not None:
        label = label.swapaxes(1,2).reshape(-1, 20*2)[smaller.flatten()]
    data = data.reshape(-1, 7)[smaller.flatten()]
    return data, label

def data_at_distance(data, distance):
    smaller = torch.linalg.norm(data[:,:,:2] - data[:,:,4:6], axis=2, ord=2) < (distance + 0.1)
    larger = torch.linalg.norm(data[:,:,:2] - data[:,:,4:6], axis=2, ord=2) > (distance - 0.1)
    both = torch.logical_and(smaller, larger)
    return data.reshape(-1, 7)[both.flatten()]

def get_analysis(path, distance):
    input, _ = load_data(path, 2)
    data = data_at_distance(input, distance)
    mean_velocity, sigma_velocity = compute_mean_std(data[:,3], visualize=False)
    angle_offset = data[:,2] - data[:,6]
    mean_angle, sigma_angle = compute_mean_std(angle_offset, visualize=False)
    return mean_velocity, sigma_velocity, mean_angle, sigma_angle

if __name__ == "__main__":
    input_all, label_all = load_data("./week5/task1/data/", 2)
    # near_data = data_nearer_than_distance(input, 5)
    compute_mean_std(input_all[:,:,3])    
    print("test")
