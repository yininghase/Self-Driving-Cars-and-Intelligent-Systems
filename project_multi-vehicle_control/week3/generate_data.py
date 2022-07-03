from cProfile import label
from cmath import pi
from platform import platform
from matplotlib.pyplot import show
import numpy as np
from model import ModelPredictiveControl
from simulation import sim_run
import sys 
import torch
import os
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from tqdm import tqdm

from week3.visualization import Visualization

options = {}
options['FIG_SIZE'] = [8, 8]

show_image = False
save_image = False
collect_training_data = True

if sys.platform == "darwin":
    FOLDER = "./results/output"
else: 
    FOLDER = "/storage/remote/atcremers57/m_vehicle_c/data"

NUM_OF_VEHICLES = 2
TRAINING_DATA_PATH = FOLDER + "/training_data_2_constrained.pt"
LABEL_DATA_PATH = FOLDER + "/label_data_2_constrained.pt"

simulation_options = {
        "start": [0, 0, 0, 0, 0, 0, 0, 0],
        "target": [0, 0, 0, 0, 0, 0],
        "horizon": 20,
        "distance_cost": 1, #100, #10000,
        "angle_cost": 1, #100, #10000,
        "collision_cost": 1,
        "control_init": None,
        "obstacle_cost": 0,
        "non_round_obstacles": [],
        "name": "random"
    }

def generate_data():
    training_data = torch.tensor([])
    label_data = torch.tensor([])

    if not os.path.exists('./results/output'):
        os.makedirs('./results/output')

    ITERATIONS = 3
    success_full_simulation = 0
    for iter in tqdm(range(ITERATIONS)):
        start, target = get_start_position(2, constraints=True)
        simulation_options["start"] = start
        simulation_options["target"] = target

        # Trainingdata: [x, y, angle, v, x_d, y_d, angle_d]
        # Label data: The ordering is like this [pedal_0, steering_0, pedal_1, steering_1, ..., pedal_horizon-1, steering_horizon-1]
        state, u, training_tensor, label_tensor, success = \
        sim_run(options, simulation_options, ModelPredictiveControl, save=save_image, show=show_image,\
             collect_training_data=collect_training_data)        
        
        if success:
            training_data = torch.cat((training_data, training_tensor))
            label_data = torch.cat((label_data, label_tensor))
            success_full_simulation += 1
        else:
            print("No training data generated because the simulation failed")
        
        if iter%20==0:
            # save the data and the labels
            print("Saving data at step {}".format(iter))
            torch.save(training_data, TRAINING_DATA_PATH)
            torch.save(label_data, LABEL_DATA_PATH)
    
    print("{}/{} simulations successful".format(success_full_simulation, ITERATIONS))
    print("Training data shpape: ", training_data.shape)
    print("Label data shape: ", label_data.shape)
    
    print("Training data memory usage: ", sys.getsizeof(label_data.storage())/1000, "KB")
    print("Label data memory usage: ", sys.getsizeof(training_data.storage())/1000, "KB")
        
    torch.save(training_data, TRAINING_DATA_PATH)
    torch.save(label_data, LABEL_DATA_PATH)

def get_start_position(num_vehicles, constraints=False):
    POSITON_RANGE = 5
    ANGLE_EPS = 1/4 * 2 * np.pi
    if num_vehicles == 2:
        if constraints:
            pos_1 = np.random.uniform(low=[-POSITON_RANGE, -POSITON_RANGE], high=[POSITON_RANGE, POSITON_RANGE])
            distance= np.random.uniform(low=5, high=30)
            if pos_1[0] > 0 and pos_1[1] > 0:
                angle_1 = 1/4 * np.pi
            elif pos_1[0] > 0 and pos_1[1] < 0:
                angle_1 = 3/4 * np.pi
            elif pos_1[0] < 0 and pos_1[1] > 0:
                angle_1 = 5/4 * np.pi
            else:
                angle_1 = 7/4 * np.pi
            # take angle and distance and compute other position
            x_d = pos_1[0] + (distance * np.cos(angle_1))
            y_d = pos_1[1] + distance * np.sin(angle_1)
            pos_2 = np.array([x_d, y_d])
            # compute the other angle so that they are facing each other
            start = pos_1.tolist() + [angle_1+np.random.random()*ANGLE_EPS] + [0] + pos_2.tolist() + [angle_1+np.pi+np.random.random()*ANGLE_EPS] + [0]
            target = pos_2.tolist() + [angle_1+np.random.random()*ANGLE_EPS] + pos_1.tolist() + [angle_1+np.pi+np.random.random()*ANGLE_EPS]
        else:
            while True:
                ref_pos = np.random.uniform(low=[-POSITON_RANGE, -POSITON_RANGE], high=[POSITON_RANGE, POSITON_RANGE])
                start_pos = np.random.uniform(low=[-POSITON_RANGE, -POSITON_RANGE], high=[POSITON_RANGE, POSITON_RANGE])
                if np.linalg.norm(ref_pos - start_pos) > 5:
                    break
            ref_angle = np.random.uniform(low=[-np.pi], high=[np.pi])
            start_angle = np.random.uniform(low=[-np.pi], high=[np.pi])
            
            start = start_pos.tolist() + start_angle.tolist() + [0] + ref_pos.tolist() + ref_angle.tolist() + [0]
            target = ref_pos.tolist() + ref_angle.tolist() + start_pos.tolist() + start_angle.tolist()
    else:
        while True:
            ref_pos = np.random.uniform(low=[-POSITON_RANGE, -POSITON_RANGE], high=[POSITON_RANGE, POSITON_RANGE])
            start_pos = np.random.uniform(low=[-POSITON_RANGE, -POSITON_RANGE], high=[POSITON_RANGE, POSITON_RANGE])
            if np.linalg.norm(ref_pos - start_pos) > 5:
                break
        ref_angle = np.random.uniform(low=[-np.pi], high=[np.pi])
        start_angle = np.random.uniform(low=[-np.pi], high=[np.pi])
        start = start_pos.tolist() + start_angle.tolist() + [0]
        target = ref_pos.tolist() + ref_angle.tolist()

    return start, target

def load_data():
    training_data = torch.load(TRAINING_DATA_PATH)
    label_data = torch.load(LABEL_DATA_PATH)
    return training_data, label_data

if __name__ == '__main__':
    X, Y = load_data()
    print("test")
    # generate_data()
    # viusalize data point with label for sanity check\
    
    


    
    