from os import listdir
from os.path import isfile, join
import sys 
import torch
import numpy as np
from tqdm import tqdm
import os
from simulation import sim_run
from model import ModelPredictiveControl
from data_analysis import get_analysis
from data_analysis import data_at_distance
from data_analysis import data_nearer_than_distance
from nn import NeuronNetwork

if sys.platform == "darwin":
    BASE = "./week5/task1/data/"
else: 
    BASE = "/storage/remote/atcremers57/m_vehicle_c/data/week5/"
FOLDER = BASE + "relearning/"

TRAINING_DATA_PATH = FOLDER + "/data55.pt"
OPT_LABEL_DATA_PATH = FOLDER + "/label_opt55.pt"
MODEL_LABEL_DATA_PATH = FOLDER + "/label_model55.pt"


simulation_options_two_vehicle = {
        "start": [0, 0, 0, 0, 0, 0, 0, 0],
        "target": [0, 0, 0, 0, 0, 0],
        "horizon": 20,
        "distance_cost": 1, #100, #10000,
        "angle_cost": 1, #100, #10000,
        "collision_cost": 1,
        # "smoothness_cost": 1,
        "control_init": None,
        "obstacle_cost": 0,
        "non_round_obstacles": [],
        "name": "random"
    }

simulation_options_one_vehicle = {
        "start": [0, 0, 0, 0],
        "target": [0, 0, 0],
        "horizon": 20,
        "distance_cost": 1, #100, #10000,
        "angle_cost": 1, #100, #10000,
        "collision_cost": 1,
        # "smoothness_cost": 1,
        "control_init": None,
        "obstacle_cost": 0,
        "non_round_obstacles": [],
        "name": "random"
    }


options = {}
options['FIG_SIZE'] = [8, 8]

save_image = False
show_image = False
collect_training_data = True

def get_start_position_reverse_into_target():
    POSITION_RANGE = 5
    target = np.random.uniform(low=[-POSITION_RANGE, -POSITION_RANGE], high=[POSITION_RANGE, POSITION_RANGE])
    target_angle = np.random.uniform(low=-np.pi, high=np.pi)
    
    velocity = np.random.uniform(low=-1, high=0)
    start_angle = np.random.uniform(low=target_angle-np.pi*1/16, high=target_angle+np.pi*1/16)

    eps = np.random.uniform(low=0, high=np.pi*1/16)
    distance = np.random.uniform(low=0, high=2)
    start_x = target[0] + (distance * np.cos(start_angle+1/2*np.pi+eps))
    start_y = target[1] + (distance * np.sin(start_angle+1/2*np.pi+eps))

    start = [start_x, start_y] + [start_angle] + [velocity]
    target = target.tolist() + [target_angle]
    return start, target


def get_start_position_missed_target():
    POSITION_RANGE = 5
    target = np.random.uniform(low=[-POSITION_RANGE, -POSITION_RANGE], high=[POSITION_RANGE, POSITION_RANGE])
    target_angle = np.random.uniform(low=-np.pi, high=np.pi)
    
    velocity = np.random.uniform(low=0, high=3)
    start_angle = np.random.uniform(low=target_angle-np.pi*1/16, high=target_angle+np.pi*1/16)

    eps = np.random.uniform(low=0, high=np.pi*1/8)
    distance = np.random.uniform(low=0, high=3)
    start_x = target[0] - (distance * np.cos(start_angle+1/2*np.pi+eps))
    start_y = target[1] - (distance * np.sin(start_angle+1/2*np.pi+eps))

    start = [start_x, start_y] + [start_angle] + [velocity]
    target = target.tolist() + [target_angle]
    return start, target

def get_start_position_close_to_target(m_vel, s_vel, m_ang, s_ang, distance_to_target):
    POSITION_RANGE = 5
    target = np.random.uniform(low=[-POSITION_RANGE, -POSITION_RANGE], high=[POSITION_RANGE, POSITION_RANGE])
    target_angle = np.random.uniform(low=-np.pi, high=np.pi)

    velocity = torch.normal(m_vel, s_vel**2)
    angle_diff = torch.normal(m_ang, s_ang**2)
    distance = np.random.uniform(low=distance_to_target-0.1, high=distance_to_target+0.1)
    distance_angle = np.random.uniform(low=target_angle-1/8*np.pi, high=target_angle+1/8*np.pi)

    
    start_x = target[0] - (distance * np.cos(distance_angle))
    start_y = target[1] - (distance * np.sin(distance_angle))
    start_angle = target_angle + angle_diff

    start = [start_x, start_y] + [start_angle] + [velocity]
    target = target.tolist() + [target_angle]
    return start, target

def get_start_position(num_vehicles, constraints=False):
    POSITON_RANGE = 5
    ANGLE_EPS = 1/4 * 2 * np.pi
    if num_vehicles == 2:
        if constraints:
            pos_1 = np.random.uniform(low=[-POSITON_RANGE, -POSITON_RANGE], high=[POSITON_RANGE, POSITON_RANGE])
            distance_1 = np.random.uniform(low=10, high=30)
            distance_2 = np.random.uniform(low=10, high=30)
            distance_3 = np.random.uniform(low=10, high=30)

            if pos_1[0] > 0 and pos_1[1] > 0:
                angle_1 = 1/4 * np.pi
            elif pos_1[0] > 0 and pos_1[1] < 0:
                angle_1 = 3/4 * np.pi
            elif pos_1[0] < 0 and pos_1[1] > 0:
                angle_1 = 5/4 * np.pi
            else:
                angle_1 = 7/4 * np.pi
            
            # take angle and distance and compute other position
            x_d = pos_1[0] + (distance_1 * np.cos(angle_1))
            y_d = pos_1[1] + (distance_1 * np.sin(angle_1))
            target_1 = np.array([x_d, y_d])

            x_d = pos_1[0] + (distance_2 * np.cos(angle_1))
            y_d = pos_1[1] + (distance_2 * np.sin(angle_1))
            pos_2 = np.array([x_d, y_d])
            x_d = pos_2[0] - (distance_3 * np.cos(angle_1))
            y_d = pos_2[1] - (distance_3 * np.sin(angle_1))
            target_2 = np.array([x_d, y_d])

            # compute the other angle so that they are facing each other
            start = pos_1.tolist() + [angle_1+np.random.random()*ANGLE_EPS] + [0] + pos_2.tolist() + [angle_1+np.pi+np.random.random()*ANGLE_EPS] + [0]
            target = target_1.tolist() + [angle_1+np.random.random()*ANGLE_EPS] + target_2.tolist() + [angle_1+np.pi+np.random.random()*ANGLE_EPS]
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

def generate_data():
    training_data = torch.tensor([])
    opt_label_data = torch.tensor([])
    model_label_data = torch.tensor([])

    if not os.path.exists('./results/output'):
        os.makedirs('./results/output')

    ITERATIONS = 1000
    success_full_simulation = 0

    num_vehicles = 2
    horizon = 20
    main_model_path= './week5/task1/test.pth'
    device = "cpu"
    main_model = NeuronNetwork(input_size=7*num_vehicles, output_size=horizon*2*num_vehicles, hidden_sizes=[30, 200, 200]).to(device)
    main_model.load_state_dict(torch.load(main_model_path))
    main_model.eval()

    distance_to_target = 3
    # mean_velocity, sigma_velocity, mean_angle, sigma_angle = get_analysis("./week5/task1/data/", distance_to_target)

    for iter in tqdm(range(ITERATIONS)):
        start, target = get_start_position(2, constraints=True)
        # simulation_options_two_vehicle["start"] = start
        # simulation_options_two_vehicle["target"] = target
        
        # start, target = get_start_position_close_to_target(mean_velocity, sigma_velocity, mean_angle, sigma_angle, distance_to_target)
        # start, target = get_start_position_missed_target()
        simulation_options_one_vehicle["start"] = start
        simulation_options_one_vehicle["target"] = target
        simulation_options_one_vehicle["name"] = "iter_{}".format(iter)

        # Trainingdata: [x, y, angle, v, x_d, y_d, angle_d]
        # Label data: The ordering is like this [pedal_0, steering_0, pedal_1, steering_1, ..., pedal_horizon-1, steering_horizon-1]
        state, training_tensor, opt_label_tensor, model_label_tensor = \
        sim_run(options, simulation_options_one_vehicle, ModelPredictiveControl, save=save_image, show=show_image,\
             collect_training_data=collect_training_data, model=main_model, take_optimization_step="model")        
        
        # if success:
        if True and collect_training_data:
            training_data = torch.cat((training_data, training_tensor))
            opt_label_data = torch.cat((opt_label_data, opt_label_tensor))
            model_label_data = torch.cat((model_label_data, model_label_tensor))
        else:
            print("No training data generated")
        
        if iter%20==0 and collect_training_data:
            # save the data and the labels
            print("Saving data at step {}".format(iter))
            print("Datapoints collected: {}".format(training_data.shape[0]))
            torch.save(training_data, TRAINING_DATA_PATH)
            torch.save(opt_label_data, OPT_LABEL_DATA_PATH)
            torch.save(model_label_data, MODEL_LABEL_DATA_PATH)
    
    if collect_training_data:
        torch.save(training_data, TRAINING_DATA_PATH)
        torch.save(opt_label_data, OPT_LABEL_DATA_PATH)
        torch.save(model_label_data, MODEL_LABEL_DATA_PATH)



def load_data(path, num_vehicles):
    input_files = sorted([f for f in listdir(path) if isfile(join(path, f))and ("training" in f or "data" in f)])
    label_files = sorted([f for f in listdir(path) if isfile(join(path, f))and "label" in f])
    input = torch.ones(1, num_vehicles*7)
    label = torch.ones(1, num_vehicles*20*2)
    for file in input_files:
        input = torch.cat([input, torch.load(join(path, file))])
    for file in label_files:
        label = torch.cat([label, torch.load(join(path, file))])
    # two vehicles with each 7 values
    return input.reshape(-1, num_vehicles*7), label.reshape(-1, 20*num_vehicles*2)

def load_data_b(path, num_vehicles):
    input_files = sorted([f for f in listdir(path) if isfile(join(path, f))and ("training" in f or "data" in f)])
    opt_label_files = sorted([f for f in listdir(path) if isfile(join(path, f))and "opt" in f])
    model_label_files = sorted([f for f in listdir(path) if isfile(join(path, f))and "model" in f])
    input = torch.ones(1, num_vehicles*7)
    opt_label = torch.ones(1, num_vehicles*20*2)
    model_label = torch.ones(1, num_vehicles*20*2)
    for file in input_files:
        input = torch.cat([input, torch.load(join(path, file))])
    for file in opt_label_files:
        opt_label = torch.cat([opt_label, torch.load(join(path, file))])
    for file in model_label_files:
        model_label = torch.cat([model_label, torch.load(join(path, file))])
    return input.reshape(-1, num_vehicles*7), opt_label.reshape(-1, 20*num_vehicles*2), model_label.reshape(-1, 20*num_vehicles*2)

def get_combined_data_a():
    i1_all, l1_all = load_data("./week5/task1/data/", 2)
    i1, l1 = data_nearer_than_distance(i1_all, l1_all, 5)
    i2, l2 = load_data("./week5/task1/data/parking_data/", 1)
    i2 = i2.reshape(-1, 7)
    l2 = l2.swapaxes(1,2).reshape(-1, 20*2)
    i3, l3 = load_data("./week5/task1/data/parking_data/reloop/", 1)
    i3 = i3.reshape(-1, 7)
    l3 = l3.swapaxes(1,2).reshape(-1, 20*2)
    return torch.cat([i1[:100*1000], i2, i3]), torch.cat([l1[:100*1000], l2, l3])

def get_combined_data_b():
    i1, l1 = load_data("./week5/task1/data", 2)
    i2, l2_opt, _ = load_data_b("./week5/task1/data/relearning/", 2)
    return torch.cat([i1, i2]), torch.cat([l1, l2_opt])

if __name__ == "__main__":
    # print("test")
    # start, target = get_start_position_close_to_target()
    generate_data()
    # get_combined_data()



    
    