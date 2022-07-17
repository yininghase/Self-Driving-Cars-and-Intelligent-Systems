import os
import random

from matplotlib.image import NonUniformImage
import numpy as np

import torch

from nn import NeuronNetwork
from model import ModelPredictiveControl
from simulation import sim_run
from tqdm import tqdm

from generate_data import generate_data, generate_data_with_collision


options = {}
options['FIG_SIZE'] = [8, 8]

show_image = True
save_image = True
collect_training_data = False

simulation_options = {
        "start": [0, 0, 0, 0],
        "target": [10, 10, 0],
        "horizon": 20,
        "distance_cost": 1, #100, #10000,
        "angle_cost": 1, #100, #10000,
        "collision_cost": 0,
        "control_init": None,
        "obstacle_cost": 0,
        "non_round_obstacles": [],
    }

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

def control_to_action(mpc, x, y):
    predicted_model = np.array([x[:, :4].flatten()])
    for j in range(1, y.shape[0]):  
        predicted = mpc.plant_model(predicted_model[-1], mpc.dt, y[j, :, :].flatten())
        predicted_model = np.append(predicted_model, np.array([predicted]), axis=0)
    return predicted_model


if __name__ == "__main__":
    
    set_seed()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device available now:', device)

    NUM_VEHICLES = 2
    HORIZON = 20

    best_model_path = './results/model/epoch_422_loss_0.1340.pth'
    # model = NeuronNetwork(input_size=7*NUM_VEHICLES, output_size=HORIZON*NUM_VEHICLES*2, hidden_sizes=[30, 200, 100]).to(device)
    model = NeuronNetwork(input_size=7*NUM_VEHICLES, output_size=HORIZON*NUM_VEHICLES*2).to(device)
    model.load_state_dict(torch.load(best_model_path))

    model.eval()
    
    # test_data = generate_data(5, NUM_VEHICLES, HORIZON)
    test_data = generate_data_with_collision(5, NUM_VEHICLES, HORIZON)
    
    starts = test_data[:,:,:4]
    targets = test_data[:,:,4:]
    

    for i in tqdm(range(len(test_data))):
        simulation_options["name"] = "model1_constrained_" + str(i)
        #start, target = get_start_position(1, constraints=False)
        start = starts[i]
        target = targets[i]
        simulation_options["start"] = start
        simulation_options["target"] = target

        state, u, input_tensor, target_tensor, success = sim_run(options, simulation_options, ModelPredictiveControl, 
                                                        save=save_image, show=show_image, collect_training_data=collect_training_data, model=model, show_optimization=False)
