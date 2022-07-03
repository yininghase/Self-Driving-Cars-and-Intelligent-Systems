
from pkgutil import get_data
import numpy as np
from generate_data import get_start_position
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import NeuronNetwork, ModelPredictiveControl
from simulation import sim_run
from tqdm import tqdm
from generate_data import load_data
from week3.visualization import Visualization


options = {}
options['FIG_SIZE'] = [8, 8]

show_image = False
save_image = True
collect_training_data = False

simulation_options = {
        "start": [0, 0, 0, 0, 0, 0, 0, 0],
        "target": [0, 0, 0, 0, 0, 0],
        "horizon": 20,
        "distance_cost": 1, #100, #10000,
        "angle_cost": 1, #100, #10000,
        "collision_cost": 0,
        "control_init": None,
        "obstacle_cost": 0,
        "non_round_obstacles": [],
    }

def control_to_action(mpc, x, y):
    predicted_model = np.array([x[:, :4].flatten()])
    for j in range(1, y.shape[0]):  
        predicted = mpc.plant_model(predicted_model[-1], mpc.dt, y[j, :, :].flatten())
        predicted_model = np.append(predicted_model, np.array([predicted]), axis=0)
    return predicted_model


if __name__ == "__main__":
    
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    print('Device available now:', device)

    best_model_path = './results/output/model/epoch_390_loss_0.0167.pth' # adapt the path accordingly before running 
    # best_model_path = './results/output/best_model.pth'
    model = NeuronNetwork(input_size=14, output_size=20*2*2, hidden_sizes=[30, 200, 100]).to(device)
    model.load_state_dict(torch.load(best_model_path))
    
    num_vehicles = 2
    horizonee = 20
    model.eval()
    NUM_OF_VEHICLES = 2

    # test on training data
    # for i in range(100,1000,100):
    #     horizon = simulation_options["horizon"]
    #     X, Y = load_data()
    #     x = X[i+15]
    #     y = Y[i+15]
    #     y_hat = model(x[None,:].type(torch.float))[0].detach()
        
    #     x = x.numpy().reshape(NUM_OF_VEHICLES, 7)
    #     y = y.numpy().reshape(horizon, NUM_OF_VEHICLES, 2)
    #     y_hat = y_hat.numpy().reshape(horizon, NUM_OF_VEHICLES, 2)

    #     mpc = ModelPredictiveControl(simulation_options)
    #     predicted_label = control_to_action(mpc, x, y)
    #     predicted_model = control_to_action(mpc, x, y_hat)

    #     visualization = Visualization(NUM_OF_VEHICLES, True, 100, simulation_options["horizon"], 
    #     x.reshape(NUM_OF_VEHICLES, -1)[:,:4], x.reshape(NUM_OF_VEHICLES, -1)[:,4:7], \
    #         name="new_{}".format(i), show=False, save=True)
        
    #     visualization.visualize_data_point(
    #         predicted_label.reshape(horizon, NUM_OF_VEHICLES, 4),
    #         predicted_model.reshape(horizon, NUM_OF_VEHICLES, 4)
    #         )

    for i in tqdm(range(10)):
        simulation_options["name"] = "model2_constrained_" + str(i)
        
        # X, Y = load_data()
        # x = X[0]
        # y = Y[0]
        # y_hat = model(x[None,:].type(torch.float))[0].detach()
        # x = x.numpy().reshape(NUM_OF_VEHICLES, 7)
        # y = y.numpy().reshape(horizon, NUM_OF_VEHICLES, 2)
        # y_hat = y_hat.numpy().reshape(horizon, NUM_OF_VEHICLES, 2)

        # mpc = ModelPredictiveControl(simulation_options)
        # predicted_label = control_to_action(mpc, x, y)
        # predicted_model = control_to_action(mpc, x, y_hat)
        # visualization = Visualization(NUM_OF_VEHICLES, True, 100, simulation_options["horizon"], 
        # x.reshape(NUM_OF_VEHICLES, -1)[:,:4], x.reshape(NUM_OF_VEHICLES, -1)[:,4:7], \
        #     name="new_{}".format(i), show=False, save=True)
        # visualization.visualize_data_point(
        #     predicted_label.reshape(horizon, NUM_OF_VEHICLES, 4),
        #     predicted_model.reshape(horizon, NUM_OF_VEHICLES, 4)
        # )
        
        # simulation_options["start"] = list(X[0].numpy().reshape(NUM_OF_VEHICLES, 7)[:, :4].flatten())
        # simulation_options["target"] = list(X[0].numpy().reshape(NUM_OF_VEHICLES, 7)[:, 4:7].flatten())
        start, target = get_start_position(2, constraints=True)
        simulation_options["start"] = start
        simulation_options["target"] = target



        state, u, input_tensor, target_tensor, success = sim_run(options, simulation_options, ModelPredictiveControl, 
                                                        save=save_image, show=show_image, collect_training_data=collect_training_data, model=model)
