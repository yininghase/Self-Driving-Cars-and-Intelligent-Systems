import numpy as np
from generate_data import get_start_position, get_start_position_close_to_target
from data_analysis import get_analysis
import torch
from nn import NeuronNetwork
from model import ModelPredictiveControl
from simulation import sim_run
from tqdm import tqdm
import sys

if sys.platform == "darwin":
    BASE = "./week5/task1/data/"
else: 
    BASE = "/storage/remote/atcremers57/m_vehicle_c/data/week5/"
FOLDER = BASE + "parking_data/reloop/"

TRAINING_DATA_PATH = FOLDER + "/test.pt"
LABEL_DATA_PATH = FOLDER + "/test.pt"

options = {}
options['FIG_SIZE'] = [8, 8]

show_image = True
save_image = False
collect_training_data = False

simulation_options_two_vehicles = {
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

simulation_options_one_vehicles = {
        "start": [0, 0, 0, 0],
        "target": [0, 0, 0],
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
    num_vehicles = 2
    horizon = 20

    parking_model_path = './week5/task1/models/parking4/test.pth'
    # main_model_path = './week5/task1/models/parking4/test.pth'
    main_model_path= './week5/task1/models/model1/test.pth'
    # best_model_path = './models/model1/test.pth' # adapt the path accordingly before running 
    
    main_model = NeuronNetwork(input_size=7*num_vehicles, output_size=horizon*2*num_vehicles, hidden_sizes=[30, 200, 200]).to(device)
    main_model.load_state_dict(torch.load(main_model_path))

    parking_model = NeuronNetwork(input_size=7, output_size=horizon*2, hidden_sizes=[30, 200, 200]).to(device)
    parking_model.load_state_dict(torch.load(parking_model_path))

    main_model.eval()
    parking_model.eval()

    training_data = torch.tensor([])
    label_data = torch.tensor([])

    model_string = "model"
    optimization_string = "optimization"

    for i in tqdm(range(1)):
        start, target = get_start_position(2, constraints=True)
        distance_to_target = 5
        # mean_velosscity, sigma_velocity, mean_angle, sigma_angle = get_analysis("./week5/task1/data/", distance_to_target)

        # start, target = get_start_position_close_to_target(mean_velocity, sigma_velocity, mean_angle, sigma_angle, distance_to_target)
        start, target = get_start_position(2, constraints=True)
        # start, target = get_start_position_close_to_target(mean_velocity, sigma_velocity, mean_angle, sigma_angle, distance_to_target)
        simulation_options_two_vehicles["start"] = start
        simulation_options_two_vehicles["target"] = target

        simulation_options_two_vehicles["name"] = "test_" + str(i)
        state, u, training_tensor, label_tensor, success = sim_run(options, simulation_options_two_vehicles, ModelPredictiveControl, 
                                                        save=save_image, show=show_image, collect_training_data=collect_training_data, \
                                                            model=main_model, show_optimization=False, parking_model=parking_model, \
                                                                take_optimization_step="model")
                                                
        if collect_training_data:
            training_data = torch.cat((training_data, training_tensor))
            label_data = torch.cat((label_data, label_tensor))
        else:
            print("No training data")
        
        if ((i%20)==0) and collect_training_data:
            # save the data and the labels
            print("Saving data at step {}".format(iter))
            print("Datapoints collected: {}".format(training_data.shape[0]))
            torch.save(training_data, TRAINING_DATA_PATH)
            torch.save(label_data, LABEL_DATA_PATH)
    if collect_training_data:
        print("Training data shpape: ", training_data.shape)
        print("Label data shape: ", label_data.shape)
        
        print("Training data memory usage: ", sys.getsizeof(label_data.storage())/1000, "KB")
        print("Label data memory usage: ", sys.getsizeof(training_data.storage())/1000, "KB")
            
        torch.save(training_data, TRAINING_DATA_PATH)
        torch.save(label_data, LABEL_DATA_PATH)
