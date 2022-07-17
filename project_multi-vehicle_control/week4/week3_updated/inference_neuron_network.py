import numpy as np
from generate_data import get_start_position
import torch

from model import NeuronNetwork, ModelPredictiveControl
from simulation import sim_run
from tqdm import tqdm


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

    # best_model_path = './results/output/model/epoch_390_loss_0.0167.pth' # adapt the path accordingly before running 
    even_model_path = './week4/week3_updated/results/output/model/weights_even.pth'
    linear_model_path = './week4/week3_updated/results/output/model/weights_linear.pth'
    exponential_model_path = './week4/week3_updated/results/output/model/weights_exponential.pth'
    
    even_model = NeuronNetwork(input_size=14, output_size=20*2*2, hidden_sizes=[30, 200, 200]).to(device)
    even_model.load_state_dict(torch.load(even_model_path))

    linear_model = NeuronNetwork(input_size=14, output_size=20*2*2, hidden_sizes=[30, 200, 100]).to(device)
    linear_model.load_state_dict(torch.load(linear_model_path))

    exponential_model = NeuronNetwork(input_size=14, output_size=20*2*2, hidden_sizes=[30, 200, 100]).to(device)
    exponential_model.load_state_dict(torch.load(exponential_model_path))
    
    num_vehicles = 2
    horizonee = 20
    NUM_OF_VEHICLES = 2

    even_model.eval()
    linear_model.eval()
    exponential_model.eval()

    for i in tqdm(range(15)):
        start, target = get_start_position(2, constraints=True)
        simulation_options["start"] = start
        simulation_options["target"] = target

        simulation_options["name"] = "even_model_" + str(i)
        state, u, input_tensor, target_tensor, success = sim_run(options, simulation_options, ModelPredictiveControl, 
                                                        save=save_image, show=show_image, collect_training_data=collect_training_data, model=even_model, show_optimization=True)

        simulation_options["name"] = "linear_model_" + str(i)
        state, u, input_tensor, target_tensor, success = sim_run(options, simulation_options, ModelPredictiveControl, 
                                                        save=save_image, show=show_image, collect_training_data=collect_training_data, model=linear_model, show_optimization=True)
        
        simulation_options["name"] = "exponential_model_" + str(i)
        state, u, input_tensor, target_tensor, success = sim_run(options, simulation_options, ModelPredictiveControl, 
                                                        save=save_image, show=show_image, collect_training_data=collect_training_data, model=exponential_model, show_optimization=True)