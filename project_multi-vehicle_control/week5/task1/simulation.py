from cProfile import label
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import time
import os
from tqdm import tqdm
import torch
from visualization import Visualization

def sim_run(options, simulation_options, MPC, save=False, name=None, show=True, collect_training_data=False, \
    model=None, show_optimization=True, parking_model=None, take_optimization_step=None):
    mpc = MPC(simulation_options)
    num_inputs = 2*mpc.num_vehicle
    is_model = (model != None)
    if mpc.control_init is None:
        u = np.zeros(mpc.horizon*num_inputs)
    else:
        u = (mpc.control_init.flatten())[:mpc.horizon*num_inputs]
        
    bounds = []

    # Set bounds for inputs bounded optimization.
    for i in range(mpc.horizon):
        bounds += [[-1, 1]]*mpc.num_vehicle
        bounds += [[-0.8, 0.8]]*mpc.num_vehicle

    ref = mpc.target

    state_i = np.array([mpc.start])

    sim_total = 50
    predict_info_opt = [state_i]
    predict_info_model = [state_i]
    
    label_data_model = np.array([[0] * mpc.horizon * mpc.num_vehicle * 2])
    label_data_opt = np.array([[0] * mpc.horizon * mpc.num_vehicle * 2])

    for i in range(1, sim_total+1):
        ### Optimization Prediction ###
        if show_optimization:
            u = np.delete(u, np.arange(2*mpc.num_vehicle))
            u = np.append(u, u[-2*mpc.num_vehicle:])        
            u_solution = minimize(mpc.cost_function, u, (state_i[-1], ref),
                                method='SLSQP',
                                bounds=bounds,
                                tol=1e-5)
            u_opt = u_solution.x
            y_opt = mpc.plant_model(state_i[-1], mpc.dt, u_opt[:2*mpc.num_vehicle])
            label_data_opt = np.append(label_data_opt, np.array([u_opt]), axis=0)
            predict_info_opt += [get_predictions(mpc, state_i, u_opt)]

        ### Model Prediction ###
        if model is not None:
            model_input = torch.tensor(np.append(state_i[-1].reshape(mpc.num_vehicle, 4), np.array(ref).reshape(mpc.num_vehicle, 3), axis=1).flatten()).type(torch.float32)[None,:]
            if parking_model is not None:
                parking_distance = 5
                current_state = model_input.reshape(1, mpc.num_vehicle, 7).squeeze()
                cars_to_be_parked = torch.linalg.norm(current_state[:, :2] - current_state[:, 4:6], axis=1, ord=2) < parking_distance
                cars_to_be_parked = cars_to_be_parked.tolist()
                park_model_inputs = current_state[cars_to_be_parked]
                park_predictions = []
                for input in park_model_inputs:
                    park_predictions.append(parking_model(input[None, :]))
                if park_predictions:
                    park_predictions = torch.cat(park_predictions, dim=0)
                else:
                    park_predictions = torch.tensor([])
            u_model = model(model_input)
            if parking_model is not None:
                if park_predictions.shape[0] > 0:
                    u_model_final = u_model.reshape(mpc.num_vehicle, mpc.horizon, 2)
                    u_model_final[cars_to_be_parked] = park_predictions.reshape(-1, mpc.horizon, 2)
                    u_model_final = u_model_final.reshape(1, -1)
                    print(cars_to_be_parked)
                else:
                    u_model_final = u_model
            else:
                u_model_final = u_model
            u_model = u_model_final[0].detach().numpy()
            y_model = mpc.plant_model(state_i[-1], mpc.dt, u_model[:2*mpc.num_vehicle])
            label_data_model = np.append(label_data_model, np.array([u_model]), axis=0)
            predict_info_model += [get_predictions(mpc, state_i, u_model)]
        
        ### Simulation ###
        if take_optimization_step == "optimization":
            state_i = np.append(state_i, np.array([y_opt]), axis=0)
        elif take_optimization_step == "model":
            state_i = np.append(state_i, np.array([y_model]), axis=0)
        else:
            raise Exception("Invalid parameter for take_optimization_step")
        
        # check if the model has not moved the last n steps and if this is not the case end the simulation
        n = 10 # if the vehicle has not mvoed more then eps for n steps we end the simulation
        eps = 0.01
        current_state = state_i.reshape((i+1, mpc.num_vehicle, 4))
        if i+1 > n:
            if np.all(np.sum(np.linalg.norm(current_state[-n:,:,:2][:-1] - current_state[-n:,:,:2][1:], axis=2), axis=0) < eps):
                break
        
    visualization = Visualization(mpc.num_vehicle, (model != None), sim_total, mpc.horizon, state_i[0], ref, simulation_options["name"], show, save, show_optimization)
    visualization.create_video(state_i, predict_info_opt, predict_info_model)
    
    state_i = state_i.reshape((-1, mpc.num_vehicle, 4))

    # we just want to plot the range of points for which the vehicle still moves or will still move
    max_j = 0
    for i in range(mpc.num_vehicle):
        change_in_pos = np.sum(np.abs(state_i[1:,i,:2] - state_i[:-1,i,:2]), axis=1)
        for j in range(sim_total):
            if np.sum(change_in_pos[j:]) < 0.1:
                break   
        max_j = max(max_j, j)

    if save or show:
        visualization.plot_trajectory(state_i[:max_j, :, :2])

    iteration_until_vehicle_moves = max_j + 15

    training_tensor = None
    label_tensor = None 
    simulation_sucessfull = False
    ###################
    # COLLECTING TRAINING DATA
    if collect_training_data:

        # For each data point we have a vehicle position (x,y), angle and velocity and the desired position and angle
        # That makes 7 values per time step times the range sim_total (num of vehicle will already be a dim to makes things easier later)
        # The ordering is like this [x, y, angle, v, x_d, y_d, angle_d]
        if mpc.num_vehicle == 1:
            first_vehicle = np.append(state_i[:, 0, :], np.tile(ref[:3], (state_i.shape[0], 1)), axis=1)
            training_tensor = torch.tensor(first_vehicle)
        else:
            first_vehicle = np.append(state_i[:, 0, :], np.tile(ref[:3], (state_i.shape[0], 1)), axis=1)
            second_vehicle = np.append(state_i[:,1, :], np.tile(ref[3:], (state_i.shape[0], 1)), axis=1)
            training_tensor = torch.tensor(np.append(first_vehicle, second_vehicle, axis=1))
        
        # Each label of each datapoints has to contain the predicted steps of the optimization, pedal and steering input at each 
        # step of the horizon
        # The ordering is like this [pedal, steering]
        label_tensor_model = torch.tensor(label_data_model.reshape(-1, mpc.horizon * 2 * mpc.num_vehicle))
        label_tensor_opt = torch.tensor(label_data_opt.reshape(-1, mpc.horizon * 2 * mpc.num_vehicle))

        # The first label should be ignored and the second label belongs to the first state
        # for the last state there is no prediction
        training_tensor = training_tensor[:-1]
        # the first label should be ignored
        label_tensor_model = label_tensor_model[1:] 
        label_tensor_opt = label_tensor_opt[1:]
    else:
        training_tensor = None
        label_tensor = None
    
    return state_i, training_tensor, label_tensor_opt, label_tensor_model

def get_predictions(mpc, state_i, u):
    y_opt = mpc.plant_model(state_i[-1], mpc.dt, u[:2*mpc.num_vehicle])
    predicted_state = np.array([y_opt])
    for j in range(1, mpc.horizon):
        predicted = mpc.plant_model(
            predicted_state[-1], mpc.dt, u[2*mpc.num_vehicle*j:2*mpc.num_vehicle*(j+1)])
        predicted_state = np.append(
            predicted_state, np.array([predicted]), axis=0)
    return predicted_state