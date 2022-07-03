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

def sim_run(options, simulation_options, MPC, save=False, name=None, show=True, collect_training_data=False, model=None):
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
    u_i = np.array([[0, 0]*mpc.num_vehicle])
    sim_total = 100
    predict_info_opt = [state_i]
    predict_info_model = [state_i]
    # the first element can be ignored later 
    label_data = np.array([[0] * mpc.horizon * mpc.num_vehicle * 2])

    for i in range(1, sim_total+1):
        u = np.delete(u, np.arange(2*mpc.num_vehicle))
        u = np.append(u, u[-2*mpc.num_vehicle:])        
                
        start_time = time.time()

        # Non-linear optimization.
        u_solution = minimize(mpc.cost_function, u, (state_i[-1], ref),
                            method='SLSQP',
                            bounds=bounds,
                            tol=1e-5)

        u_opt = u_solution.x
        y_opt = mpc.plant_model(state_i[-1], mpc.dt, u_opt[:2*mpc.num_vehicle])
        predicted_state = np.array([y_opt])
        for j in range(1, mpc.horizon):
            predicted = mpc.plant_model(
                predicted_state[-1], mpc.dt, u_opt[2*mpc.num_vehicle*j:2*mpc.num_vehicle*(j+1)])
            predicted_state = np.append(
                predicted_state, np.array([predicted]), axis=0)
        predict_info_opt += [predicted_state]
        
        # update step
        if not is_model:
            state_i = np.append(state_i, np.array([y_opt]), axis=0)
            u_i = np.append(u_i, np.array([u_opt[:2*mpc.num_vehicle]]), axis=1)
            label_data = np.append(label_data, np.array([u_opt]), axis=0)
        
        # Model prediction
        else:
            model_input = torch.tensor(np.append(state_i[-1].reshape(mpc.num_vehicle, 4), np.array(ref).reshape(mpc.num_vehicle, 3), axis=1).flatten()).type(torch.float32)[None,:]
            u_model = model(model_input)
            u_model = u_model[0].detach().numpy()
            y_model = mpc.plant_model(state_i[-1], mpc.dt, u_model[:2*mpc.num_vehicle])
            
            predicted_state = np.array([y_model])
            for j in range(1, mpc.horizon):
                predicted = mpc.plant_model(
                    predicted_state[-1], mpc.dt, u_model[2*mpc.num_vehicle*j:2*mpc.num_vehicle*(j+1)])
                predicted_state = np.append(
                    predicted_state, np.array([predicted]), axis=0)
            predict_info_model += [predicted_state]

            state_i = np.append(state_i, np.array([y_model]), axis=0)
            u_i = np.append(u_i, np.array([u_model[:2*mpc.num_vehicle]]), axis=1)

    visualization = Visualization(mpc.num_vehicle, (model != None), sim_total, mpc.horizon, state_i[0], ref, simulation_options["name"], show, save)
    visualization.create_video(state_i, predict_info_opt, predict_info_model)
    
    state_i = state_i.reshape((sim_total+1, mpc.num_vehicle, 4))

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

    # TODO does not work for more then one vehicle
    iteration_until_vehicle_moves = max_j

    training_tensor = None
    label_tensor = None 
    simulation_sucessfull = False
    ###################
    # COLLECTING TRAINING DATA
    if collect_training_data:
        # check if the simulation was successful
        end_position = state_i[iteration_until_vehicle_moves][:, :3]
        desired_position = np.array(ref).reshape(mpc.num_vehicle, -1)

        distance_to_goal = np.mean(np.linalg.norm(end_position - desired_position, axis=1))
        EPS = 0.1
        
        simulation_sucessfull = distance_to_goal < EPS
        # For each data point we have a vehicle position (x,y), angle and velocity and the desired position and angle
        # That makes 7 values per time step times the range sim_total (num of vehicle will already be a dim to makes things easier later)
        # The ordering is like this [x, y, angle, v, x_d, y_d, angle_d]
        first_vehicle = np.append(state_i[:,0,:], np.tile(ref[:3], (state_i.shape[0], 1)), axis=1)
        second_vehicle = np.append(state_i[:,1,:], np.tile(ref[3:], (state_i.shape[0], 1)), axis=1)
        training_tensor = torch.tensor(np.append(first_vehicle, second_vehicle, axis=1)[:iteration_until_vehicle_moves])

        # Each label of each datapoints has to contain the predicted steps of the optimization, pedal and steering input at each 
        # step of the horizon
        # The ordering is like this [pedal, steering]
        label_tensor = torch.tensor(label_data.reshape(-1, mpc.horizon * 2 * mpc.num_vehicle)[1:iteration_until_vehicle_moves+1, :])
    
    return state_i, u_i, training_tensor, label_tensor, simulation_sucessfull
