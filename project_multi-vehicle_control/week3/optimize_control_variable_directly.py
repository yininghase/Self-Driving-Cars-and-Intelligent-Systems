import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import ModelPredictiveControl
import matplotlib.patches as mpatches
from visualization import Visualization
from generate_data import get_start_position
from inference_neuron_network import control_to_action

NUM_OF_VEHICLES = 1

class ControlVariable(nn.Module):
    def __init__(self, num_vehicle, num_control_variable, num_horizon):
        super(ControlVariable, self).__init__()
        
        self.num_vehicle = num_vehicle
        self.num_control_variable = num_control_variable
        self.num_horizon = num_horizon
        
        self.control_variables = nn.Parameter(torch.zeros(num_horizon, 
                                                          num_vehicle, 
                                                          num_control_variable, 
                                                          dtype=torch.float32,
                                                          requires_grad=True))
        
        self.dt = 0.2

    def forward(self, start_state):
        
        assert start_state.size() == (self.num_vehicle, 4)
        end_state = torch.zeros_like(start_state)

        for i in range(self.num_vehicle):
            
            x = start_state[i,0]
            y = start_state[i,1]
            psi = start_state[i,2]
            v = start_state[i,3]
            
            for j in range(self.num_horizon):
                
                x = x + v*self.dt*torch.cos(psi)
                y = y + v*self.dt*torch.sin(psi)
                psi = psi + v*self.dt*torch.tan(self.control_variables[j,i,1])/2.0
                v = 0.99*v + self.control_variables[j,i,0]*self.dt 
                
            end_state[i,0] = x
            end_state[i,1] = y
            end_state[i,2] = psi
            end_state[i,3] = v
        
        return end_state
    
    def get_control_variable(self):
        return self.control_variables
        
        
class Loss(nn.Module):
    def __init__(self, pos_cost=10, orient_cost=1, velocity_cost=0):
        super().__init__()
        self.pos_cost = pos_cost
        self.orient_cost = orient_cost
        self.velocity_cost = velocity_cost
    
    def forward(self, end_position, goal_position):
        
        loss = torch.sum(torch.abs(end_position[:,:2]-goal_position[:,:2]))*self.pos_cost \
               + torch.sum(torch.abs(end_position[:,2]-goal_position[:,2]))*self.orient_cost \
               + torch.sum(torch.abs(end_position[:,3]-goal_position[:,3]))*self.velocity_cost
               
        return loss

def one_step_vorward(prev_state, control_variables, dt=0.2):
    
    num_vehicle = len(prev_state)
    next_state = torch.zeros_like(prev_state)
    
    for i in range(num_vehicle):
            
        x = prev_state[i,0]
        y = prev_state[i,1]
        psi = prev_state[i,2]
        v = prev_state[i,3]
            
        next_state[i,0] = x + v*dt*torch.cos(psi)
        next_state[i,1] = y + v*dt*torch.sin(psi)
        next_state[i,2] = psi + v*dt*torch.tan(control_variables[i,1])/2.0
        next_state[i,3] = 0.99*v + control_variables[i,0]*dt 
    return next_state

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device available now:', device)
    
    torch.autograd.set_detect_anomaly(True)
    
    num_vehicle = 1
    num_control_variable = 2
    num_horizon = 20
    total_time = 100
    
    epochs = 300
    patience = 50
    weight_decay = 1e-6
    lr_patience = 10
    lr_factor = 0.1

    for index in tqdm(range(10)):
        start, target = get_start_position(1, False)

        model = ControlVariable(num_vehicle = num_vehicle , 
                                num_control_variable = num_control_variable,
                                num_horizon = num_horizon)
        
        model.train()
        
        start_position_init = torch.tensor([start], dtype=torch.float32)
        goal_position_init = torch.tensor([list(target)+[0]], dtype=torch.float32)

        start_position = start_position_init
        goal_position = goal_position_init
        
        # 3 dimensional tensor, first dimension is time
        #                       second dimension is vehicle
        #                       third dimension: x, y, psi, v
        trajectory = torch.zeros((total_time, num_vehicle, 4)) 
        prediction = torch.zeros((total_time, num_horizon, num_vehicle, 4))
        
        for t in range(total_time):
            
            best_loss = np.inf
            patience_f = patience
            learning_rate = 0.1
            
            optimizer = Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=lr_patience, verbose=True, factor=lr_factor)
            criterion = Loss()
            
            for epoch in range(epochs):
                
                optimizer.zero_grad()
                
                end_position = model.forward(start_position)
                loss = criterion(end_position, goal_position)
                loss.backward()
                
                optimizer.step()
                scheduler.step(loss)
                
                msg = f'Epoch: {epoch+1}/{epochs} | Loss: {loss:.6}'
                # print(msg)
                

                if  loss.item() <= best_loss:
                    best_loss = loss.item()
                    patience_f = patience
                    control_variables = model.get_control_variable().detach().clone().cpu()
                    
                    if t == 0:
                        control_variables_t0 = control_variables
                
                else:
                    patience_f -= 1
                
                if patience_f == 0:
                    break

            # apply the model to the prediction to get a predicted trajectory
            prediction_t = torch.zeros((num_horizon, num_vehicle, 4))
            curr_pos = start_position
            for i in range(num_horizon):
                prediction_t[i] = one_step_vorward(curr_pos, control_variables[i,:,:])
                curr_pos = prediction_t[i]

            prediction[t,:,:,:] = prediction_t
            start_position = one_step_vorward(start_position, control_variables[0,:,:])
            trajectory[t,:,:] = start_position
            
            if torch.sum(torch.abs(start_position[:,:3]-goal_position[:,:3])) <= 1e-2:
                break
        

        visualization = Visualization(num_vehicle, False, total_time, num_horizon, start_position_init[0].numpy(), goal_position_init[0,:3].numpy(), "gradient_descent_{}".format(index), False, True)
        visualization.create_video(trajectory.numpy(), prediction, None, )
        visualization.plot_trajectory(trajectory.numpy())
    
    
