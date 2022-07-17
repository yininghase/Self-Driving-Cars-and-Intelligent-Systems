import torch.nn as nn
import numpy as np

class ModelPredictiveControl:
    def __init__(self, options):
        self.horizon = options["horizon"]
        self.dt = 0.2
        self.start = options["start"]
        self.target = options["target"]
        self.control_init = options["control_init"]
        self.dis_cost = options["distance_cost"]
        self.ang_cost = options["angle_cost"]
        self.col_cost = options["collision_cost"]
        self.non_round_obs = options["non_round_obstacles"]
        self.obs_cost = options["obstacle_cost"]
        
        self.non_round_obs_four = []
        for obs in self.non_round_obs:
            self.non_round_obs_four.append([obs["p"], [obs["p"][0]+obs["w"], obs["p"][1]], [obs["p"][0], obs["p"][1]+obs["h"]], [obs["p"][0]+obs["w"], obs["p"][1]+obs["h"]]])
        self.non_round_obs_four = np.array(self.non_round_obs_four)

        assert len(self.start)/4 == len(self.target)/3
        self.num_vehicle = int(len(self.start)/4)

    def plant_model(self, prev_state, dt, control):
        
        next_state = []
        
        for i in range(self.num_vehicle):
            x_t = prev_state[4*i+0]
            y_t = prev_state[4*i+1]
            psi_t = prev_state[4*i+2]
            v_t = prev_state[4*i+3]
            pedal = control[2*i]
            steering = control[2*i+1]

            # Vehicle Dynamic Equation
            x_t = x_t+v_t*np.cos(psi_t)*dt
            y_t = y_t+v_t*np.sin(psi_t)*dt
            psi_t = psi_t + v_t*dt*np.tan(steering)/2.0
            v_t = 0.99*v_t+pedal*dt
            next_state.extend([x_t, y_t, psi_t, v_t])

        return next_state

    def cost_function(self, u, *args):
        ''' u[2*N*t+2*i]: pedal of vehicle i at time t;  
            u[2*N*t+2*i+1]: steering angle of vehicle i at time t;
            state[4*i+0]: x of vehicle i;    
            state[4*i+1]: y of vehicle i;
            state[4*i+2]: angle of vehicle i;
            state[4*i+3]: velocity of vehicle i;
        '''
        state = np.array(args[0])
        ref = np.array(args[1])
        
        state_history =[state]
        
        for i in range(self.horizon):
            control = u[2*self.num_vehicle*i:2*self.num_vehicle*(i+1)]
            state = self.plant_model(state, self.dt, control)
            state_history.append(state)

        # target cost + obstale cost + smoothness cost (steering angle and pedal input)
        state = np.array(state).reshape(self.num_vehicle,4)
        ref = np.array(ref).reshape(self.num_vehicle,3)
        cost = np.sum(np.linalg.norm(state[:,:2]-ref[:,:2], axis=1, ord=2))*self.dis_cost \
               + np.sum(np.abs(state[:,2]-ref[:,2]))*self.ang_cost

        state_history = np.array(state_history)
        state_history = state_history.reshape((self.horizon+1, self.num_vehicle, 4))
        
        for obs in self.non_round_obs_four:
            dis_x = np.clip(np.maximum(np.min(obs[:,0]) - state_history[:,:,0], state_history[:,:,0] - np.max(obs[:,0])), a_min=0, a_max=None)
            dis_y = np.clip(np.maximum(np.min(obs[:,1]) - state_history[:,:,1], state_history[:,:,1] - np.max(obs[:,1])), a_min=0, a_max=None)
            dis = np.linalg.norm(np.stack([dis_x, dis_y], axis=1), axis=1, ord=2)
            if self.obs_cost > 0:
                cost += np.mean((1/dis) * (dis < 3)) * self.obs_cost


        if self.col_cost and self.num_vehicle > 1:
            dist = np.array([np.inf]) 
            for i in range(self.num_vehicle-1):
                dist_i = state_history[:, i, :2][:,None,:] - state_history[:, (i+1):, :2]
                dist_i = np.linalg.norm(dist_i,ord=2,axis=-1).flatten()
                dist = np.append(dist, dist_i)
            cost += np.sum((1/dist) * (dist < 3))*self.col_cost

        
        return cost



class NeuronNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        modules = []
        layer = len(hidden_sizes)
        for i in range(layer):
            if i == 0:
                modules.append(nn.Linear(input_size, hidden_sizes[i]))
                modules.append(nn.BatchNorm1d(hidden_sizes[i]))
                modules.append(nn.ReLU())
            elif i == layer-1:
                modules.append(nn.Linear(hidden_sizes[i-1], output_size))
            else:
                modules.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                modules.append(nn.BatchNorm1d(hidden_sizes[i]))
                modules.append(nn.ReLU())
        self.net = nn.Sequential(*modules)
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.net(x)
        return x        

