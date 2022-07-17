import torch.nn as nn
import torch

class Loss(nn.Module):
    def __init__(self, horizon, num_vehicle, distance_cost, angle_cost, velocity_cost, 
                 smoothness_cost, traveled_distance_cost, collision_cost, collision_threshold = 5):
        super().__init__()
        self.horizon = horizon
        self.num_vehicle = num_vehicle
        self.distance_cost = distance_cost
        self.angle_cost = angle_cost
        self.velocity_cost = velocity_cost
        self.smoothness_cost = smoothness_cost
        self.traveled_distance_cost = traveled_distance_cost
        self.collision_cost = collision_cost
        self.collision_threshold = collision_threshold
        self.dt = 0.2
        
        
    
    # prev_states: dims -> [batchsize, vehicles, 4]
    # controls: dims -> [batchsize, vehicles, 2]
    # dt: scalar
    # return: dims -> [batchsize, vehicles, 4]
    def plant_models(self, prev_states, controls):
        assert prev_states.shape[1] == self.num_vehicle
        assert prev_states.shape[2] == 4
        assert controls.shape[1] == self.num_vehicle
        assert controls.shape[2] == 2

        x_t = prev_states[:,:,0]+prev_states[:,:,3]*torch.cos(prev_states[:,:,2])*self.dt
        y_t = prev_states[:,:,1]+prev_states[:,:,3]*torch.sin(prev_states[:,:,2])*self.dt
        psi_t = prev_states[:,:,2]+prev_states[:,:,3]*self.dt*torch.tan(controls[:,:,1])/2.0
        v_t = prev_states[:,:,3]+controls[:,:,0]*self.dt
        return torch.stack([x_t, y_t, psi_t, v_t], axis=2)
        
    
    # states: dims -> [batchsize, vehicles, 4]
    # states: refs -> [batchsize, vehicles, 3]
    # controls: dims -> [batchsize, horizon ,vehicles, 2]
    # return: cost -> scalar
    def forward(self, states, refs, controls):
        # check the dims of prev_states and controls
        assert states.shape[1] == self.num_vehicle
        assert states.shape[2] == 4
        assert refs.shape[1] == self.num_vehicle
        assert refs.shape[2] == 3
        assert controls.shape[1] == self.horizon
        assert controls.shape[2] == self.num_vehicle
        assert controls.shape[3] == 2

        # dims -> [horizon, batchsize, vehicles, 4]
        state_histories = states[None, :, :, :]
        current_states = states
        for i in range(self.horizon):
            current_states = self.plant_models(current_states, controls[:,i,:,:])
            state_histories = torch.cat([state_histories, current_states[None, :, :, :]], axis=0)

        cost = 0.0
        # TODO what distance cost is the correct one here? Previously we used 1
        # distance cost 1
        # cost += torch.mean(torch.linalg.norm(current_states[:, :, :2] - refs[:, :, :2], axis=2, ord=2))*self.distance_cost
        # distance cost 2

        cost += torch.mean(torch.linalg.norm(state_histories[:, :, :, :2] - refs[:,:,:2], axis=3, ord=2))*self.distance_cost

        # TODO same with angle cost
        # angle cost 1
        # cost += torch.mean(torch.abs(current_states[:, :, 2] - refs[:, :, 2]))*self.angle_cost
        # angle cost 2

        cost += torch.mean(torch.abs(state_histories[:, :, :, 2] - refs[:, :, 2]))*self.angle_cost
        
        # TODO velocity cost: the vehicle should stop at the goal position
        # velocity cost 1
        # cost += torch.mean(torch.abs(current_states[:, :, 3]))*self.velocity_cost

        # smoothness cost
        cost += torch.mean(torch.sum(torch.abs(torch.diff(controls[:,:,:,1], axis=1)), axis=1)) * self.smoothness_cost

        # distance traveled cost
        cost += torch.mean(torch.sum(torch.linalg.norm(torch.abs(torch.diff(state_histories[:,:,:,:2], axis=0)), axis=3), axis=0)) * self.traveled_distance_cost

        # collision cost
        # we can implement this later because we start with one vehicle
        for i in range(self.num_vehicle-1):
            dist_i = state_histories[:,:,i,:2][:,:,None,:] - state_histories[:,:,(i+1):,:2]
            dist_i = torch.linalg.norm(dist_i, ord=2, dim=-1)+1e-10
            
            cost += torch.mean(torch.sum((self.collision_cost/dist_i)*(dist_i<self.collision_threshold), axis=0))       
        
        return cost