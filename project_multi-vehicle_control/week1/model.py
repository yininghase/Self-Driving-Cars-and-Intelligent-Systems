import numpy as np
from simulation import sim_run

# Simulator options.


class ModelPredictiveControl:
    def __init__(self, options):
        self.horizon = options["horizon"]
        self.dt = 0.2
        self.reference1 = options["reference"]
        self.reference2 = None
        self.obstacles = options["obstacles"]
        self.dis_cost = options["distance_cost"]
        self.ang_cost = options["angle_cost"]
        self.obs_cost = options["obstacle_cost"]
        self.smooth_cost = options["smoothness_cost"]

    def plant_model(self, prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]

        # Vehicle Dynamic Equation
        x_t = x_t+v_t*np.cos(psi_t)*dt
        y_t = y_t+v_t*np.sin(psi_t)*dt
        psi_t = psi_t + v_t*dt*np.tan(steering)/2.0
        v_t = 0.99*v_t+pedal*dt

        return [x_t, y_t, psi_t, v_t]

    def cost_function(self, u, *args):
        ''' u[2*i]: pedal;  u[2*i+1]: steering angle
            state[0]: x;    state[1]: y
            state[2]: angle;state[3]: velocity
        '''
        state = np.array(args[0])
        ref = np.array(args[1])

        if self.smooth_cost:
            state_history = np.array([state])
            control_history = np.zeros((1, 2))
        
        if self.obs_cost and len(self.obstacles) != 0:   
            obstacle_history = np.array([np.inf])

        for i in range(self.horizon):
            pedal = u[2*i]
            steering = u[2*i+1]
            state = self.plant_model(state, self.dt, pedal, steering)
            if self.smooth_cost:
                state_history = np.append(state_history, [state], axis=0)
                control_history = np.append(control_history, [u[2*i:2*i+2]], axis=0)

            if self.obs_cost and len(self.obstacles) != 0:
                obstacle_dist = np.linalg.norm(
                    state[:2]-self.obstacles, axis=-1)
                obstacle_history = np.append(obstacle_history, obstacle_dist)

        # target cost + obstale cost + smoothness cost (steering angle and pedal input)
        cost = np.linalg.norm(state[:2]-ref[:2]) * self.dis_cost + np.abs(state[2]-ref[2]) * self.ang_cost
        if self.obs_cost and len(self.obstacles) != 0:
            cost += np.sum((1/obstacle_history) * (obstacle_history < 1.5)) * self.obs_cost
        if self.smooth_cost:
            cost += np.mean(np.abs(control_history[1:, 1]-np.abs(control_history[:-1, 1]))) * self.smooth_cost \
                + np.mean(np.abs(control_history[1:, 0] - np.abs(control_history[:-1, 0]))) * self.smooth_cost
        return cost
