import numpy as np
from pyrsistent import v
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = True

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 20
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference1 = [10, 8, 0]
        self.reference2 = None #[10, 2, 3.14/2]
        self.obstacles = np.array([[5.0, 0.0],
                                   [5.0, 2.0], 
                                   [5.0, 4.0]])

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
        
        state_history = np.array([state])
        control_history = np.zeros((1,2))
        obstacle_history = np.array([np.inf])
        
        for i in range(self.horizon):
            pedal = u[2*i]
            steering = u[2*i+1]
            state = self.plant_model(state, self.dt, pedal, steering)
            state_history = np.append(state_history, [state], axis=0)
            control_history = np.append(control_history, [u[2*i:2*i+2]], axis=0)
            
            obstacle_dist = np.linalg.norm(state[:2]-self.obstacles,axis=-1)
            obstacle_history = np.append(obstacle_history, obstacle_dist)
        
        # target cost + obstale cost + smoothness cost (steering angle and acceleration)
        cost = np.linalg.norm(state[:2]-ref[:2]) + np.abs(state[2]-ref[2]) \
                + np.sum((10000/obstacle_history) * (obstacle_history<1.5)) \
                + 0.1*np.mean(np.abs(control_history[1:,1]-np.abs(control_history[:-1,1]))) \
                + 0.1*np.mean(np.abs(state_history[1:,3]-np.abs(state_history[:-1,3]))) 
               
        


        return cost

sim_run(options, ModelPredictiveControl)
