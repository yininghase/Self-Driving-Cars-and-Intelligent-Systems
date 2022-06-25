from cmath import pi
from matplotlib.pyplot import show
import numpy as np
from model import ModelPredictiveControl
from simulation import sim_run



options = {}
options['FIG_SIZE'] = [8, 8]
use_random = False
show_image = True
save_image = True


if __name__ == '__main__':
    simulation_options = {
        "horizon": 15,
        "start": [0,  0,  3.14/2,  0,
                  10, 10, 0,  0,],
        "target": [10, 10, 0,
                    0, 0, 3.14/2],
        "distance_cost": 1, #100, #10000,
        "angle_cost": 1, #100, #10000,
        "collision_cost": 0,
        "control_init": None,
        "obstacle_cost": 0,
        "non_round_obstacles": [[]],
        "name": "no_collision_cost"
    }
    
    if use_random:
        rand1 = np.random.uniform(low=[-5,-5], high=[5,5])
        rand2 = np.random.uniform(low=[-np.pi], high=[np.pi])
        rand3 = np.random.uniform(low=[-np.pi], high=[np.pi])
        if np.linalg.norm(rand1)<=3:
            rand1 += 3*rand1/np.linalg.norm(rand1)
        pos1 = np.array([5,5])+rand1
        pos2 = np.array([5,5])-rand1
        
        orient1 = rand2
        orient2 = rand3 #rand2
        # orient2 = rand2+np.pi if rand2<=0 else rand2-np.pi
        
        
        simulation_options["start"] = pos1.tolist() + orient1.tolist() + [0] \
                                    + pos2.tolist() + orient2.tolist() + [0]
        simulation_options["target"] = pos2.tolist() + orient1.tolist() \
                                     + pos1.tolist() + orient2.tolist()
            
    
    # # # # # # simulation with no collision cost
    state, u = sim_run(options, simulation_options, ModelPredictiveControl, save=save_image, show=show_image)
    
    # # # # # # simulation with collision cost
    simulation_options["collision_cost"] = 1 #100 #10000
    simulation_options["name"] = "with_collision_cost"
    _, _ = sim_run(options, simulation_options, ModelPredictiveControl, save=save_image, show=show_image)

    # # # # # # simulation with collision cost and better initialization
    simulation_options["control_init"] = u
    simulation_options["name"] = "with_collision_cost_and_better_initialization"
    _, _ = sim_run(options, simulation_options, ModelPredictiveControl, save=save_image, show=show_image)
    
    
    ###### 4-vehicle simulation ######
    
    simulation_options["start"] = [0,  0,  3.14/2,  0,
                                   10, 0,  3.14,    0,
                                   10, 10, -3.14/2,  0,
                                   0, 10, 0,  0,
                                   ]
    simulation_options["target"] = [10, 10, 3.14/2,
                                    0,  10, 3.14,
                                    0,   0, -3.14/2,
                                    10,  0, 0,
                                    ]
    simulation_options["control_init"] = None
    
    # # # # # simulation with no collision cost
    simulation_options["collision_cost"] = 0
    simulation_options["name"] = "no_collision_cost"
    state, u = sim_run(options, simulation_options, ModelPredictiveControl, save=save_image, show=show_image)
    
    # # # # # simulation with collision cost
    simulation_options["collision_cost"] = 1
    simulation_options["name"] = "with_collision_cost"
    _, _ = sim_run(options, simulation_options, ModelPredictiveControl, save=save_image, show=show_image)
    
    # # # # # # simulation with collision cost and better initialization
    simulation_options["control_init"] = u
    simulation_options["name"] = "with_collision_cost_and_better_initialization"
    _, _ = sim_run(options, simulation_options, ModelPredictiveControl, save=save_image, show=show_image)
    
    ###### NON-ROUND OBSTACLES ######

    # # # # # simulation without collision cost but with obstacles cost
    simulation_options["non_round_obstacles"] =  [
        {"p": [4, 4], "w": 2, "h": 2},
    ]
    simulation_options["collision_cost"] = 0
    simulation_options["obstacle_cost"] = 1
    simulation_options["name"] = "obstacle_without_collision"
    _, _ = sim_run(options, simulation_options, ModelPredictiveControl, save=save_image, show=show_image)

    # OBSTACLE_WIDTH = 0.125
    # ROAD_WIDTH = 3
    # ROAD_LENGTH = 15

    # simulation_options["non_round_obstacles"] =  [
    #     {"p": [2, 8], "w": OBSTACLE_WIDTH, "h": -ROAD_LENGTH},
    #     {"p": [2, 8], "w": ROAD_LENGTH, "h": OBSTACLE_WIDTH},
    #     {"p": [2+ROAD_WIDTH, 8-ROAD_WIDTH], "w": OBSTACLE_WIDTH, "h": -ROAD_LENGTH},
    #     {"p": [2+ROAD_WIDTH, 8-ROAD_WIDTH], "w": ROAD_LENGTH, "h": OBSTACLE_WIDTH}
    # ]

    # simulation_options["control_init"] = None
    # simulation_options["obstacle_cost"] = 10
    # simulation_options["start"] = [3.5,  0,  3.14/2,  0]
    # simulation_options["target"] = [10, 6.5, 0]

    # _, _ = sim_run(options, simulation_options, ModelPredictiveControl, save=True, show=True)
    
    