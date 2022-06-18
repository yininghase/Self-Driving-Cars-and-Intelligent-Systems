import numpy as np
from model import ModelPredictiveControl
from simulation import sim_run

options = {}
options['FIG_SIZE'] = [8, 8]
options['OBSTACLES'] = True


if __name__ == '__main__':
    simulation_options = {
        "reference": [10, 2, -3.14/2],
        "horizon": 30,
        "obstacles": np.array([]),
        "distance_cost": 1,
        "angle_cost": 1,
        "obstacle_cost": 0,
        "smoothness_cost": 0,
        "name": "base"
    }
    # base simulation
    starting_positions = [[10, 2, -3.14/2], [10, 10, -3.14/2], [10, 10, 0]]
    for index, s in enumerate(starting_positions):
        simulation_options["reference"] = s
        simulation_options["name"] = "base_{}".format(index)
        sim_run(options, simulation_options,
                ModelPredictiveControl, save=True, show=False)

    # obstacle simulation
    obstacles_1 = np.array([[5.0, -1.0]])
    obstacles_2 = np.array([[5.0, 0.0], [5.0, 2.0], [5.0, -2.0]])
    obs_list = [obstacles_1, obstacles_2]
    desired_position = [[10, 0, 0], [10, 0, 3.14/2]]
    for index, s in enumerate(desired_position):
        for index_o, o in enumerate(obs_list):
            simulation_options["reference"] = s
            simulation_options["obstacles"] = o
            simulation_options["obstacle_cost"] = 1
            simulation_options["name"] = "obstacles_s{}_o{}".format(
                index, index_o)
            sim_run(options, simulation_options,
                    ModelPredictiveControl, save=True, show=False)

    # smoothness simulation
    simulation_options["reference"] = [10, 10, 3.14/2]
    simulation_options["obstacles"] = np.array([])
    simulation_options["smoothness_cost"] = 0.0
    smoothness_list = [0, 0.1, 1, 10]
    for index, s in enumerate(smoothness_list):
        simulation_options["smoothness_cost"] = s
        simulation_options["name"] = "smoothness_{}".format(index)
        sim_run(options, simulation_options,
                ModelPredictiveControl, save=True, show=False)

    # random simulations 1
    simulation_options["obstacles"] = np.array(
        [[5.0, 0.0], [5.0, 2.0], [5.0, -2.0], [5.0, -4.0]])
    simulation_options["reference"] = [10, 0, -3.14/2]
    simulation_options["obstacle_cost"] = 1
    simulation_options["smoothness_cost"] = 1
    simulation_options["name"] = "random_1"
    sim_run(options, simulation_options,
            ModelPredictiveControl, save=True, show=False)

    simulation_options["obstacle_cost"] = 10000
    simulation_options["smoothness_cost"] = 1
    simulation_options["name"] = "random_2"
    sim_run(options, simulation_options,
            ModelPredictiveControl, save=True, show=False)

    # random simulation 2
    simulation_options["obstacles"] = np.array(
        [[5.0, 0.0], [5.0, 2.0], [5.0, -2.0], [5.0, -4.0]])
    simulation_options["reference"] = [10, 0, 0]
    simulation_options["obstacle_cost"] = 10000
    simulation_options["smoothness_cost"] = 10
    simulation_options["name"] = "random_3"
    sim_run(options, simulation_options,
            ModelPredictiveControl, save=True, show=False)

    simulation_options["smoothness_cost"] = 0.1
    simulation_options["name"] = "random_4"
    sim_run(options, simulation_options,
            ModelPredictiveControl, save=True, show=False)

    # random simultaion 3
    simulation_options["horizon"] = 5
    simulation_options["reference"] = [10, 0, -3.14/2]
    simulation_options["obstacles"] = np.array(
        [[5.0, 0.0], [5.0, 2.0], [5.0, -2.0]])
    simulation_options["obstacle_cost"] = 10000
    simulation_options["smoothness_cost"] = 0.1
    simulation_options["name"] = "random_5"
    sim_run(options, simulation_options,
            ModelPredictiveControl, save=True, show=False)

    simulation_options["horizon"] = 30
    simulation_options["name"] = "random_6"
    sim_run(options, simulation_options,
            ModelPredictiveControl, save=True, show=False)
