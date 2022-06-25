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

def sim_run(options, simulation_options, MPC, save=False, name=None, show=True):
    start = time.process_time()
    # Simulator Options
    FIG_SIZE = options['FIG_SIZE']  # [Width, Height]

    mpc = MPC(simulation_options)

    num_inputs = 2*mpc.num_vehicle
    
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
    predict_info = [state_i]

    for i in tqdm(range(1, sim_total+1)):
        # if mpc.control_init is None:
        u = np.delete(u, np.arange(2*mpc.num_vehicle))
        u = np.append(u, u[-2*mpc.num_vehicle:])
        # else:
        #     if mpc.horizon+i < sim_total+1:
        #         u = (mpc.control_init.flatten())[i*num_inputs:(mpc.horizon+i)*num_inputs]
        #     else:
        #         u = (mpc.control_init.flatten())[i*num_inputs:]
        #         u = np.append(u, np.zeros(mpc.horizon*num_inputs-len(u)))
                
                
        start_time = time.time()

        # Non-linear optimization.
        u_solution = minimize(mpc.cost_function, u, (state_i[-1], ref),
                              method='SLSQP',
                              bounds=bounds,
                              tol=1e-5)
        # print('Step ' + str(i) + ' of ' + str(sim_total) +
        #       '   Time ' + str(round(time.time() - start_time, 5)))
        u = u_solution.x
        y = mpc.plant_model(state_i[-1], mpc.dt, u[:2*mpc.num_vehicle])

        predicted_state = np.array([y])
        for j in range(1, mpc.horizon):
            predicted = mpc.plant_model(
                predicted_state[-1], mpc.dt, u[2*mpc.num_vehicle*j:2*mpc.num_vehicle*(j+1)])
            predicted_state = np.append(
                predicted_state, np.array([predicted]), axis=0)
        predict_info += [predicted_state]
        state_i = np.append(state_i, np.array([y]), axis=0)
        u_i = np.append(u_i, np.array([u[:2*mpc.num_vehicle]]), axis=0)

    ###################
    # SIMULATOR DISPLAY

    # Total Figure
    fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
    # gs = gridspec.GridSpec(8, 8)

    # Elevator plot settings.
    ax = fig.add_subplot()

    plt.xlim(-3, 13)
    ax.set_ylim([-3, 13])
    plt.xticks(np.arange(0, 11, step=2))
    plt.yticks(np.arange(0, 11, step=2))
    plt.title('MPC 2D')

    # Time display.
    time_text = ax.text(6, 0.5, '', fontsize=15)
    
    patch_cars = []
    patch_goals = []
    predicts = []

    # Main plot info.
    for i in range(mpc.num_vehicle):
        car_width = 1.0
        start_position = (mpc.start[4*i+0],mpc.start[4*i+1])
        goal_position = (mpc.target[3*i+0],mpc.target[3*i+1])
        patch_cars.append(mpatches.Rectangle(start_position, car_width, 2.5, fc='k', fill=False))
        patch_goals.append(mpatches.Rectangle(goal_position, car_width, 2.5, fc='b', ls='dashdot', fill=False))

        ax.add_patch(patch_cars[-1])
        ax.add_patch(patch_goals[-1])
        predict, = ax.plot([], [], 'r--', linewidth=1)
        predicts.append(predict)

    patch_obs = []
    for obs in simulation_options["non_round_obstacles"]:
        patch_obs.append(mpatches.Rectangle(obs["p"], obs["w"], obs["h"], fc='k', fill=True))
        ax.add_patch(patch_obs[-1])

    # Shift xy, centered on rear of car to rear left corner of car.

    def car_patch_pos(x, y, psi):
        # return [x,y]
        x_new = x - np.sin(psi)*(car_width/2)
        y_new = y + np.cos(psi)*(car_width/2)
        return [x_new, y_new]

    def update_plot(num):
        
        for i in range(mpc.num_vehicle):
            # vehicle
            x, y, psi = state_i[num, i, :3]
            x = x - 1.25 * np.cos(psi)
            y = y - 1.25 * np.sin(psi)

            patch_cars[i].set_xy(car_patch_pos(x, y, psi))
            patch_cars[i].angle = np.rad2deg(psi)-90

            # Goal.
            x_goal, y_goal, psi_goal = ref[i*3:(i+1)*3]
            x_goal = x_goal- 1.25 * np.cos(psi_goal)
            y_goal = y_goal - 1.25 * np.sin(psi_goal)

            patch_goals[i].set_xy(car_patch_pos(x_goal, y_goal, psi_goal))
            patch_goals[i].angle = np.rad2deg(psi_goal)-90
        
            predicts[i].set_data(predict_info[num][:, 4*i+0], predict_info[num][:, 4*i+1])

        return patch_cars, time_text

    #print("Compute Time: ", round(time.process_time() - start, 3), "seconds.")
    # Animation.

    def plot_trajectory(points):
        fig_traj = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
        gs_traj = gridspec.GridSpec(8, 8)
        ax_traj = fig_traj.add_subplot(gs_traj[:8, :8])
        # settings
        plt.xlim(-3, 14)
        ax_traj.set_ylim([-3, 14])
        plt.xticks(np.arange(0, 11, step=2))
        plt.yticks(np.arange(0, 11, step=2))

        max_time = max([p.shape[0] for p in points])

        # plot the trajectory with coloring according to throttle
        for i in range(mpc.num_vehicle):
            veh_points = points[i][:,None,:]
            norm = plt.Normalize(0, max_time)
            segments = np.concatenate([veh_points[:-1], veh_points[1:]], axis=1)
            lc = LineCollection(segments, cmap="viridis", norm=norm)
            lc.set_array(range(sim_total+1))
            lc.set_linewidth(4)
            line = ax_traj.add_collection(lc)

            start_i = simulation_options["start"][i*4:(i+1)*4]
            ref_i = ref[i*3:(i+1)*3]

            # start
            x, y, psi = start_i[0], start_i[1], start_i[2]
            x = x - 1.25 * np.cos(psi)
            y = y - 1.25 * np.sin(psi)

            patch_car = mpatches.Rectangle([0,0], car_width, 2.5, fc='k', fill=False)
            patch_car.set_xy(car_patch_pos(x, y, psi))
            patch_car.angle = np.rad2deg(psi)-90

            # goal
            x_goal, y_goal, psi_goal = ref_i[0], ref_i[1], ref_i[2]
            x_goal = x_goal- 1.25 * np.cos(psi_goal)
            y_goal = y_goal - 1.25 * np.sin(psi_goal)

            patch_goal = mpatches.Rectangle([0,0], car_width, 2.5, fc='b', ls='dashdot',  fill=False)
            patch_goal.set_xy(car_patch_pos(x_goal, y_goal, psi_goal))
            patch_goal.angle = np.rad2deg(psi_goal)-90

            ax_traj.add_patch(patch_car)
            ax_traj.add_patch(patch_goal)

        patch_obs = []
        for obs in simulation_options["non_round_obstacles"]:
            patch_obs.append(mpatches.Rectangle(obs["p"], obs["w"], obs["h"], fc='k', fill=True))
            ax_traj.add_patch(patch_obs[-1])
        
        cbar = fig_traj.colorbar(line, ax=ax_traj)
        cbar.ax.set_ylabel("Timestep", fontsize=15)

        if save:
            if not os.path.exists('./results/output'):
                os.makedirs('./results/output')
            plt.savefig('./results/output/' +
                        simulation_options['name'] + '.png', bbox_inches='tight')
        if show:
            plt.show()

    car_ani = animation.FuncAnimation(fig, update_plot, frames=range(
        1, len(state_i)), interval=100, repeat=True, blit=False)
    
    state_i = state_i.reshape((sim_total+1, mpc.num_vehicle, 4))

    # we just want to plot the range of points for which the vehicle still moves or will still move
    points_plot = []
    for i in range(mpc.num_vehicle):
        change_in_pos = np.sum(np.abs(state_i[1:,i,:2] - state_i[:-1,i,:2]), axis=1)
        for j in range(sim_total):
            if np.sum(change_in_pos[j:]) < 0.1:
                points_plot.append(state_i[:j, i, :2])
                break

    plot_trajectory(points_plot)

    if save:
        if not os.path.exists('./results/output'):
            os.makedirs('./results/output')
        car_ani.save('./results/output/' +
                     simulation_options['name'] + '.gif')
    if show:
        plt.show()
    
    return state_i, u_i
