from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import os
import torch

class Visualization:
    def __init__(self, num_vehicles, is_model, simulation_length, horizon, start, target, name, show, save):
        self.figure_size = 8
        self.figure_limit = 25
        self.step = 5
        self.car_width = 1.0
        self.num_vehicles = num_vehicles
        self.is_model = is_model
        self.folder = "./results/plots/"
        self.simulation_length = simulation_length
        self.horizon = horizon
        self.start = start.reshape(self.num_vehicles, 4)[:,:3]
        self.target = np.array(target).reshape(self.num_vehicles, 3)
        self.name = name
        self.show = show
        self.save = save
    
    # start: [num_vehicle, [x, y, psi]]
    # target: [num_vehicle, [x, y, psi]]
    def base_plot(self, is_trajectory):
        self.fig = plt.figure(figsize=(self.figure_size, self.figure_size))
        self.ax = self.fig.add_subplot()
        plt.xlim(-self.figure_limit, self.figure_limit)
        self.ax.set_ylim([-self.figure_limit, self.figure_limit])
        plt.xticks(np.arange(-self.figure_limit, self.figure_limit, step=self.step))
        plt.yticks(np.arange(-self.figure_limit, self.figure_limit, step=self.step))

        self.patch_vehicles = []
        self.patch_target = []
        self.predicts_opt = []
        self.predicts_model = []

        for i in range(self.start.shape[0]):
            # cars
            x, y, psi = self.start[i]
            x_target, y_target, psi_target = self.target[i]
            
            x = x - 1.25 * np.cos(psi)
            y = y - 1.25 * np.sin(psi)
            x_target = x_target- 1.25 * np.cos(psi_target)
            y_target = y_target - 1.25 * np.sin(psi_target)

            patch_car = mpatches.Rectangle([0,0], self.car_width, 2.5, fc='k', fill=False)
            patch_car.set_xy(car_patch_pos(x, y, psi, self.car_width))
            patch_car.angle = np.rad2deg(psi)-90
            self.patch_vehicles.append(patch_car)

            patch_goal = mpatches.Rectangle([0,0], self.car_width, 2.5, fc='b', ls='dashdot',  fill=False)
            patch_goal.set_xy(car_patch_pos(x_target, y_target, psi_target, self.car_width))
            patch_goal.angle = np.rad2deg(psi_target)-90
            self.patch_target.append(patch_goal)

            self.ax.add_patch(patch_car)
            self.ax.add_patch(patch_goal)

            # trajectories
            if i == 0 and (not is_trajectory):
                predict_opt, = self.ax.plot([], [], 'r--', linewidth=1, label="Optimization")
            else:
                predict_opt, = self.ax.plot([], [], 'r--', linewidth=1)
            self.predicts_opt.append(predict_opt)
            if self.is_model:
                if i == 0 and (not is_trajectory):
                    predict_model, = self.ax.plot([], [], 'b--', linewidth=1, label="Model Prediction")
                else:
                    predict_model, = self.ax.plot([], [], 'b--', linewidth=1)
                self.predicts_model.append(predict_model)
            plt.legend(loc='upper left', fontsize=12)
    
    def create_video(self, data, predict_opt, predict_model):
        self.base_plot(is_trajectory=False)
        self.data = data.reshape(-1, self.num_vehicles, 4)[:,:,:3]
        if self.is_model:
            self.predict_model = np.concatenate(predict_model[1:], axis=0).reshape(self.simulation_length, self.horizon, self.num_vehicles, 4)
        if torch.is_tensor(predict_opt):
            self.predict_opt = predict_opt[:, :, :, :2]
        else:
            self.predict_opt = np.concatenate(predict_opt[1:], axis=0).reshape(self.simulation_length, self.horizon, self.num_vehicles, 4)
        car_animation = animation.FuncAnimation(self.fig, self.update_plot, frames=range(len(data)-1), interval=100, repeat=True, blit=False)
        if self.save:
            car_animation.save(os.path.join(self.folder, self.name+".gif"))
        if self.show:
            plt.show()

    def update_plot(self, num):    
        for i in range(self.num_vehicles):
            # vehicle
            x, y, psi = self.data[num, i, :]
            x = x - 1.25 * np.cos(psi)
            y = y - 1.25 * np.sin(psi)

            self.patch_vehicles[i].set_xy(car_patch_pos(x, y, psi, self.car_width))
            self.patch_vehicles[i].angle = np.rad2deg(psi)-90

            # Goal.
            x_goal, y_goal, psi_goal = self.target[i,:]
            x_goal = x_goal- 1.25 * np.cos(psi_goal)
            y_goal = y_goal - 1.25 * np.sin(psi_goal)

            self.patch_target[i].set_xy(car_patch_pos(x_goal, y_goal, psi_goal, self.car_width))
            self.patch_target[i].angle = np.rad2deg(psi_goal)-90


            self.predicts_opt[i].set_data(self.predict_opt[num, :, i, 0], self.predict_opt[num, :, i, 1])
            if self.is_model:
                self.predicts_model[i].set_data(self.predict_model[num, :, i, 0], self.predict_model[num, :, i, 1])
    
    def plot_trajectory(self, points):
        self.base_plot(is_trajectory=True)
        max_time = points.shape[0]
        for i in range(self.num_vehicles):
            veh_points = points[:, i, :2][:,None,:]
            segments = np.concatenate([veh_points[:-1], veh_points[1:]], axis=1)
            norm = plt.Normalize(0, max_time)
            lc = LineCollection(segments, cmap="viridis", norm=norm)
            lc.set_array(range(self.simulation_length+1))
            lc.set_linewidth(4)
            line = self.ax.add_collection(lc)
        cbar = self.fig.colorbar(line, ax=self.ax)
        cbar.ax.set_ylabel("Timestep", fontsize=15)
        if self.show:
            plt.show()
        if self.save:
            plt.savefig(os.path.join(self.folder, self.name+".png"), bbox_inches='tight')
    
    def visualize_data_point(self, label, prediction):
        self.base_plot(is_trajectory=False)
        for i in range(self.num_vehicles):
            self.predicts_opt[i].set_data(label[:, i, 0], label[:, i, 1])
            self.predicts_model[i].set_data(prediction[:, i, 0], prediction[:, i, 1])
        if self.show:
            plt.show()
        if self.save:
            plt.savefig(os.path.join(self.folder, self.name+".png"), bbox_inches='tight')

def car_patch_pos(x, y, psi, car_width):
    x_new = x - np.sin(psi)*(car_width/2)
    y_new = y + np.cos(psi)*(car_width/2)
    return [x_new, y_new]
        