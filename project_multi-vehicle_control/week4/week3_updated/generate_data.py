import sys 
import torch
import numpy as np

if sys.platform == "darwin":
    FOLDER = "./week4/week3_updated/data"
else: 
    FOLDER = "/storage/remote/atcremers57/m_vehicle_c/data"

TRAINING_DATA_PATH = FOLDER + "/training_data_2_constrained.pt"
LABEL_DATA_PATH = FOLDER + "/label_data_2_constrained.pt"

def get_start_position(num_vehicles, constraints=False):
    POSITON_RANGE = 5
    ANGLE_EPS = 1/4 * 2 * np.pi
    if num_vehicles == 2:
        if constraints:
            pos_1 = np.random.uniform(low=[-POSITON_RANGE, -POSITON_RANGE], high=[POSITON_RANGE, POSITON_RANGE])
            distance= np.random.uniform(low=5, high=30)
            if pos_1[0] > 0 and pos_1[1] > 0:
                angle_1 = 1/4 * np.pi
            elif pos_1[0] > 0 and pos_1[1] < 0:
                angle_1 = 3/4 * np.pi
            elif pos_1[0] < 0 and pos_1[1] > 0:
                angle_1 = 5/4 * np.pi
            else:
                angle_1 = 7/4 * np.pi
            # take angle and distance and compute other position
            x_d = pos_1[0] + (distance * np.cos(angle_1))
            y_d = pos_1[1] + distance * np.sin(angle_1)
            pos_2 = np.array([x_d, y_d])
            # compute the other angle so that they are facing each other
            start = pos_1.tolist() + [angle_1+np.random.random()*ANGLE_EPS] + [0] + pos_2.tolist() + [angle_1+np.pi+np.random.random()*ANGLE_EPS] + [0]
            target = pos_2.tolist() + [angle_1+np.random.random()*ANGLE_EPS] + pos_1.tolist() + [angle_1+np.pi+np.random.random()*ANGLE_EPS]
        else:
            while True:
                ref_pos = np.random.uniform(low=[-POSITON_RANGE, -POSITON_RANGE], high=[POSITON_RANGE, POSITON_RANGE])
                start_pos = np.random.uniform(low=[-POSITON_RANGE, -POSITON_RANGE], high=[POSITON_RANGE, POSITON_RANGE])
                if np.linalg.norm(ref_pos - start_pos) > 5:
                    break
            ref_angle = np.random.uniform(low=[-np.pi], high=[np.pi])
            start_angle = np.random.uniform(low=[-np.pi], high=[np.pi])
            
            start = start_pos.tolist() + start_angle.tolist() + [0] + ref_pos.tolist() + ref_angle.tolist() + [0]
            target = ref_pos.tolist() + ref_angle.tolist() + start_pos.tolist() + start_angle.tolist()
    else:
        while True:
            ref_pos = np.random.uniform(low=[-POSITON_RANGE, -POSITON_RANGE], high=[POSITON_RANGE, POSITON_RANGE])
            start_pos = np.random.uniform(low=[-POSITON_RANGE, -POSITON_RANGE], high=[POSITON_RANGE, POSITON_RANGE])
            if np.linalg.norm(ref_pos - start_pos) > 5:
                break
        ref_angle = np.random.uniform(low=[-np.pi], high=[np.pi])
        start_angle = np.random.uniform(low=[-np.pi], high=[np.pi])
        start = start_pos.tolist() + start_angle.tolist() + [0]
        target = ref_pos.tolist() + ref_angle.tolist()

    return start, target


def load_data():
    training_data = torch.load(TRAINING_DATA_PATH)
    label_data = torch.load(LABEL_DATA_PATH)
    return training_data, label_data



    
    