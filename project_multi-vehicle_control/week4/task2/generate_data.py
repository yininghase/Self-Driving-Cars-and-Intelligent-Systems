import numpy as np

def generate_data(data_length, num_vehicles, position_range=5):
    low_range = [-position_range, -position_range, -np.pi, -10]
    high_range = [position_range, position_range, np.pi, 10]
    starts = np.random.uniform(low=low_range, high=high_range, size=(data_length, num_vehicles, 4))
    targets = np.random.uniform(low=low_range[:3], high=high_range[:3], size=(data_length, num_vehicles, 3))
    
    # pos_diff = np.linalg.norm(starts[:,:,:2]-targets[:,:,:2], axis=-1)
    # factor = pos_diff/10
    # factor[factor>=1] = 1
    
    # angle_diff = starts[:,:,2]-targets[:,:,2]
    # angle_diff *= factor
    # targets[:,:,2] = starts[:,:,2] - angle_diff
    
    return np.append(starts, targets, axis=2)