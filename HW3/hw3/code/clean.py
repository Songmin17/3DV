import numpy as np
import os 

parent_path = '/Users/Olive1/Documents/2023-spring/3DV/new-hw/HW3/hw3_revised/results'
for item in os.listdir('../results'):
    if '.jpg' in item or '.obj' in item:
        os.remove(f'{parent_path}/{item}')
