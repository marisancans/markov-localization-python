import cv2
import numpy as np
import math
from enum import Enum

class Direction(Enum): # direction
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class Action(Enum):
    MEASUREMENT = 1
    MOVEMENT = 2


def draw_map(img_w: int, img_h: int, map_np, beliefs_np, measure_map_np=None):
    img = np.zeros((img_w, img_h, 3), dtype=np.uint8) + 255

    grid_rows, grid_cols = map_np.shape
    
    cell_w = int(img_w / grid_cols)
    cell_h = int(img_h / grid_rows)

    cells = []

    # for color
    c_max = beliefs_np.max()

    for row in range(grid_rows):
        row_of_cells = []

        for col in range(grid_cols):
            if map_np[row, col]:
                c = (map_np[row, col] + 1.0)
                cell_color = (c, c, c)
            # else:

            x = (beliefs_np[row, col] / c_max)
            cell_color = (0, x, 0)
                
            cell = np.zeros((cell_w, cell_h, 3)) + cell_color
            txt_x = int(cell_w/2) - int(cell_h * 0.4)
            txt_y = cell_h - int(cell_h * 0.4)

            # value text
            if map_np[row, col]:
                cell = cv2.putText(cell, f'{beliefs_np[row, col]:.3f}', (txt_x, txt_y), 3, 0.7, (0, 0, 255), 1, cv2.LINE_AA) 

            if isinstance(measure_map_np, np.ndarray):
                if measure_map_np[row, col]:
                    cell = cv2.rectangle(cell, (5, 5), (cell_w - 4, cell_h - 4), (0, 0, 255), 2)
            
            cell = cv2.rectangle(cell, (1, 1), (cell_w, cell_h), (0.5, 0.5, 0.5), 1)


            row_of_cells.append(cell)
        cells.append(np.concatenate(np.array(row_of_cells), axis=1))

    cells_np = np.array(cells)
    cells_np = np.concatenate(cells_np, axis=0)

    img = cells_np
    return img



def find_measurement_cells(map_np, direction: Direction, measurement):  
    grid_rows, grid_cols = map_np.shape
    direction_map = np.zeros_like(map_np)

    for row in range(grid_rows):
        for col in range(grid_cols): 
            nth = 0
            
            if direction == Direction.UP:
                for i in range(row, -1, -1): 
                    if map_np[i, col]:
                        nth += 1
                    else:
                        break

            if direction == Direction.DOWN:
                for i in range(row, grid_rows):
                    if map_np[i, col]:
                        nth += 1    
                    else:
                        break

            if direction == Direction.RIGHT:
                for i in range(col, grid_cols):
                    if map_np[row, i]:
                        nth += 1    
                    else:
                        break

            if direction == Direction.LEFT:
                for i in range(col, -1, -1):
                    if map_np[row, i]:
                        nth += 1    
                    else:
                        break

            if nth == measurement:
                direction_map[row, col] = 1

    return direction_map

def find_movement_map(beliefs_np, direction: Direction):
    move_map = beliefs_np.copy()

    # Shift array and pad it to the movement direction
    if direction == Direction.RIGHT:
        move_map = np.pad(move_map,((0,0),(1,0)), mode='constant')[:, :-1]

    if direction == Direction.UP:
        move_map = np.pad(move_map,((0,1),(0,0)), mode='constant')[1:, :]
   
    if direction == Direction.DOWN:
        move_map = np.pad(move_map,((1,0),(0,0)), mode='constant')[:-1, :]

    if direction == Direction.LEFT:
        move_map = np.pad(move_map,((0,0),(0,1)), mode='constant')[:, 1:]

    return move_map

def measurement_action(sensor_function, measurement_map, measurement, belief_map):
    sensor_probability = sensor_function(measurement)
    measurement_map_inverted = measurement_map ^ 1 # binary inverse

    measurement_map = measurement_map.astype(np.float64) * sensor_probability
    measurement_map_inverted = measurement_map_inverted.astype(np.float64) * (1 - sensor_probability)

    belief_map = belief_map * (measurement_map + measurement_map_inverted)
    return belief_map

def movement_action(belief_map, direction):
    movement_map = find_movement_map(belief_map, direction)
    belief_map = (map_np * movement_map.copy() * movement_probability) + (belief_map * (1 - movement_probability))
    return belief_map

def display_map(img_w, img_h, map_np, belief_map, img_name, measurement_map=None):
    img = draw_map(img_w, img_h, map_np, belief_map, measurement_map)
    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
    cv2.imshow(img_name, img)

def normalize(belief_maps):
    return belief_maps / belief_maps.sum()

if __name__ == "__main__":


    img_w = 800
    img_h = 800

    max_measurement = 5
    step_measurement = 1 / max_measurement 
    sensor_function = lambda x: math.pow(math.e, (-0.5 * float(x) * step_measurement))
    movement_probability = 0.9

    map = [
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    ]

    dimensions = 4 # up right down left
    map_np = np.array(map)

    grid_rows, grid_cols = map_np.shape

    init_value = (np.count_nonzero(map_np == 1) * dimensions)
    belief_maps = [map_np * (1 / init_value) for x in range(4)]
    belief_maps = np.array(belief_maps)
    belief_maps_sum = belief_maps.sum()

    # set the order of actions
    actions = [
        [ Action.MEASUREMENT, 2 ],
        [ Action.MOVEMENT, Direction.LEFT ],
        [ Action.MEASUREMENT, 2 ],
        [ Action.MOVEMENT, Direction.DOWN ],
        [ Action.MEASUREMENT, 2 ],
        [ Action.MOVEMENT, Direction.LEFT ],
        [ Action.MEASUREMENT, 3 ],
    ]

    # Execute actions

    # initialization
    # display_map(img_w, img_h, map_np, belief_maps[0], 'initialization')
    # cv2.waitKey(0)

    directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

    for action, value in actions:
        measurement_maps = []

        for nth, direction in enumerate(directions): 
            measurement_map = None
            print(action, value)

            if action == Action.MEASUREMENT:
                # Measurement
                measurement_map = find_measurement_cells(map_np, direction, value)
                belief_maps[nth] = measurement_action(sensor_function, measurement_map, value, belief_maps[nth])
            
            if action == Action.MOVEMENT:
                belief_maps[nth] = movement_action(belief_maps[nth], value)
            
            measurement_maps.append(measurement_map)
            # display_map(img_w, img_h, map_np, belief_maps[nth], f'{direction}_{action}', measurement_map)
        
        # Normalize
        belief_maps = normalize(belief_maps)

        for (nth, direction), measurement_map in zip(enumerate(directions), measurement_maps):
            display_map(img_w, img_h, map_np, belief_maps[nth], f'{direction}_{action}_normalization', measurement_map)

    cv2.waitKey(0)