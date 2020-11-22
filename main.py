import cv2
import numpy as np
from scipy.ndimage import convolve

def draw_map(img_w: int, img_h: int, map_np, beliefs_np, measure_map_np):
    img = np.zeros((img_w, img_h, 3), dtype=np.uint8) + 255

    grid_rows, grid_cols = map_np.shape
    
    cell_w = int(img_w / grid_cols)
    cell_h = int(img_h / grid_rows)

    cells = []

    for row in range(grid_rows):
        row_of_cells = []

        for col in range(grid_cols):
            cell = np.zeros((cell_w, cell_h, 3)) + (map_np[row, col] * 255)
            txt_x = int(cell_w/2) - int(cell_h * 0.3)
            txt_y = cell_h - int(cell_h * 0.4)

            # value text
            if map_np[row, col]:
                cell = cv2.putText(cell, f'{beliefs_np[row, col]:.2f}', (txt_x, txt_y), 3, 0.7, (0, 0, 255), 1, cv2.LINE_AA) 

            if measure_map_np[row, col]:
                cell = cv2.rectangle(cell, (5, 5), (cell_w - 4, cell_h - 4), (0, 0, 255), 2)


            row_of_cells.append(cell)
        cells.append(np.concatenate(np.array(row_of_cells), axis=1))

    cells_np = np.array(cells)
    cells_np = np.concatenate(cells_np, axis=0)

    img = cells_np

    # draw squares
    # img = cv2.circle(img, (x1, y1), 5, (123, 123, 0), 4)
    # img = cv2.circle(img, (x2, y2), 3, (255, 1, 0), 2)


    # x1 = int(col * cell_w)
    # y1 = int(row * cell_h) - 1
    # x2 = img_w
    # y2 = int(row * cell_h) - 1

    # img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1) 
    # img = cv2.line(img, (y1, x1), (y2, x2), (0, 0, 0), 1) 
            
    return img



def find_measurement_cells(map_np, measurement):  
    pad = (measurement, measurement)
    map_padded_np = np.pad(map_np, [pad, pad], mode='constant')
    
    grid_rows, grid_cols = map_padded_np.shape
    direction_maps = [np.zeros_like(map_np) for x in range(4)]

    mask_size = measurement * 2 + 1
    mask = np.zeros((mask_size, mask_size),  dtype=np.int64)
    mask_up = mask.copy()
    mask_up[0:measurement+1, measurement] = 1

    mask_right = mask.copy()
    mask_right[measurement, measurement:] = 1

    mask_down = mask.copy()
    mask_down[measurement:, measurement] = 1

    mask_left = mask.copy()
    mask_left[measurement, 0:measurement + 1] = 1

    for row in range(grid_rows - mask_size + 1):
        for col in range(grid_cols - mask_size + 1):
            print(row, col)
            s_from = slice(row, row + mask_size)
            s_to = slice(col, col + mask_size)

            patch = map_padded_np[s_from, s_to].copy()

            for m, dm in zip([mask_up, mask_right, mask_down, mask_left], direction_maps):
                patch_masked = patch & m
                if np.sum(patch_masked) == measurement + 1:
                    dm[row, col] = 1

    return direction_maps



img_w = 800
img_h = 800


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
beliefs_np = map_np * (1 / init_value)

measurement = 2
up, right, down, left = find_measurement_cells(map_np, measurement)

img = draw_map(img_w, img_h, map_np, beliefs_np, down)

cv2.imshow('img', img)
cv2.waitKey(0)