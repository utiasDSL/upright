import numpy as np
import rigeo as rg

import IPython


BASE_BOARD_MASS = 0.076
BOTTLE_MASS = 0.722
BOX1_MASS = 0.933 - BASE_BOARD_MASS - BOTTLE_MASS
BOX2_MASS = 1.046 - BASE_BOARD_MASS - BOTTLE_MASS - BOX1_MASS

base_board = rg.Box.from_side_lengths(
    side_lengths=[0.15, 0.15, 0.005], center=[0, 0, 0.0025]
)
box1 = rg.Box.from_side_lengths(
    side_lengths=[0.15, 0.15, 0.28], center=[0, 0, 0.14]
)
box2 = box1.transform(translation=[0, 0, 0.28])
bottle = rg.Cylinder(radius=0.0375, length=0.2)

base_board_params = base_board.uniform_density_params(BASE_BOARD_MASS)
box1_params = box1.hollow_density_params(BOX1_MASS)
box2_params = box2.hollow_density_params(BOX2_MASS)
bottle_params = bottle.uniform_density_params(BOTTLE_MASS)

bottle_x = box1.half_extents[0] - bottle.radius
bottle_y = -bottle_x
arr1_params = (
    base_board_params
    + box1_params
    + bottle_params.transform(translation=[bottle_x, bottle_y, 0.1])
)
arr2_params = (
    base_board_params
    + box1_params
    + box2_params
    + bottle_params.transform(translation=[bottle_x, bottle_y, 0.1 + 0.28])
)

com_offset1 = arr1_params.com - box1.center
com_offset2 = arr2_params.com - 0.5 * (box1.center + box2.center)

IPython.embed()
