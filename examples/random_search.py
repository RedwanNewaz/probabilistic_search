import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import random
import numpy as np
GRID_SIZE = 50
FILTER_SIZE = 5
NUM_OBJECTS = 10

# create random objects in the grid
objects = []
for i in range(NUM_OBJECTS):
    x = random.randint(0, GRID_SIZE - FILTER_SIZE - 1)
    y = random.randint(0, GRID_SIZE - FILTER_SIZE - 1)
    objects.append(Circle((x, y), radius=1, color='red'))



# draw  rectangle for grid
grid_rect = Rectangle((0,0), GRID_SIZE, GRID_SIZE, fill = False, edgecolor = 'red')
filter_rect = Rectangle((1,1), FILTER_SIZE, FILTER_SIZE, fill = False, edgecolor = 'blue')
fig, ax = plt.subplots()
ax.add_patch(grid_rect)
for obj in objects:
    ax.add_patch(obj)


robot = ax.add_patch(filter_rect)
ax.set_xlim(-1, GRID_SIZE + 1)
ax.set_ylim(-1, GRID_SIZE + 1)
plt.axis('equal')
# change robot coordinate (1,1)
def show_animation(robot, x, y):
    robot.set_xy((x, y))
    plt.pause(0.001)


def isInsideGrid(x, y):
    # use AABB collision detection to check if the robot is inside the grid
    return (x >= 0 and x + FILTER_SIZE <= GRID_SIZE) and (y >= 0 and y + FILTER_SIZE <= GRID_SIZE)

def get_available_controls(robot):
    x, y = robot.get_xy()
    controls = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    available_controls = []
    for control in controls:
        new_x = x + control[0]
        new_y = y + control[1]
        if isInsideGrid(new_x, new_y):
            available_controls.append(control)
    return available_controls

def sample_control(robot):
    available_controls = get_available_controls(robot)
    if available_controls:
        return random.choice(available_controls)
    return None

def move_randomly(robot):
    # get current position of the robot
    x, y = robot.get_xy()
    # choose available control by checking the boundary of the grid
    u = sample_control(robot)
    while u:
        new_x = x + u[0]
        new_y = y + u[1]

        if not isInsideGrid(new_x, new_y):
            u = sample_control(robot)
        else:
            show_animation(robot, new_x, new_y)
            x, y = new_x, new_y


move_randomly(robot)

plt.show()

