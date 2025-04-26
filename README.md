# === IMPORTS ===
from controller import Supervisor
import numpy as np
import math

# === INIT ===
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Devices
gps = robot.getDevice('gps')
compass = robot.getDevice('compass')
lidar = robot.getDevice('Hokuyo')  # Adjust if your lidar is named differently
display = robot.getDevice('display')

gps.enable(timestep)
compass.enable(timestep)
lidar.enable(timestep)
display_width = display.getWidth()
display_height = display.getHeight()

# Lidar params
lidar.enablePointCloud()
lidar.enableRangeImage()

# Map params
map_size = 300  # Example size, depends on environment dimensions
prob_map = np.full((map_size, map_size), 0.5)  # Probabilistic map (0.5 = unknown)

trajectory = []  # To store robot's path for display

# === CONTROL PARAMS ===
MAX_SPEED = 6.28  # [rad/s], check robot specs
k_rho = 5
k_alpha = 15

# Robot physical params
robot_radius_m = 0.3  # ~30 cm
world_size = 6.0  # World is 6m x 6m approx
meters_per_pixel = world_size / map_size

# Waypoints (X, Y) - fill these!!
waypoints = [
    [1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]
    # TODO: Put your full path waypoints around the table here
]
# Waypoints for second round
waypoints += waypoints[::-1]  # Reverse path for second circle
wp_index = 0

# Supervisor marker (optional to visualize current waypoint)
marker = robot.getFromDef('marker').getField('translation')

# Flags
mapping_finished = False
cspace_computed = False

# === HELPER FUNCTIONS ===

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def gps_to_map(x, y):
    # Map GPS (x,y) to pixel (i,j)
    i = int((x + world_size/2) / meters_per_pixel)
    j = int((y + world_size/2) / meters_per_pixel)
    return i, j

def display_robot_path():
    for (i, j) in trajectory:
        display.setColor(255, 0, 0)  # Red for robot path
        display.drawPixel(i, j)

def display_prob_map():
    for x in range(map_size):
        for y in range(map_size):
            intensity = int(prob_map[x, y] * 255)
            display.setColor(intensity, intensity, intensity)
            display.drawPixel(x, y)

# === MAIN LOOP ===
while robot.step(timestep) != -1:

    # === SENSOR READINGS ===
    gps_pos = gps.getValues()
    compass_values = compass.getValues()
    
    # Compute heading angle from compass
    heading = math.atan2(compass_values[0], compass_values[1])

    # === GOAL TRACKING ===
    if wp_index < len(waypoints):
        goal_x, goal_y = waypoints[wp_index]
        
        dx = goal_x - gps_pos[0]
        dy = goal_y - gps_pos[1]
        
        rho = math.sqrt(dx**2 + dy**2)
        goal_theta = math.atan2(dy, dx)
        alpha = normalize_angle(goal_theta - heading)

        # Simple proportional controller
        v = k_rho * rho
        w = k_alpha * alpha

        # Cap speeds
        v = min(v, 3.0)  # m/s
        w = max(min(w, 3.0), -3.0)

        # Differential drive conversion
        left_speed = v - w
        right_speed = v + w

        left_speed = np.clip(left_speed, -MAX_SPEED, MAX_SPEED)
        right_speed = np.clip(right_speed, -MAX_SPEED, MAX_SPEED)

        # Set motor speeds
        left_motor = robot.getDevice('wheel_left_joint')
        right_motor = robot.getDevice('wheel_right_joint')

        left_motor.setPosition(float('inf'))
        right_motor.setPosition(float('inf'))
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        # Waypoint reached?
        if rho < 0.3:
            wp_index += 1
            if wp_index < len(waypoints):
                marker.setSFVec3f([waypoints[wp_index][0], 0.0, waypoints[wp_index][1]])

    else:
        # Stop robot
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        mapping_finished = True

    # === UPDATE TRAJECTORY ===
    i, j = gps_to_map(gps_pos[0], gps_pos[1])
    trajectory.append((i, j))

    # === MAPPING ===
    ranges = lidar.getRangeImage()
    lidar_fov = lidar.getFov()
    lidar_resolution = lidar.getHorizontalResolution()

    angle_min = -lidar_fov / 2
    angle_increment = lidar_fov / lidar_resolution

    for idx, distance in enumerate(ranges):
        if idx < 80 or idx > lidar_resolution - 80:
            continue  # skip shielded areas

        if math.isinf(distance):
            distance = 100  # very far

        angle = angle_min + idx * angle_increment
        x_rel = distance * math.cos(angle)
        y_rel = distance * math.sin(angle)

        # Adjust for Lidar offset
        lidar_offset_x = 0.0
        lidar_offset_y = 0.0
        x_world = gps_pos[0] + lidar_offset_x + (x_rel * math.cos(heading) - y_rel * math.sin(heading))
        y_world = gps_pos[1] + lidar_offset_y + (x_rel * math.sin(heading) + y_rel * math.cos(heading))

        i, j = gps_to_map(x_world, y_world)
        if 0 <= i < map_size and 0 <= j < map_size:
            prob_map[i, j] = min(1.0, prob_map[i, j] + 0.01)  # Increase occupancy

    # === DISPLAY ===
    display_prob_map()
    display_robot_path()

    # === COMPUTE C-SPACE MAP ===
    if mapping_finished and not cspace_computed:
        from scipy.ndimage import binary_dilation

        cspace = (prob_map > 0.9).astype(np.uint8)
        # Grow obstacles based on robot size
        grow_kernel = np.ones((5, 5))  # Adjust size to fit your robot (5x5 pixels example)
        cspace = binary_dilation(cspace, structure=grow_kernel)

        # Optional: display C-space map
        display.setColor(0, 0, 0)
        display.fillRectangle(0, 0, display_width, display_height)
        for x in range(map_size):
            for y in range(map_size):
                if cspace[x, y]:
                    display.setColor(0, 0, 0)
                else:
                    display.setColor(255, 255, 255)
                display.drawPixel(x, y)

        cspace_computed = True
