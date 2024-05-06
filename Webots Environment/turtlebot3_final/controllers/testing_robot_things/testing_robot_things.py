from controller import Supervisor
import numpy as np
import time

def run_robot(robot):
    timestep = int(robot.getBasicTimeStep())
    max_speed = 6.67  # Adjust based on your robot's specifications
    
    bot = robot.getFromDef('TB3')
    r = bot.getField('rotation')
    l = bot.getField('translation')
    
    # Initialize motors
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))  # Set to infinity for velocity control
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    
    start_time = time.time()
    stopwatch = time.time() - start_time
    old_angle = 0
    rot_sum = 0
    init_loc = l.getSFVec3f()
    tstep = 0
    
    # activating lidar
    lidar = robot.getDevice('LDS-01')
    lidar_main_motor = robot.getDevice('LDS-01_main_motor')
    lidar_secondary_motor = robot.getDevice('LDS-01_secondary_motor')
    lidar.enable(timestep)
    lidar.enablePointCloud()
    lidar_main_motor.setPosition(float('inf'))
    lidar_secondary_motor.setPosition(float('inf'))
    lidar_main_motor.setVelocity(30.0)
    lidar_secondary_motor.setVelocity(60.0)
    
    n_lidar_sectors = 12
    
    while robot.step(timestep) != -1:
        lidar_points = lidar.getPointCloud()
        
        # converting the lidar values into usable observation data
        degree = 360/n_lidar_sectors
        lidar_scan = []
        for i in range(n_lidar_sectors):
            point = round(i*degree)
            # taking average of 3 to get sliglty better coverage and accuracy. Optionally could use all points.
            avg_n = 3
            avg_of_points = 0
            for j in range(avg_n):
                val = lidar_points[point-j]
                value = np.sqrt(val.x**2 + val.y**2)
                if value>1000:
                    value = 3.5 # after 3.5 meters the scan values go to infinite
                avg_of_points += value
            lidar_scan.append(avg_of_points/avg_n)
            
        print(np.round(lidar_scan,2))
    
    """
    while robot.step(timestep) != -1 and stopwatch<10:
        x = 3.11#6.67
        left_motor.setVelocity(x)
        right_motor.setVelocity(x)
        loc = l.getSFVec3f()
        distance_x = np.abs(init_loc[0]-loc[0])
        print('Start x:', np.round(init_loc[0],2), '\tCurrent x:', np.round(loc[0],2), '\tDistance:', np.round(distance_x,2))
        stopwatch = time.time() - start_time
        tstep += 1
    print(tstep)
    robot.simulationSetMode(robot.SIMULATION_MODE_PAUSE)


    
    while robot.step(timestep) != -1 and stopwatch<10:
        x = 1.75#6.67
        left_motor.setVelocity(-x)
        right_motor.setVelocity(x)
        rotation = r.getSFRotation()[3]
        rot_sum += np.abs(np.abs(rotation)-np.abs(old_angle))/(2*np.pi)
        stopwatch = time.time() - start_time
        print('Time:  ', np.round(stopwatch,2), '\tAngle:  ', np.round(rotation,2), '\tRotations:  ', np.round(rot_sum,2))
        old_angle = rotation
        tstep += 1
    print(tstep)
    robot.simulationSetMode(robot.SIMULATION_MODE_PAUSE)
    """
        
if __name__ == "__main__":
    my_robot = Supervisor()
    run_robot(my_robot)
