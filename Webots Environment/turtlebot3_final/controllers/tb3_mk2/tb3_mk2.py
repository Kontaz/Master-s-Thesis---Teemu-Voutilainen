import gymnasium as gym
import math
import numpy as np
import os
import sys
import time
from controller import Supervisor
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class Turtlebot3_RL(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()
        
        # observation and action space initialization
        self.__n_lidar_sectors = 12
        self.__action_history_size = 30
        
        self.action_space = gym.spaces.Discrete(4) # forward, backward, turn right, and turn left. Optionally stop, slow, continue
        self.observation_space = gym.spaces.Box(low=-1000, high=1000,
                                                shape=(self.__n_lidar_sectors+self.__action_history_size,),
                                                dtype=np.float32)
        self.observation = None
        
        # variable initialization for robot parts
        self.__wheels = None
        self.__lidar = None
        self.__lidar_main_motor = None
        self.__lidar_secondary_motor = None
        self.__max_speed = 6.67 #max speed
        self.__left_speed = None
        self.__right_speed = None
        
        # other variables
        self.__timestep = int(self.getBasicTimeStep())
        self.__nresets = 1
        self.__tb = super().getFromDef('TB3')
        self.__start_loc = self.__tb.getField('translation')
        self.__start_dir = self.__tb.getField('rotation')
        
    def reset(self, seed=None):
    
        # simulation reset
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)
        self.t_steps = 1 #keeps count on the number of time steps
        self.max_run_length = 5000
        self.action_history = list(np.array(np.ones(self.__action_history_size)*-1, dtype=int))
        self.__start_loc.setSFVec3f([np.random.uniform(-0.8,0.8),np.random.uniform(-0.8,0.8),0]) #random starting location
        self.__start_dir.setSFRotation([0, 0, 1, np.random.uniform(0,np.pi*2)])
        self.__left_speed = 0
        self.__right_speed = 0
        
        # activating wheels
        self.__wheels = []
        for name in ['left wheel motor', 'right wheel motor']:
            wheel = self.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
            self.__wheels.append(wheel)
            
        # activating lidar
        self.__lidar = self.getDevice('LDS-01')
        self.__lidar_main_motor = self.getDevice('LDS-01_main_motor')
        self.__lidar_secondary_motor = self.getDevice('LDS-01_secondary_motor')
        self.__lidar.enable(self.__timestep)
        self.__lidar.enablePointCloud()
        self.__lidar_main_motor.setPosition(float('inf'))
        self.__lidar_secondary_motor.setPosition(float('inf'))
        self.__lidar_main_motor.setVelocity(30.0)
        self.__lidar_secondary_motor.setVelocity(60.0)
        
        # initial scan for observation
        super().step(self.__timestep)
        lidar_points = self.__lidar.getPointCloud()
        #super().step(self.__timestep) # THIS MIGHT NOT BE NEEDED
        
        degree = 360/self.__n_lidar_sectors
        lidar_scan = []
        for i in range(self.__n_lidar_sectors):
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
            
        self.observation = np.array(lidar_scan + self.action_history)
        info = {}
        
        return self.observation.astype(np.float32), info

    def step(self, action):
        self.action_history = [action] + self.action_history[:-1]
        
        # setting motor speed based on action
        if action == 0: # forward
            self.__left_speed = self.__max_speed
            self.__right_speed = self.__max_speed
        elif action == 1: # backward
            self.__left_speed = -self.__max_speed
            self.__right_speed = -self.__max_speed
        elif action == 2: # left
            self.__left_speed = -self.__max_speed
            self.__right_speed = self.__max_speed
        elif action == 3: # right
            self.__left_speed = self.__max_speed
            self.__right_speed = -self.__max_speed
        #elif action == 4: #stop
        #    self.__left_speed = 0
        #    self.__right_speed = 0
        #elif action == 5: #continue
        #    pass

        self.__wheels[0].setVelocity(self.__left_speed)
        self.__wheels[1].setVelocity(self.__right_speed)
        
        # new scan for observation
        super().step(self.__timestep)
        lidar_points = self.__lidar.getPointCloud()
        
        degree = 360/self.__n_lidar_sectors
        lidar_scan = []
        for i in range(self.__n_lidar_sectors):
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
            
        self.observation = np.array(lidar_scan + self.action_history)
        
        # calculating reward
        if self.t_steps>self.max_run_length: # ran too long
            done = True
            reward = 0 # possible negative reward for too long run
        #elif self.action_history.count(4)>=30: # option to stop early at the goal
            #done = True
            #reward = (self.max_run_length-self.t_steps)*(1 - ((max(lidar_scan) - min(lidar_scan))/2))# - 5000/(0.01 * np.exp(0.01 * self.t_steps) + 1) + 1000
        #elif min(lidar_scan)<0.2: # too close to wall
            #done = False
            #reward = min(lidar_scan)-0.2
        else:
            done = False
            #reward = 1 - ((max(lidar_scan) - min(lidar_scan))/2)
            #for 2.08m arena dividing by 2 should about normalize values to 0-1
            reward = 0
            half = int(self.__n_lidar_sectors/2)
            
            for i in range(half):
                num = lidar_scan[i]
                opp = lidar_scan[i+half]
                normalized_distance = abs(num-opp)/2.08
                reward += (1 - normalized_distance) * (1/half)
            
        info = {}
        truncated = False
        self.t_steps += 1
        
        return self.observation.astype(np.float32), reward, done, truncated, info

def main():
    ID = "MK2_PPO_test1"
    t = int(time.time())
    parent_dir = "F:\Webots\TB3_FINAL_LOGS_AND_MODELS_MK2"
    models_dir = parent_dir + "\\" + f"models\{ID}.{t}"
    log_dir = parent_dir + "\\" + f"logs\{ID}.{t}"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    env = Turtlebot3_RL()
    check_env(env)
    env.reset()
    print("Environment check successful")
    
    # n_steps=256 starts to seem very stuttery with current computer setup, try 32, 64, 128 if this happens.
    model = PPO('MlpPolicy', env, n_steps=128, verbose=1, tensorboard_log=log_dir)
    
    TIMESTEPS = 100000 # webots crashes with values nearing 10m
    for i in range(1,1000):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS*i}")
    
if __name__ == '__main__':
    main()