import rclpy
import numpy as np
import os
import time
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from stable_baselines3 import PPO
from stable_baselines3 import DQN

class TurtlebotController(Node):
    def __init__(self):
        super().__init__('turtlebot_controller')
        self.reward = 0
        self.tstep = 0
        self.subscription = self.create_subscription(LaserScan, '/scan', self.controller, qos_profile=rclpy.qos.qos_profile_sensor_data)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription # prevent unused variable warning
        self.max_steps = 500
        self.done = 0
        
        # reset temporary data file
        self.temp_file = 'temp.txt'
        with open(self.temp_file, 'w') as file:
            file.write("")

        # trained models
        PPO1 = 'PPO1_2300000.zip'
        PPO2 = 'PPO2_14900000.zip'
        PPO3 = 'PPO3_11000000.zip'
        DQN1 = 'DQN1_13700000.zip'
        DQN2 = 'DQN2_14200000.zip'
        DQN3 = 'DQN3_9300000.zip'
        self.active_model = DQN1 # swap active to use specific model
        # remember to also change setup name with np.savetxt
        
        self.run_path = 'RL_data'
        self.reward_data = np.zeros((self.max_steps,1))
        model_path = os.path.join(os.path.expanduser('~'), 'RL_models', self.active_model)
        #self.model = PPO.load(model_path) #PPO
        self.model = DQN.load(model_path) #DQN
        self.action_history = list(np.array(np.ones(30)*-1, dtype=int))
        self.reward_history = list(np.array(np.zeros(30)))
        
    def controller(self, msg):
        reward_message = 'timestep: ' + str(self.tstep) + '\treward: ' + str(np.round(self.reward,2)) + '\taverage reward: ' + str(np.round(np.average(self.reward_history),2))
        self.get_logger().info(reward_message)
        lidar = self.process_scan(msg.ranges)
        obs = np.array(lidar + self.action_history)
        r = self.calculate_reward(lidar)
        self.reward += r
        velocity = Twist()
        action, _ = self.model.predict(obs.astype(np.float32), deterministic=True)
        self.action_history = [action] + self.action_history[:-1]
        self.reward_history = [self.reward] + self.reward_history[:-1]
        
        if self.tstep < self.max_steps: #500
            self.reward_data[self.tstep] = r
            with open(self.temp_file, 'a') as file:
            	file.write(f"{r}\n")
            
            # in this turtlebot3 the motors seem to have been installed backwards
            if action == 0:
                velocity.linear.x = 0.036908
                velocity.angular.z = 0.0
            elif action == 1:
                velocity.linear.x = -0.036908
                velocity.angular.z = 0.0
            elif action == 2:
                velocity.linear.x = 0.0
                velocity.angular.z = 0.399132
            elif action == 3:
                velocity.linear.x = 0.0
                velocity.angular.z = -0.399132
        elif self.done == 0:
            velocity.linear.x = 0.0
            velocity.angular.z = 0.0
            np.savetxt((self.active_model[:4] + '.S1.REAL' + str(int(time.time())) + '.txt'), self.reward_data)
            self.done = 1
        else:
            # proper exit things can be added here
            time.sleep(300)
        
        self.tstep += 1
        self.publisher.publish(velocity)
        
    def process_scan(self, ranges):
        """
        For some reason the scanners in webots simulation and reality rotate
        in different directions and starts at different positions so 1) and 2)
        compensate for that. Because 10 hour simulations in webots are already
        done, this is easier way to correct.
        """
        ranges = ranges[::-1] # 1) turning the order to clockwise
        split_size = 12
        avg_n = 3
        section_size = len(ranges) / split_size
        scan_values = []
        
        for i in range(split_size):
            point = round(i*section_size)
            scan_values.append((ranges[point] + ranges[point+1] + ranges[point+2])/3)
        
        
        # 2) matching the start order to the simulation
        scan_values = scan_values[6:] + scan_values[:6]
        
        return scan_values
        
    def calculate_reward(self, lidar):
        n = len(lidar)
        half = int(n/2)
        reward = 0

        for i in range(half):
            num = lidar[i]
            opp = lidar[i+half]
            normalized_distance = abs(num-opp)/2.08
            reward += (1- normalized_distance) * (1/half)
        return reward

def main(args=None):
    rclpy.init(args=args)
    controller = TurtlebotController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
