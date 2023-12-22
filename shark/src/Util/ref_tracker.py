import numpy as np
import math
import csv
import os
import rospkg
from Kalman import KalmanPos2Vel
import rospy

class Gain():
    def __init__(self, Kp=1, Kd=0, Kpa=0.1):
        self.Kp = Kp
        self.Kd = Kd
        self.Kpa = Kpa

    def get(self):
        return self.Kp, self.Kd, self.Kpa
    
class velocity_range():
    start_idx = 0
    end_idx = 0
    velocity = 0

class RefTracker():
    def __init__(self, gain_lowspeed, gain_highspeed, look_ahead_dist=0.7, dt=1/50):
        self.dt = dt
        self.look_ahead_dist = look_ahead_dist

        self.kf_cte_vel = KalmanPos2Vel(P0 = np.array([[1,0],[0,1]]), x0=np.array([[0],[0]]))

        self.gain_lowspeed = gain_lowspeed
        self.gain_highspeed = gain_highspeed

    def do(self, ego):
        if ego.x == 0:
            return 0, 5, 0
        ax, ay = self.calc_ahead_point(ego)
        idx,theta = self.calc_nearest_point(ax, ay)
        theta = 2*np.pi + theta if theta < 0 else theta
        ref_x, ref_y = self.ref_path[idx][0], self.ref_path[idx][1]
       
        dx, dy = ego.x - ref_x , ego.y - ref_y
        cte = - np.dot([dx, dy], [np.cos(ego.heading + np.pi/2), np.sin(ego.heading + np.pi/2)])
        d_cte,_ = self.kf_cte_vel.update(cte, self.dt)

        velocity = self.get_velocity(idx)

        Kp, Kd, Kpa = self.gain_lowspeed.get() if ego.velocity < 10 else self.gain_highspeed.get()
        steering = Kp*cte + Kd*d_cte + Kpa*(theta-ego.heading)
        steering = 1 if steering > 1 else -1 if steering <-1 else steering


        #rospy.loginfo(f'idx: {idx}, vel: {velocity}({ego.velocity:0.1f}), head: {ego.heading:0.1f}, theta: {theta:0.1f}')
        return steering, velocity, idx

    def get_velocity(self, idx):
        velocity = 5
        for vp in self.velocity_profile:
            if vp.start_idx <= idx and idx <= vp.end_idx:
                velocity = vp.velocity
                break
        return velocity

    def calc_ahead_point(self, ego):
        dx = self.look_ahead_dist * np.cos(ego.heading)
        dy = self.look_ahead_dist * np.sin(ego.heading)

        ahead_x = ego.x + dx
        ahead_y = ego.y + dy

        return ahead_x, ahead_y
        
    def calc_nearest_point(self, ahead_x, ahead_y):

        distances = np.sum((self.ref_path - (ahead_x,ahead_y))**2, axis=1)
        closest_index = np.argmin(distances)

        if closest_index < len(self.ref_path) - 2:
            dx = self.ref_path[closest_index+2, 0] - self.ref_path[closest_index, 0]
            dy = self.ref_path[closest_index+2, 1] - self.ref_path[closest_index, 1]
        else:
            dx = self.ref_path[closest_index, 0] - self.ref_path[closest_index-2, 0]
            dy = self.ref_path[closest_index, 1] - self.ref_path[closest_index-2, 1]

        theta = np.arctan2(dy, dx)

        return closest_index, theta

    def set_ref_path(self, filename):
        rospack = rospkg.RosPack()
        filename = os.path.join(rospack.get_path('dku_morai_driver'), 
                                                 f'ref_path/{filename}.csv')        
        self.ref_path = []
        with open(filename, newline='') as csvfile:
            path_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in path_reader:
                self.ref_path.append([float(row[0]), float(row[1])])
        
        self.ref_path = np.array(self.ref_path)

    def set_velocity_profile(self, filename):
        rospack = rospkg.RosPack()
        filename = os.path.join(rospack.get_path('dku_morai_driver'), 
                                                 f'ref_path/{filename}.csv')
        self.velocity_profile = []
        with open(filename, newline='') as csvfile:
            path_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in path_reader:
                vr = velocity_range()
                vr.start_idx = float(row[0])
                vr.end_idx = float(row[1])
                vr.velocity = float(row[2])
                self.velocity_profile.append(vr)

