#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/root/catkin_ws/src/shark/src')
import cv2
import time
import rospy
import ros_numpy
import numpy as np
from util import PID

from sensor_msgs.msg import PointCloud2
from lidar_obs_detect.msg import obstacle_msg
from morai_msgs.msg import CtrlCmd, Lamps
from morai_msgs.msg import EgoVehicleStatus

from sklearn.cluster import DBSCAN


class AccDrive():
    def __init__(self):
        self.rate = 50
        
        self.xy = None
        self.dbscan = DBSCAN(eps = 0.3, min_samples=20)
        self.obs_x_list = list()
        self.obs_y_list = list()
        self.obs_width = list()
        self.obs_height = list()
        self.obs_ylim_list = list()
        self.obs_ymin_list = list()
        self.width = 512
        self.height = 512
        self.scale = -25
        self.v =0
        self.img =np.zeros((self.width,self.height))
        
        self.target_distance = 7
        self.obs_dist = 0
        self.dist_pid = PID(Kp=0.005, Ki=0.0, Kd=0.0, dt=1/self.rate)

        self.ctrl_cmd = CtrlCmd()
        self.ctrl_cmd.longlCmdType = 1
        self.ctrl_cmd.accel = 0
        self.sub_ego = rospy.Subscriber('/Ego_topic', EgoVehicleStatus, self.status_callback)
        self.rm_points = rospy.Subscriber('/grd_removed_points', PointCloud2, self.callback)
        rospy.wait_for_message('/grd_removed_points', PointCloud2)
        # self.obs_msg = rospy.Subscriber('/obstacle', obstacle_msg, self.obs_callback)
        # rospy.wait_for_message('/obstacle', PointCloud2)
        
        self.pub_ctrl = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
    
    def callback(self, msg):
        pc_arr = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        self.xy = pc_arr[:,:2]
        
        xy = self.xy[ (self.xy[:,1]<2) & (self.xy[:,1]>-2) & (self.xy[:,0]>1) & (self.xy[:,0]<15)]
        if len(xy) != 0:
            self.obs_dist = np.min( xy[:, 0])
        
    def status_callback(self, data):
        self.v = data.velocity.x
        
    def speed_control(self, speed:float) -> tuple:
        """speed to [accel, brake]

        Args:
            speed (float): 목표 속도

        Returns:
            [float, float]: accel, manual_brake
        """
        speed_error = speed - self.v
        if speed_error > 0:
            accel = PID(1, 0.15, 0, 1/50, 1).do(speed_error)
            manual_brake = 0
        else:
            accel = 0
            manual_brake = PID(0.1, 0.1, 0, 1/50, 0.5).do(speed_error)
            
        return accel, manual_brake
    
    def run(self):
        
        dist_err =  self.obs_dist - self.target_distance
        u = self.dist_pid.do(dist_err)
        u = 0.5 if u > 0.5 else -0.5 if u < -0.5 else u
        a, b =self.speed_control(2)
        if u > 0:
            self.ctrl_cmd.accel = u + a
            self.ctrl_cmd.brake = 0
        else:
            self.ctrl_cmd.accel = 0
            self.ctrl_cmd.brake = -u + b

        
        return u, b
        

        
if __name__ == '__main__':
    rospy.init_node('Accdriving')
    
    acc= AccDrive()
    
    rate = rospy.Rate(acc.rate)
    while not rospy.is_shutdown():
        acc.run()
        # acc.showplot()

        rate.sleep()