#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import open3d as o3d
from open3d_ros_helper import open3d_ros_helper as orh

import rospy
import ros_numpy

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int32, Bool
from lidar_obs_detect.msg import obstacle_msg

from tf.transformations import euler_from_quaternion
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
from sympy import Symbol, solve
matplotlib.use('Agg')


class LidarProcessor():
    def __init__(self):
        self.shadow_count_sub = rospy.Subscriber('/shadow_count', Int32, self.shadow_count_cb)
        self.is_shadow_sub    = rospy.Subscriber('/is_shadow', Bool, self.is_shadow_cb)
        self.pc_sub           = rospy.Subscriber('/grd_removed_points', PointCloud2, self.pc_callback)
        rospy.wait_for_message('/grd_removed_points', PointCloud2) 
         
        self.mission_err_pub  = rospy.Publisher('/lidar_error', Float32, queue_size=1)
        self.is_obs_pub       = rospy.Publisher('/is_obs', Int32, queue_size=1)
        self.parking_pub      = rospy.Publisher('/parking/location', Float32, queue_size=1)
        self.is_parking_pub   = rospy.Publisher('/parking/is', Int32, queue_size=1)
        self.front_dist_pub   = rospy.Publisher('/front_dist', Float32, queue_size=1)
        self.shadow_counter   = 0
        self.is_shadow = False
        self.removed_xy = None
        
    def shadow_count_cb(self, msg):
        self.shadow_counter = msg.data
    
    def is_shadow_cb(self, msg):
        self.is_shadow = msg.data
        
    def pc_callback(self, msg):
        pc_3d = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        self.removed_xy = pc_3d[:, :2]
     
    def first_shadow_zone(self):
        """첫번째 음역구간
        바리게이트 진입후 첫번째 음영구역
        꼬깔콘 주행
        """
        ROI_start_y = 5
        ROI_end_y = 15
        try:
            x_left = self.removed_xy[(self.removed_xy[:,1]>0) & (self.removed_xy[:,1]<5) & (self.removed_xy[:,0]>ROI_start_y) & (self.removed_xy[:,0]<ROI_end_y)]
            x_left_min = np.min(x_left[:,1])
            #ro(f'x_left_min:{x_left_min:0.2f}')
            # x_left_min < 2 : steer to right
            # x_left_min > 2 : steer to left
            left_error = 2 - x_left_min
            self.mission_err_pub.publish(left_error)
             
        except:
            left_error = -0.1
            self.mission_err_pub.publish(left_error)
            #ro("can't find left sided larva cone")
           
    def get_parking_area(self):
        """주차 구역 탐색
        """
        ROI_start_x, ROI_end_x = 2.5, 4.5
        ROI_start_y, ROI_end_y = -4, 15

        PARKING_AREA_MIN_WIDTH = 1.5

        # Recognize parking area
        try:
            right_side = self.removed_xy[(self.removed_xy[:,1]<-ROI_start_x) & (self.removed_xy[:,1]>-ROI_end_x) & (self.removed_xy[:,0]>ROI_start_y) & (self.removed_xy[:,0]<ROI_end_y)]
            right_side_y = np.sort(right_side[:,0])[::-1] # < 0
            
            right_side_y_with_gap = [(right_side_y[i], right_side_y[i] - right_side_y[i+1]) for i in range(len(right_side_y)-1)]
            filtered_gaps = [(start, gap) for start, gap in right_side_y_with_gap if gap > PARKING_AREA_MIN_WIDTH]
            filtered_gaps.sort(key=lambda x: x[0])  

            if filtered_gaps:
                gap_start_right, gap = filtered_gaps[0]
            
            if gap_start_right > 0 and gap > PARKING_AREA_MIN_WIDTH:
                #ro('Found Parking area on right side, gap : {0}, location : {1}'.format(gap, gap_start_right))
                self.parking_pub.publish(gap_start_right)
                self.is_parking_pub.publish(1)
                                       
        except:
            self.is_parking_pub.publish(0)
            #ro("can't find point in ROI")
        
    def get_front_dist(self):
        try:
            pc_y = self.removed_xy[(self.removed_xy[:,1]<2) & (self.removed_xy[:,1]>-2) & (self.removed_xy[:,0]>1) & (self.removed_xy[:,0]<15)]
            y_min = np.min(pc_y[:,0])
           
            self.front_dist_pub.publish(y_min)
        except:
            rospy.loginfo("none")

    def get_center_betcar(self, zone_number):
        # hyper parameter
        ROI_start_y = 2
        ROI_end_y   = 6
        
        # removed_xy[1] : x ,  removed_xy[0] : y
        if zone_number == 3:
                
            try:
                x_left = self.removed_xy[(self.removed_xy[:,1]>0) & (self.removed_xy[:,1]<8) & (self.removed_xy[:,0]>2) & (self.removed_xy[:,0]<ROI_end_y)]
                x_right = self.removed_xy[(self.removed_xy[:,1]<0) & (self.removed_xy[:,1]>-8) & (self.removed_xy[:,0]>2) & (self.removed_xy[:,0]<ROI_end_y)]
                #print(len(x_left), len(x_right))
                if len(x_right) != 0 and len(x_left) != 0:
                        
                    x_left_min = np.min(x_left[:,1])
                    x_right_min = np.max(x_right[:,1])
                    center_error = (x_left_min+x_right_min)
                elif len(x_left) == 0:
                    x_right_min = np.max(x_right[:,1])
                    center_error = (x_right_min + 1.9)
                    #print("right", x_right_min, center_error)
                elif len(x_right) == 0:
                    x_left_min = np.min(x_left[:,1])
                    center_error = x_left_min - 1.9
                    #print("left", x_left_min, center_error)
                    
        
                    
            except:
                ##ro("can't find larva cone center")
                center_error = 0
                
                    

        elif zone_number == 2:
            try:
                x_right = self.removed_xy[(self.removed_xy[:,1]<0) & (self.removed_xy[:,0]>ROI_start_y) & (self.removed_xy[:,0]<ROI_end_y)]
                x_right_min = np.max(x_right[:,1])
                center_error = x_right_min + 2.6
                # print(center_error)
            except:
                center_error = 0
                # #ro("can't find larva cone center")
                
        #print(center_error)
        
        self.mission_err_pub.publish(center_error)
    
    def obstacle_avoid(self):
        """정적 장애물 회피
        """
        ROI_width = 4.5
        ROI_height = 15
        obs_steer = 0
        obs_flag = 0
        try:
            ROI_xy = self.removed_xy[(self.removed_xy[:,1]<ROI_width/2) & (self.removed_xy[:,1]>-ROI_width/2) & (self.removed_xy[:,0]>2) & (self.removed_xy[:,0]<ROI_height)]
    
            
            if len(ROI_xy) != 0:
                # #ro("obs")
                obs_flag = 1
                left = ROI_xy[(ROI_xy[:,1]>0)]
                right = ROI_xy[(ROI_xy[:,1]<0)]
                #print(len(left), len(right))
                if len(left) != 0 and len(right) == 0:
                    obs_left_closest = np.min(left[:,1])
                    obs_right_closest = -10
                elif len(left) ==0 and len(right) !=0:
                    obs_right_closest = np.max(right[:,1])
                    obs_left_closest = 10
                else:
                    obs_left_closest = np.min(left[:,1])
                    obs_right_closest = np.max(right[:,1])
                # print(obs_left_closest, obs_right_closest)
                if np.abs(obs_left_closest) < np.abs(obs_right_closest):
                    obs_steer = (2.5-obs_left_closest)
                else:
                    obs_steer = (-obs_right_closest-2.5)
                
        except:
            #ro("none") 
            obs_flag = 0
        self.is_obs_pub.publish(obs_flag)
        self.mission_err_pub.publish((obs_steer/7))
        ##ro(obs_steer)
    

if __name__ == '__main__':
    rospy.init_node("lidar_processing")
    lidar_processor = LidarProcessor()
    #ro("start lidar post processing")

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        #print("count:",lidar_processor.shadow_counter)
        if not lidar_processor.is_shadow:
            lidar_processor.obstacle_avoid()
        else:
            if lidar_processor.shadow_counter == 1:
                lidar_processor.first_shadow_zone()    
            elif lidar_processor.shadow_counter == 2:
                lidar_processor.get_center_betcar(2)
                lidar_processor.get_parking_area()
                lidar_processor.get_front_dist()
            elif lidar_processor.shadow_counter == 3:
                lidar_processor.get_center_betcar(3)
        
        #lidar_processor.get_parking_area()
        rate.sleep()
                
