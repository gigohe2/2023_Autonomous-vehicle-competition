#!/usr/bin/env python
# -*- coding: utf-8 -*-

# morai_gps_drive_shark.py

import numpy as np
from tf.transformations import euler_from_quaternion
import csv
import pandas as pd
import math
import matplotlib.pyplot as plt

import rospy
import time

from morai_msgs.msg import CtrlCmd, Lamps
from morai_msgs.srv import MoraiEventCmdSrv
from morai_msgs.msg import EgoVehicleStatus

from std_msgs.msg import Int32
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

from util import Carstatus
from util import PID
from morai_sensor import MoraiOdom
from morai_sensor import MoraiCamera
import os

username = os.environ.get('USER') or os.environ.get('USERNAME')

class ShadowDriver():
    GEAR_P = 1
    GEAR_R = 2
    GEAR_N = 3
    GEAR_D = 4
    def __init__(self) -> None:

        self.my_car = Carstatus()
        self.ctrl_cmd = CtrlCmd()
        self.odom = MoraiOdom()

        self.min_idx_pub = rospy.Publisher('min_idx', Int32, queue_size=1)    
        self.pub_ctrl = rospy.Publisher('ctrl_cmd', CtrlCmd, queue_size=1)
        self.srv_event_cmd = rospy.ServiceProxy('Service_MoraiEventCmd', MoraiEventCmdSrv)       
        self.mission_err_sub = rospy.Subscriber('/lidar_error', Float32, self.mission_err_cb)    
        self.parking_sub = rospy.Subscriber('/parking/location', Float32, self.parking_loc_cb)
        self.is_parking_sub = rospy.Subscriber('/parking/is', Int32, self.is_parking_cb)
        self.front_dist_sub = rospy.Subscriber('/front_dist', Float32, self.front_dist_cb)
        rospy.wait_for_message('/odom',Odometry)        
        self.sub_ego = rospy.Subscriber('/Ego_topic', EgoVehicleStatus, self.status_callback)
        rospy.wait_for_message('/Ego_topic', EgoVehicleStatus)
        rospy.loginfo('sub_/Ego_topic')
        self.manual_steering, self.manual_accel, self.manual_brake = 0, 0.5, 0
        self.event_cmd_srv = MoraiEventCmdSrv()
        self.event_cmd_srv.ctrl_mode=3
        self.event_cmd_srv.gear=-1
        self.event_cmd_srv.lamps = Lamps()
        self.ctrl_cmd.longlCmdType = 1
        self.event_cmd_srv.set_pause=False
        self.gear = 1               
        self.target_speed = 30
        self.accel = 0 
        self.second_run_count = 0
        self.front_dist = 15
        self.third_pose_xy = [0, 0]
        

        ## 경로파일 설정 후 사용!
               
        self.pid_steer = PID(0.5,0,0,1/50,1)
        self.shadow_count = 0
        self.mission_err = 0
        self.is_shadow = 0
        self.parking_flag = 0
        self.parking_loc = 0
        self.is_parking = 0
        self.shadow_speed = 2
        self.rate = rospy.Rate(50)
        
    def set_gear(self, gear):
        while abs(self.my_car.v > 1):
            continue

        self.event_cmd_srv.option = 2
        self.event_cmd_srv.gear = gear
        _ = self.srv_event_cmd(self.event_cmd_srv)

    def status_callback(self, data):
        self.my_car.v = data.velocity.x

    def mission_err_cb(self, msg):
        self.mission_err = msg.data
        
    def parking_loc_cb(self, msg):
        self.parking_loc = msg.data
        
    def is_parking_cb(self, msg):
        self.is_parking = msg.data

    def front_dist_cb(self, msg):
        self.front_dist = msg.data
        
    ## clear   
    def drive_pub(self, angle, speed: float):
        accel, brake = self.speed_control(speed)
        self.ctrl_cmd.steering = angle
        self.ctrl_cmd.accel = accel
        self.ctrl_cmd.brake = brake
        self.pub_ctrl.publish(self.ctrl_cmd)

    def accel_pub(self, angle, accel, brake):
        self.ctrl_cmd.steering = angle
        self.ctrl_cmd.accel = accel
        self.ctrl_cmd.brake = brake
        self.pub_ctrl.publish(self.ctrl_cmd)
    ## clear
    def speed_control(self, speed:float) -> tuple:
        """speed to [accel, brake]

        Args:
            speed (float): 목표 속도

        Returns:
            [float, float]: accel, manual_brake
        """
        speed_error = speed - self.my_car.v
        if speed_error > 0:
            accel = PID(1, 0.15, 0, 1/50, 1-self.manual_steering).do(speed_error)
            manual_brake = 0
        else:
            accel = 0
            manual_brake = PID(0.1, 0.1, 0, 1/50, 0.5).do(speed_error)
            
        return accel, manual_brake

    ## clear
    def first_run(self, my_car:Carstatus):
        """첫번재 음영구간 꼬깔콘 주행

        Args:
            my_car (Carstatus): 

        Returns:
            _type_: _description_
        """        
        angle = -self.mission_err / 12
        accel, brake = self.speed_control(4)
        #rospy.loginfo(f'angle:{angle}, accel:{accel}')
        return [angle, accel, brake]
        
    def second_run(self, my_car):
        """_summary_
            auto parking and left sided car tracking   
        Args:
            my_car (_type_): _description_
        """
        self.my_car = my_car
        # rospy.loginfo("enter 2nd shadow zone")
        
        if self.second_run_count == 0:
            while True:
                self.accel = 0
                self.drive_pub(0, 2)
                if self.front_dist < 9:
                    break
                rospy.sleep(0.02)
                # rospy.loginfo(self.front_dist)
            while True:
                # rospy.loginfo("left")
                self.accel = 0
                self.manual_brake = 0
                self.drive_pub(1, 2)
                rospy.sleep(0.02)
                if (np.rad2deg(self.my_car.heading) >-92 and np.rad2deg(self.my_car.heading) < -88):
                    self.second_run_count += 1
                    break
             
        
        
        self.drive_pub(self.mission_err / 12, self.shadow_speed)
        
        if self.parking_flag == 0 and self.is_parking == 1:
            # print("auto parking")
            max_loc = self.parking_loc - 1.5

            while self.parking_loc >= 1:
                angle = self.mission_err / 12
                speed = 2 * ((self.parking_loc-1.5) / max_loc)
                # print(self.parking_loc, max_loc, speed)
                
                self.drive_pub(angle, speed)
                rospy.sleep(0.02)
            
            self.park_car()
            self.shadow_speed += 1
            self.parking_flag += 1
    
    # tracking between car
    def third_run(self, my_car:Carstatus)->list:
        """차 사이 주행

        Args:
            my_car (Carstatus): _description_

        Returns:
            list: [angle, accel, brake]
        """
        self.my_car = my_car
        heading = np.pi + self.my_car.heading
        #print(np.rad2deg(heading))
        self.third_pose_xy[0] += self.my_car.v * np.cos(heading) / 50
        self.third_pose_xy[1] += self.my_car.v * np.sin(heading) / 50

        # print(self.third_pose_xy, self.mission_err==0)
        
        if self.third_pose_xy[1] > 45 and self.mission_err == 0:
            if (np.rad2deg(self.my_car.heading) < -175) or (np.rad2deg(self.my_car.heading) > 175):
                angle = 0
            else:
                angle = 0.3
           

            accel, brake = self.speed_control(5)
            # rospy.loginfo(angle)
            return [angle, accel, brake] 
        
        else:
            angle = self.mission_err / 20
            accel, brake = self.speed_control(5)

            return [angle, accel, brake]
        
 
    def park_car(self):
        time.sleep(0.1)
        self.stop(1)
        

        for i in range(130):
            self.drive_pub(1, 1)
            rospy.sleep(0.02)

        self.stop(1)
        self.set_gear(self.GEAR_R)

        while True:
            deg = np.rad2deg(self.my_car.heading)
            # print(deg)
            if (deg>179 and deg < 180) or (deg <-179 and deg > -180):
                self.backward()
                break
                
            else:
                self.accel_pub(-2, 0.1, 0)

    def backward(self):
        self.stop(1)
        
        for i in range(55):
            self.accel_pub(0, 0.1, 0)
            rospy.sleep(0.1)
        self.stop(1)
        self.set_gear(self.GEAR_P)
        rospy.sleep(3)
        self.set_gear(self.GEAR_D)
        for i in range(120):
            self.drive_pub(0, 1)
            rospy.sleep(0.02)
        for i in range(200):
            self.drive_pub(-2, 2)
            rospy.sleep(0.02)
            deg = np.rad2deg(self.my_car.heading)
            if (deg>-92) and (deg<-88):
                break
         
    def stop(self, time):
        for i in range(time*50):

            self.ctrl_cmd.steering=0.0
            self.ctrl_cmd.accel = 0
            self.ctrl_cmd.brake = 1
            self.pub_ctrl.publish(self.ctrl_cmd)
            rospy.sleep(0.02)

   

