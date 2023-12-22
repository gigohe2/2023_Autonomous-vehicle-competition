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
from std_msgs.msg import String, Bool
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
from shadow_drive import ShadowDriver
from morai_sensor import MoraiOdom
from morai_sensor import MoraiCamera
from yolo.yolo_run import TrafficLight
import os
from ACC_drive.acc_drive import AccDrive
from ref_tracker import RefTracker, Gain


username = os.environ.get('USER') or os.environ.get('USERNAME')

class GpsDriver():
    GEAR_P = 1
    GEAR_R = 2
    GEAR_N = 3
    GEAR_D = 4
    
    def __init__(self) -> None:

        self.my_car = Carstatus()
        self.ctrl_cmd = CtrlCmd()
        self.odom = MoraiOdom()

        self.shadow_counter = 0 
        self.manual_steering, self.manual_accel, self.manual_brake = 0, 0.5, 0
        self.theta = 0

        self.min_idx_pub = rospy.Publisher('min_idx', Int32, queue_size=1)
        self.pub_ctrl = rospy.Publisher('ctrl_cmd', CtrlCmd, queue_size=1)
        self.shadow_count_pub = rospy.Publisher('/shadow_count', Int32, queue_size=1)
        self.is_shadow_pub = rospy.Publisher('/is_shadow', Bool, queue_size=1)

        self.srv_event_cmd = rospy.ServiceProxy('Service_MoraiEventCmd', MoraiEventCmdSrv)
        
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self.car_status)
        rospy.wait_for_message('/odom', Odometry)
        rospy.loginfo('sub_/odom')
        
        self.sub_ego = rospy.Subscriber('/Ego_topic', EgoVehicleStatus, self.status_callback)
        rospy.wait_for_message('/Ego_topic', EgoVehicleStatus)
        rospy.loginfo('sub_/Ego_topic')
        
        self.sub_tf_cls=rospy.Subscriber('tf_cls',Float32MultiArray,self.tf_cls_callback)
        rospy.wait_for_message('tf_cls',Float32MultiArray)
        rospy.loginfo('sub_/tf_cls')
        
        self.is_obs_sub = rospy.Subscriber('/is_obs', Int32, self.is_obs_cb)
        rospy.wait_for_message('/is_obs', Int32)
        rospy.loginfo('sub_/is_obs')

        self.mission_err_sub = rospy.Subscriber('/lidar_error', Float32, self.mission_err_cb)
        rospy.wait_for_message('/lidar_error', Float32)
        rospy.loginfo('sub_/lidar_error')

        self.parking_sub = rospy.Subscriber('/parking/location', Float32, self.parking_loc_cb)
        # rospy.wait_for_message('/parking/location', Float32)
        rospy.loginfo('sub_/parking/location')

        self.is_parking_sub = rospy.Subscriber('/parking/is', Int32, self.is_parking_cb)
        # rospy.wait_for_message('/parking/is', Int32)
        rospy.loginfo('sub_/parking/is')
        
        self.person_sub = rospy.Subscriber('/person', String, self.person_cb)
        rospy.wait_for_message('/person', String)
        rospy.loginfo('sub_person')
        rospy.loginfo("success sub topics")


        
        
        
        self.event_cmd_srv = MoraiEventCmdSrv()
        self.event_cmd_srv.ctrl_mode=3
        self.event_cmd_srv.gear=-1
        self.event_cmd_srv.lamps = Lamps()
        self.ctrl_cmd.longlCmdType = 1
        self.event_cmd_srv.set_pause=False

        self.gear = 1
        
        self.obs_steer = Float64()
        self.my_car.v = 0
        self.target_speed = 30
        self.accel = 0 
        self.stanley_k = 5
        self.minimum_idx = 0 
        self.slope_heading = 0
        self.stanley_error = 0
        self.x_list, self.y_list, self.road, self.event = [], [], [], []
        
        self.ref_tracker = RefTracker(gain_lowspeed=Gain(Kp=0.25, Kd=0.01, Kpa=0.),
                                      gain_highspeed=Gain(Kp=0.1, Kd=0.01, Kpa=0.1),
                                      look_ahead_dist=2,
                                      dt=1/50)

        
        self.LOOK_AHEAD_DIST = 0.2
        self.pid_steer = PID(0.6,0,0,1/50,1)
        self.tf_cls=[]
        self.mission_err = 0
        self.prev_x, self.prev_y = 100, 100
        self.is_shadow = False
        self.loop_count = 0
        self.parking_flag = 0
        self.parking_loc = 0
        self.is_parking = 0
        self.shadow_speed = 2
        self.is_obs = 0
        self.person = String()
        
    def set_gear(self, gear):
        while abs(self.my_car.v > 1):
            continue

        self.event_cmd_srv.option = 2
        self.event_cmd_srv.gear = gear
        _ = self.srv_event_cmd(self.event_cmd_srv)

    def is_obs_cb(self, msg):
        self.is_obs = msg.data

    def mission_err_cb(self, msg):
        self.mission_err = msg.data
        
    def parking_loc_cb(self, msg):
        self.parking_loc = msg.data
        
    def is_parking_cb(self, msg):
        self.is_parking = msg.data

    def person_cb(self, msg):
        self.person = msg.data
        
    def car_status(self,msg):
        self.my_car.x, self.my_car.y = msg.pose.pose.position.x, msg.pose.pose.position.y
        #print(self.my_car.x ,self.my_car.y)
        if (self.my_car.x < 0 and self.my_car.y < 0):
            self.is_shadow = True
            if (self.prev_x != self.my_car.x and self.prev_y != self.my_car.y):
                self.shadow_counter += 1
                if self.shadow_counter == 1:
                    self.ref_tracker.set_ref_path('shark_gps_2')
                    self.ref_tracker.set_velocity_profile('velocity_profile_2')
                elif self.shadow_counter == 2:
                    self.ref_tracker.set_ref_path('shark_gps_3')
                    self.ref_tracker.set_velocity_profile('velocity_profile_3')
                    
        else:
            self.is_shadow = False
        self.is_shadow_pub.publish(self.is_shadow)
        self.shadow_count_pub.publish(self.shadow_counter)
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        euler_angles = euler_from_quaternion(quaternion)
        self.my_car.heading = self.nomarlize_pi(-euler_angles[2]+np.pi/2) # radian
        #print(np.rad2deg(self.my_car.heading))
        self.prev_x, self.prev_y = self.my_car.x, self.my_car.y

    def tf_cls_callback(self, msg):
        """0: Red, 1: Green, 5: Yellow
        """
        self.tf_cls = msg.data

    def status_callback(self, data):
        self.my_car.v = data.velocity.x

    def drive_pub(self, angle: float, accel: float, brake:float):

        self.ctrl_cmd.steering = angle
        self.ctrl_cmd.accel = accel
        self.ctrl_cmd.brake = brake
        self.pub_ctrl.publish(self.ctrl_cmd)

    def global_path(self, file_name):
        road_data = pd.read_csv(file_name)
        self.x_list = road_data['x'].values
        self.y_list = road_data['y'].values
        self.road = road_data['road'].values
        self.event = road_data['event'].values
        
        self.len_list = len(self.x_list) - 1
        rospy.loginfo("----------Path loaded----------")

    #----- Path tracking -------#
    def calc_ahead_point(self):
        dx = self.LOOK_AHEAD_DIST * np.cos(self.my_car.heading)
        dy = self.LOOK_AHEAD_DIST * np.sin(self.my_car.heading)

        ahead_x = self.my_car.x + dx
        ahead_y = self.my_car.y + dy

        return ahead_x, ahead_y

    def calc_nearest_point(self, ahead_x, ahead_y):
        self.minimum_idx = 0
        minimum_dist = 1e7
        for i, (rx, ry) in enumerate(zip(self.x_list, self.y_list)):
            dist = math.dist((ahead_x, ahead_y), (rx, ry))            
            if (dist < minimum_dist):
                minimum_dist = dist
                self.minimum_idx = i
        first = (self.x_list[self.minimum_idx], self.y_list[self.minimum_idx])

        self.min_idx_pub.publish(self.minimum_idx)

        return first
    #----- Path tracking -------#

    def nomarlize_pi(self, data):
        if data > np.pi:
            data = -2*np.pi + data
        elif data < -np.pi:
            data = 2*np.pi + data
        return data

    def stanley_control(self):
        self.target_x, self.target_y = self.calc_nearest_point(self.my_car.x, self.my_car.y)
        dx, dy = self.my_car.x - self.target_x , self.my_car.y - self.target_y
        self.Calc_slopeofpath()
        cte = - np.dot([dx, dy], [np.cos(self.my_car.heading + np.pi/2), np.sin(self.my_car.heading + np.pi/2)])
        cross_track_steering = np.arctan(self.stanley_k * cte / self.my_car.v + 1e-6)
        heading_error = self.nomarlize_pi(self.slope_heading - self.my_car.heading)
        self.stanley_error = self.nomarlize_pi(cross_track_steering + heading_error)
        steer = -self.pid_steer.do(self.stanley_error)  #if abs(cte) > 0.05 else 0
        #print("cte : {0}, heading_error: {1}, cts: {2}".format(cte, heading_error, cross_track_steering))
        return steer
    
    # Calculate slope of current path
    def Calc_slopeofpath(self):
        idx_1 = self.minimum_idx
        if (self.minimum_idx + 1) > self.len_list:
            idx_2 = 0
        else:
            idx_2 = self.minimum_idx + 1
        
        x_1, y_1 = self.x_list[idx_1], self.y_list[idx_1]
        x_2, y_2 = self.x_list[idx_2], self.y_list[idx_2]

        dx = x_1 - x_2
        dy = y_1 - y_2

        self.slope_heading = self.nomarlize_pi(- np.arctan2(dx , dy) - np.pi/2)

    # change speed control to Seoul
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

    def run(self, speed, show=False): 
        #print(self.is_obs)
    
        if self.is_obs == 1 and (self.minimum_idx>2300 and self.minimum_idx<2850):
            rospy.loginfo("obs steer")
            angle = self.mission_err/2
            accel, brake = self.speed_control(speed)
            
        elif self.minimum_idx > 7000:
            self.acc.detect_obs()
            self.accel, self.barake = self.acc.run()
            print("u :", self.accel)
        
            angle = self.stanley_control()
        else:
            angle = self.stanley_control()
            accel, brake = self.speed_control(speed)
        if show:
            self.update_plot()
        
        return [angle, accel, brake]

    def stop(self):
        self.ctrl_cmd.steering = 0.0
        self.ctrl_cmd.accel = 0.0
        self.ctrl_cmd.brake = 1
        sec = 2000  # ms

        return [0, 0, 1]
        
        for i in range(sec):
            self.ctrl_cmd.brake += 1/sec
            self.pub_ctrl.publish(self.ctrl_cmd)
            rospy.sleep(1/sec)


## 사용 안함
    def gps_run(self):
        flag = False
        Current_Idx=gps_drive.minimum_idx # gps => index
        event = gps_drive.event
        road_state = gps_drive.road
        camera=gps_drive.tf_cls
        camera = list(camera)
        
        
        # print(f"위치 : {Current_Idx}    카메라 : {camera}      도로 : {road_state[Current_Idx]}      이벤트 : {event[Current_Idx]}")

        if event[Current_Idx] == 'run':
            gps_drive.run(5, show=False)

        if (event[Current_Idx] == 'TrafficLight'):
            if((1.0 not in camera) and (0.0 in camera or 5.0 in camera)):
                print("stop")
                gps_drive.stop()
            if(1.0 in camera or 2.0 in camera):
                print("go")
                gps_drive.run(5, show=False)
                
        if(event[Current_Idx] == 'accel>=20'):
            gps_drive.run(5, show=False)
  

   
if __name__ == '__main__':
    rospy.init_node('morai_gps_drive')
    
    gps_drive = GpsDriver()
    print('start gps drive')
    
    shadow_drive = ShadowDriver()
    print('start shadow drive')
    
    acc_drive = AccDrive()
    
    

    gps_drive.ref_tracker.set_ref_path('shark_gps_1')
    gps_drive.ref_tracker.set_velocity_profile('velocity_profile_1')
    print('path loaded')
    
    rate = rospy.Rate(30)
    print('start drive node')
    
    aab = [0, 0, 0]

    while not rospy.is_shutdown():
        
        
        if (gps_drive.person == 'go' or gps_drive.person == 'slow') and not (0.0 in gps_drive.tf_cls or 5.0 in gps_drive.tf_cls):
            if not gps_drive.is_shadow:
                idx, t = gps_drive.ref_tracker.calc_nearest_point(gps_drive.my_car.x, gps_drive.my_car.y)
                rospy.loginfo(f'{gps_drive.tf_cls}, {idx}')
                
                # 추차와 차 사이 주행 사이 공간(잠시 GPS 음영 구간 X) 
                if gps_drive.shadow_counter == 2:
                    shadow_drive.third_run(gps_drive.my_car)

                # GPS 주행
                else:
                    rospy.loginfo(f"gps driving, shadow:{gps_drive.is_shadow}")
                    steer, speed, idx = gps_drive.ref_tracker.do(gps_drive.my_car)
                    
                    if gps_drive.shadow_counter == 3 and (idx<350 or (idx<1050 and idx>910)):
                        accel, brake = acc_drive.run()
                    if gps_drive.shadow_counter == 1 and (idx>200 and idx<530) and gps_drive.is_obs == 1:
                        steer = gps_drive.mission_err
                        accel, brake = gps_drive.speed_control(speed)
                    else:
                        accel, brake = gps_drive.speed_control(speed)
                        
                    # if ((idx<480 and idx>430) and (0.0 in gps_drive.tf_cls or (5.0 in gps_drive.tf_cls))):
                    #     print("stop")
                    #     accel, brake = 0, 0.5

                    aab = [-steer, accel, brake]
            # 음영 구역 주행
            else:
                # 첫번째 음영구간 왼쪽 꼬깔콘 주행
                if gps_drive.shadow_counter == 1:
                    #rospy.loginfo("enter 1st shadow zone")
                    aab = shadow_drive.first_run(gps_drive.my_car)
                    rospy.loginfo(aab)

                # 주차, 차 사이(꼬깔콘) 주행
                if gps_drive.shadow_counter == 2:
                    shadow_drive.second_run(gps_drive.my_car)
                    
                
                elif gps_drive.shadow_counter == 3:
                    aab = shadow_drive.third_run(gps_drive.my_car)
            
            # 사람이 시야에 들어오면 서행
            if gps_drive.person == 'slow':
                #steer, speed, idx = gps_drive.ref_tracker.do(gps_drive.my_car)
                accel, brake = gps_drive.speed_control(speed)
                aab[0] = 0
                aab[1] = accel
                aab[2] = brake
                
        
        # 사람이 보이거나 신호등이 빨간. 노란색 일때 멈춤
        elif gps_drive.person == 'stop':
            rospy.loginfo("stop")
            aab = gps_drive.stop()
        elif (0.0 in gps_drive.tf_cls or (5.0 in gps_drive.tf_cls)) and not gps_drive.is_shadow:
            rospy.loginfo("light stop")
            idx, t = gps_drive.ref_tracker.calc_nearest_point(gps_drive.my_car.x, gps_drive.my_car.y)
            if (idx<480 and idx>430) and gps_drive.shadow_counter ==0:
                aab = gps_drive.stop()
            elif (idx<1400 and idx>1350) and gps_drive.shadow_counter == 3:
                aab = gps_drive.stop()
            
        if gps_drive.shadow_counter != 2:
            gps_drive.drive_pub(aab[0],aab[1], aab[2])
        
        rate.sleep()