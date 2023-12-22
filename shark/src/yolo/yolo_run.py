#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/root/catkin_ws/src/shark/src')

import cv2, os
from ultralytics import YOLO
import numpy as np
import time

import rospy
from morai_sensor import MoraiCamera
from morai_msgs.msg import CtrlCmd
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String

# username = os.environ.get('USER') or os.environ.get('USERNAME')

class TrafficLight():
    def __init__(self) -> None:
        self.model = YOLO('/root/catkin_ws/src/shark/src/yolo/best.pt')  # model 입력
        self.cls_pub = rospy.Publisher('/tf_cls', Float32MultiArray, queue_size=1) # tf_cls publish
        self.result_img = None
        self.msg = Float32MultiArray()

    def run(self,img):
        """신호등 인식
            빨간:0, 직좌:2, 노란:5
        Args:
            img (_type_): _description_
        """
        self.results = self.model(img,verbose=False, conf= 0.55)
        self.result_img = self.results[0].plot()
        for r in self.results:
            size = r.boxes.xywh.cpu().numpy()
            if len(size) !=0:
                if size[0][0] > 170 and size[0][1]>20 and size[0][2]> 45:
                    self.msg.data = r.boxes.cls.cpu().numpy()
            else:
                self.msg.data = list()
                
            self.cls_pub.publish(self.msg)
    
class Person():
    def __init__(self) -> None:
        self.model = YOLO('/root/catkin_ws/src/shark/src/yolo/person_s.pt')  # model 입력
        self.cls_pub = rospy.Publisher('/person', String, queue_size=1) # tf_cls publish
        
        self.result_img = None
        self.msg = String()
        self.msg.data = 'go'

    def run(self,img):
        self.results = self.model(img,verbose=False,classes=[0], conf=0.50)
        self.result_img = self.results[0].plot()
        for r in self.results:
            size = r.boxes.xywh.cpu().numpy()
            # rospy.loginfo(r.boxes.cls.cpu().numpy())
            if len(size) != 0:
                x = size[0][0]
                y = size[0][1]
                w = size[0][2]
                h = size[0][3]
                if (x <480 and x > 185) and (y >230) and w > 20:
                    self.msg.data = 'stop'
                elif (x <550 and x > 80) and (y >180) and w > 5:
                    self.msg.data = 'slow'
                else:
                    self.msg.data = 'go'
                # rospy.loginfo(f'x:{x:0.1f}, y:{y:0.1f}, w:{w:0.1f}, h:{h:0.1f}')
            else:
                # rospy.loginfo('not detect persons')
                self.msg.data = 'go'                
        self.cls_pub.publish(self.msg)

if __name__ == '__main__':
    rospy.init_node('yolo_run')
    print('start yolo node')
    cam0 = MoraiCamera()
    detect = TrafficLight()
    person = Person()
    
    rate = rospy.Rate(12)
    while not rospy.is_shutdown():
        # start_t = time.time()
        detect.run(cam0.img)
        person.run(cam0.img)
        # cv2.imshow('Orign-result',detect.result_img)
        # cv2.waitKey(1)
        rate.sleep()
        rospy.loginfo(f'light: {detect.msg.data}, person:{person.msg.data}')

        # rospy.loginfo(f'{(time.time() - start_t)*1000:0.1f} ms')