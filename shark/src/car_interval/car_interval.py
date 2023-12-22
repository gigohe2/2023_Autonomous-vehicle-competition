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

from sklearn.cluster import DBSCAN

class CarInter():
    def __init__(self):
        self.rate = 5
        
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
        
        self.img =np.zeros((self.width,self.height))
        
        self.target_distance = 5
        self.obs_dist = 0
        self.dist_pid = PID(Kp=0.6, Ki=0.0, Kd=0.0, dt=1/self.rate)

        self.ctrl_cmd = CtrlCmd()
        self.ctrl_cmd.longlCmdType = 3
        self.ctrl_cmd.accel = 0

        self.rm_points = rospy.Subscriber('/removed_pcd', PointCloud2, self.callback)
        rospy.wait_for_message('/removed_pcd', PointCloud2)
        # self.obs_msg = rospy.Subscriber('/obstacle', obstacle_msg, self.obs_callback)
        # rospy.wait_for_message('/obstacle', PointCloud2)
        
        self.pub_ctrl = rospy.Publisher('ctrl_cmd', CtrlCmd, queue_size=1)
        
        
    def callback(self, msg):
        pc_arr = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        self.xy = pc_arr[:,:2]

        if True:
            db = self.dbscan.fit_predict(self.xy)
            n_cluster = np.max(db) + 1
            
            for c in range(n_cluster):
                # clusters x position : self.removed_xy[db==c, 1]
                # clusters y position : self.removed_xy[db==c, 0]
                
                # center point : c_tmp
                # x : c_tmp[1] / y : c_tmp[0]
                c_tmp = np.mean(self.xy[db==c, :], axis=0)

                #print(c_tmp, len(self.removed_xy[db==c,:]))
                x_min = np.min(self.xy[db==c, 1])
                x_max = np.max(self.xy[db==c, 1])
                y_min = np.min(self.xy[db==c, 0])
                y_max = np.max(self.xy[db==c, 0])

                # radius = np.max(np.linalg.norm(c_tmp-self.removed_xy[db==c,:], axis=1))
                #print("({0}, {1}), ({2}, {3}), radius:{4}".format(x_min, y_min, x_max, y_max, radius))
                
                self.obs_x_list.append(c_tmp[1])
                self.obs_y_list.append(c_tmp[0])
                self.obs_width.append(x_max-x_min)
                self.obs_height.append(y_max-y_min)
                self.obs_ymin_list.append(y_min)