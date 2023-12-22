#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud2
import ros_numpy
import numpy as np
import open3d as o3d
from open3d_ros_helper import open3d_ros_helper as orh
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from std_msgs.msg import Int32
from shark.msg import obstacle_msg
import cv2
from tf.transformations import euler_from_quaternion
from scipy.optimize import curve_fit
import message_filters
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
import matplotlib

matplotlib.use('Agg')

class LidarHandler():
    def __init__(self, update_plot=False):
        self.scan_sub = message_filters.Subscriber('/velodyne_points',PointCloud2) # 10hz -> 0.1s

        self.imu_sub = message_filters.Subscriber('/imu', Imu) # 50hz -> 0.02s
        self.syncher = message_filters.ApproximateTimeSynchronizer([self.scan_sub, self.imu_sub], queue_size=5, slop=0.04)
        self.pc_pub = rospy.Publisher('/grd_removed_points', PointCloud2, queue_size=1)
        self.obs_pub = rospy.Publisher('/obstacle', obstacle_msg, queue_size=1)

        self.quaternion = None
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.removed_xy = None
        self.syncher.registerCallback(self.callback)
        self.update_plot = update_plot
        self.dbscan = DBSCAN(eps = 0.2, min_samples=10)

    def updateplot(self, obstacle):
        x = obstacle.x_min
        y = obstacle.y_min
        width = np.subtract(obstacle.x_max, obstacle.x_min)
        height = np.subtract(obstacle.y_max, obstacle.y_min)

        plt.clf()
        plt.xlim([-10, 10])
        plt.ylim([-10, 20])
        
        plt.scatter(-self.removed_xy[:,1], self.removed_xy[:,0], s=0.3)
      
        ax = plt.gca()  # Get current axes
        for i in range(len(obstacle.x_min)):
            x = obstacle.x_min[i]
            y = obstacle.y_min[i]
            width = np.subtract(obstacle.x_max[i], x)
            height = np.subtract(obstacle.y_max[i], y)

            # Create a rectangle and add it to the plot
            ROI = patches.Rectangle((-2.5,1.5), 5, 10, linewidth=1, edgecolor='g', facecolor='none')
            rect = patches.Rectangle((-x-width, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.add_patch(ROI)
        
        plt.savefig('lidar.png')
        img = cv2.imread('lidar.png')
        cv2.imshow('ld', img)
        cv2.waitKey(1)

    def callback(self, msg, data):
        """lidar + imu -> 바닥 제거

        Args:
            msg (_type_): Lidar data
            data (_type_): IMU data
        """
        # IMU
        self.quaternion = (data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w)
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.quaternion)
        
        pc_arr = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        xyz = pc_arr[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        # roll : right + left -
        R = pcd.get_rotation_matrix_from_xyz([self.roll, self.pitch, 0])  
        
        pcd.rotate(R, center=(0, 0, 0))
        xyz = np.asarray(pcd.points)
        # 0 : y / 1 : x / 2 : z
        xyz = xyz[(xyz[:,1]<20) & (xyz[:,1]>-10) & (xyz[:,1]>-10) & (xyz[:,1]<10) & (xyz[:,2]>-1.33) & (xyz[:,2]<0)]
        xyz[(xyz[:,0] <2.5) & (xyz[:,0]>-2.5) & (xyz[:,1] <1.5) & (xyz[:,1]>-1.5)] = 0
        xyz = xyz[~np.all(xyz==0, axis=1)]
        removed_pcd = o3d.geometry.PointCloud()
        removed_pcd.points = o3d.utility.Vector3dVector(xyz)
        
        self.removed_xy = np.asarray(removed_pcd.points)[:, :2]
    
        db = self.dbscan.fit_predict(self.removed_xy)
        n_cluster = np.max(db) + 1
        
        obs_msg = obstacle_msg()
        for c in range(n_cluster):
            c_tmp = np.mean(self.removed_xy[db==c, :], axis=0)

            x_min = np.min(self.removed_xy[db==c, 1])
            x_max = np.max(self.removed_xy[db==c, 1])
            y_min = np.min(self.removed_xy[db==c, 0])
            y_max = np.max(self.removed_xy[db==c, 0])

            radius = np.max(np.linalg.norm(c_tmp-self.removed_xy[db==c,:], axis=1))
            #print("({0}, {1}), ({2}, {3}), radius:{4}".format(x_min, y_min, x_max, y_max, radius))
            
            obs_msg.x_center.append(c_tmp[1])
            obs_msg.y_center.append(c_tmp[0])
            obs_msg.x_min.append(x_min)
            obs_msg.y_min.append(y_min)
            obs_msg.x_max.append(x_max)
            obs_msg.y_max.append(y_max)
            obs_msg.radius.append(radius)
        self.obs_pub.publish(obs_msg)
        
    
        if self.update_plot:
            self.updateplot(obs_msg)
        ros_pcd = orh.o3dpc_to_rospc(removed_pcd, 'map')
        self.pc_pub.publish(ros_pcd)
        #rospy.loginfo("pc_pub")
        #rospy.loginfo("loop")


if __name__ == '__main__':
    rospy.init_node("velodyne_callback")
    rospy.loginfo("start lidar node")
    rospy.wait_for_message('/velodyne_points', PointCloud2)
    lidar_handler = LidarHandler(update_plot=False)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():

        rate.sleep()
