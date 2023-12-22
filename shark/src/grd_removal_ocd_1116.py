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
from lidar_obs_detect.msg import obstacle_msg
import cv2
from tf.transformations import euler_from_quaternion
from scipy.optimize import curve_fit
import message_filters
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
from sympy import Symbol, solve
matplotlib.use('Agg')


class ObsDetector():
    def __init__(self, update_plot=False):
        self.scan_sub = message_filters.Subscriber('/velodyne_points',PointCloud2) # 10hz -> 0.1s
        self.imu_sub = message_filters.Subscriber('/imu', Imu) # 50hz -> 0.02s
        self.syncher = message_filters.ApproximateTimeSynchronizer([self.scan_sub, self.imu_sub], queue_size=5, slop=0.04)
        self.pc_pub = rospy.Publisher('/grd_removed_points', PointCloud2, queue_size=1)
        self.obs_pub = rospy.Publisher('/obstacle', obstacle_msg, queue_size=1)
        self.mission_err_pub = rospy.Publisher('/lidar_error', Float32, queue_size=1)
        self.is_obs_pub = rospy.Publisher('/is_obs', Int32, queue_size=1)
        self.parking_pub = rospy.Publisher('/parking/location', Float32, queue_size=1)
        self.is_parking_pub = rospy.Publisher('/parking/is', Int32, queue_size=1)
        self.quaternion = None
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.imu_count = 0
        self.lidar_count = 0
        self.removed_xy = None
        self.syncher.registerCallback(self.callback)
        self.update_plot = update_plot
        self.dbscan = DBSCAN(eps = 0.2, min_samples=10)
        
    def func(self, x, a, b):
        return a*x**2+b*x
    
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
        
    
    def first_shadow_zone(self):
        ROI_start_y = 5
        ROI_end_y = 10
        front = 5
        try:
            x_left = self.removed_xy[(self.removed_xy[:,1]>0) & (self.removed_xy[:,1]<7.5) & (self.removed_xy[:,0]>-5) & (self.removed_xy[:,0]<20)]
            df = pd.DataFrame(x_left, columns=['y', 'x'])          
            df_min = df.groupby(df['y'].round(decimals=0)).min()
            x_left_min = df_min.values
           
            x = x_left_min[:,1]
            y = x_left_min[:, 0]
            popt, pcov = curve_fit(self.func, x, y)
            curve_y = self.func(x, popt[0], popt[1])
            
            # y = popt[0] * x**3 + popt[1] * x**2 + popt[2] * x
            s=Symbol('s')
            equation = popt[0] * s**2 + popt[1] * s - 5
            left_err = np.max(solve(equation))
            
            # get distance between curve and (0, front)
            self.mission_err_pub.publish(left_err)
            
            rospy.loginfo("tracking left sided larva cone")

        except:
            rospy.loginfo("can't find left sided larva cone")

    def get_parking_area(self):
        ROI_start_x, ROI_end_x = 2.5, 5
        ROI_start_y, ROI_end_y = -4, 15

        PARKING_AREA_MIN_WIDTH = 1.5

        # Recognize parking area
        try:
            # Set ROI
            #left_side = self.removed_xy[(self.removed_xy[:,1]>ROI_start_x) & (self.removed_xy[:,1]<ROI_end_x) & (self.removed_xy[:,0]>ROI_start_y) & (self.removed_xy[:,0]<ROI_end_y)]
            right_side = self.removed_xy[(self.removed_xy[:,1]<-ROI_start_x) & (self.removed_xy[:,1]>-ROI_end_x) & (self.removed_xy[:,0]>ROI_start_y) & (self.removed_xy[:,0]<ROI_end_y)]
            
            # y axis projection
            #left_side_y = np.sort(left_side[:,0]) # > 0
            right_side_y = np.sort(right_side[:,0])[::-1] # < 0

            # find empty space
            #max_gap_left, gap_start_left = 0, 0
            max_gap_right, gap_start_right = 0, 0

            #max_gap_left, gap_start_left = max((start, gap) for start, gap in ((left_side_y[i+1] - left_side_y[i], left_side_y[i]) for i in range(len(left_side_y)-1)))
            max_gap_right, gap_start_right = max((start, gap) for start, gap in ((right_side_y[i] - right_side_y[i+1], right_side_y[i]) for i in range(len(right_side_y)-1)))
            
            #gap_center_left = (gap_start_left + max_gap_left) / 2 
            gap_center_right  = (gap_start_right + max_gap_right) / 2

            # if max_gap_left > max_gap_right and gap_start_left > 3 and max_gap_left > PARKING_AREA_MIN_WIDTH:
            #     rospy.loginfo('Found Parking area on left side, gap : {0}, location : {1}'.format(max_gap_left, gap_start_left))
            if gap_start_right > 0 and max_gap_right > PARKING_AREA_MIN_WIDTH:
                rospy.loginfo('Found Parking area on right side, gap : {0}, location : {1}'.format(max_gap_right, gap_start_right))
                self.parking_pub.publish(gap_start_right)
                self.is_parking_pub.publish(1)
                # Stop at center of parking area
                # go forward gat_start_right[m]
                    
        except:
            self.is_parking_pub.publish(0)
            rospy.loginfo("can't find point in ROI")
        
        # Auto parking
        
        # 1. Control vehicle to parking area center


    def get_center_betcar(self, zone_number):
        # hyper parameter
        ROI_start_y = 2
        ROI_end_y   = 6
        # removed_xy[1] : x ,  removed_xy[0] : y
        if zone_number == 3:
            try:
                x_left = self.removed_xy[(self.removed_xy[:,1]>0) & (self.removed_xy[:,0]>ROI_start_y) & (self.removed_xy[:,0]<ROI_end_y)]
                x_right = self.removed_xy[(self.removed_xy[:,1]<0) & (self.removed_xy[:,0]>ROI_start_y) & (self.removed_xy[:,0]<ROI_end_y)]
                x_left_min = np.min(x_left[:,1])
                x_right_min = np.max(x_right[:,1])
                # if center_error < 0  -> steer to left
                # if center_error > 0  -> steer to right
                print(x_left_min, x_right_min)
                center_error = x_left_min + x_right_min
            except:
                center_error = 0
                rospy.loginfo("can't find larva cone center")

        elif zone_number == 2:
            try:
                x_right = self.removed_xy[(self.removed_xy[:,1]<0) & (self.removed_xy[:,0]>ROI_start_y) & (self.removed_xy[:,0]<ROI_end_y)]
                x_right_min = np.max(x_right[:,1])
                center_error = x_right_min + 2.6
                print(center_error)
            except:
                center_error = 0
                rospy.loginfo("can't find larva cone center")
        
        self.mission_err_pub.publish(center_error)
    
    def obstacle_passing(self):
        obs_steer = 0
        try:
            ROI_xy = self.removed_xy[(self.removed_xy[:,1]<3.5) & (self.removed_xy[:,1]>-3.5) & (self.removed_xy[:,0]>0) & (self.removed_xy[:,0]<20)]
            if len(ROI_xy) !=0:
                left = ROI_xy[(ROI_xy[:,1]>0)]
                right = ROI_xy[(ROI_xy[:,1]<0)]
                if len(left) != 0 and len(right) == 0:
                    obs_left_closest = np.min(left[:,1])
                    obs_right_closest = -10
                elif len(left) ==0 and len(right) !=0:
                    obs_right_closest = np.max(right[:,1])
                    obs_left_closest = 10
                else:
                    obs_left_closest = np.min(left[:,1])
                    obs_right_closest = np.max(right[:,1])
                # left : +, right : -
                #rospy.loginfo("passing", obs_left_closest, obs_right_closest)
                
                obs_steer = obs_left_closest + obs_right_closest
        except:
            rospy.loginfo("None")
        self.mission_err_pub.publish(-obs_steer)

    def obstacle_avoid(self):
        ROI_width = 5
        ROI_height = 10
        obs_steer = 0
        obs_flag = 0
        try:
            ROI_xy = self.removed_xy[(self.removed_xy[:,1]<ROI_width/2) & (self.removed_xy[:,1]>-ROI_width/2) & (self.removed_xy[:,0]>3) & (self.removed_xy[:,0]<ROI_height)]
            
            if len(ROI_xy) != 0:
                rospy.loginfo("obs")
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
                print(obs_left_closest, obs_right_closest)
                if np.abs(obs_left_closest) < np.abs(obs_right_closest):
                    obs_steer = (2.5-obs_left_closest)
                else:
                    obs_steer = (-obs_right_closest-2.5)
                
        except:
            rospy.loginfo("none") 
            obs_flag = 0
        self.is_obs_pub.publish(obs_flag)
        self.mission_err_pub.publish(-obs_steer)
        rospy.loginfo(obs_steer)

    def callback(self, msg, data):
        # IMU
        self.quaternion = (data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w)
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.quaternion)
        self.imu_count += 1
        #print(self.roll)

        # LIDAR
        self.lidar_count += 1
        #print("imu:{0}, lidar:{1}".format(self.imu_count, self.lidar_count))
        pc_arr = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        xyz = pc_arr[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        # roll : right + left -
        R = pcd.get_rotation_matrix_from_xyz([self.roll, self.pitch, 0])  
        
        pcd.rotate(R, center=(0, 0, 0))
        xyz = np.asarray(pcd.points)
        # 0 : y / 1 : x / 2 : z
        xyz = xyz[(xyz[:,1]<20) & (xyz[:,0]>-10) & (xyz[:,1]>-10) & (xyz[:,1]<10) & (xyz[:,2]>-1.1) & (xyz[:,2]<0)]
        xyz[(xyz[:,0] <2.5) & (xyz[:,0]>-2.5) & (xyz[:,1] <1.5) & (xyz[:,1]>-1.5)] = 0
        xyz = xyz[~np.all(xyz==0, axis=1)]
        removed_pcd = o3d.geometry.PointCloud()
        removed_pcd.points = o3d.utility.Vector3dVector(xyz)
        
        self.removed_xy = np.asarray(removed_pcd.points)[:, :2]

        if False:
            plane_model, inliers = removed_pcd.segment_plane(distance_threshold=0.1, ransac_n=10, num_iterations=100)
            inlier_cloud = removed_pcd.select_by_index(inliers)
            outlier_cloud = removed_pcd.select_by_index(inliers, invert=True)
            
            inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
            outlier_cloud.paint_uniform_color([1, 0, 0])

            #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        if False:
            db = self.dbscan.fit_predict(self.removed_xy)
            n_cluster = np.max(db) + 1
            
            obs_msg = obstacle_msg()
            for c in range(n_cluster):
                # clusters x position : self.removed_xy[db==c, 1]
                # clusters y position : self.removed_xy[db==c, 0]
                
                # center point : c_tmp
                # x : c_tmp[1] / y : c_tmp[0]
                c_tmp = np.mean(self.removed_xy[db==c, :], axis=0)

                #print(c_tmp, len(self.removed_xy[db==c,:]))
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
            #print(len(obs_msg.radius))
            self.obs_pub.publish(obs_msg)
        elif False:
            self.clusterer.fit(np.array(pcd.points))
            labels = self.clusterer.labels_

            max_label = labels.max()
            print(f'point cloud has {max_label + 1} clusters')
            colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
            colors[labels < 0] = 0
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            # o3d.visualization.draw_geometries([removed_cloud])
            

        if True:
            self.first_shadow_zone()


        # if self.shadow_counter == 2:
        #     self.get_center_betcar(2)
        #     self.get_parking_area()

        # if self.shadow_counter == 3:
        #     self.get_center_betcar(3)





        # if self.update_plot:
        #     self.updateplot(obs_msg)
        ros_pcd = orh.o3dpc_to_rospc(removed_pcd, 'map')
        self.pc_pub.publish(ros_pcd)
        #rospy.loginfo("loop")


    


if __name__ == '__main__':
    rospy.init_node("velodyen")
    
    rate = rospy.Rate(10)

    obs_detect = ObsDetector(update_plot=True)

    rospy.wait_for_message('/velodyne_points', PointCloud2) 
    rospy.spin()
