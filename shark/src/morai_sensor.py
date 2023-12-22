#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
import tf
from cv_bridge import CvBridge, CvBridgeError

from math import pi
from pyproj import Transformer
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import ros_numpy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32, Float32MultiArray


from nav_msgs.msg import Odometry
from morai_msgs.msg import GPSMessage


class MoraiCamera():
    def __init__(self,num=''):
        self.bridge = CvBridge()
        self.img = None
        topic = f'/image_jpeg{num}/compressed'
        self.camera_sub = rospy.Subscriber(topic,CompressedImage,self.cam_callback,queue_size=1)
        rospy.wait_for_message(topic,CompressedImage)
        
    def cam_callback(self,msg):
        try:
            self.img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        
            
# lidar 값을 이용한 전방 장애물까지의 거리 추출
class MoraiLidar():
    def __init__(self):
        self.scan_sub = rospy.Subscriber('/velodyne_points',PointCloud2,self.points_callback)
        rospy.wait_for_message('/velodyne_points', PointCloud2)
        # self.dist_pub = rospy.Publisher("dist_point", Float32, queue_size=10)
        self.xyz = None
        # plt.figure(figsize=(150,150))
        # plt.ion()
        # plt.show()
    
    def points_callback(self,msg):
        xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        # 0 : y / 1 : x / 2 : z
        self.xyz = xyz[(xyz[:,0]>0) & (xyz[:,0]<60) & (xyz[:,1]>-20) & (xyz[:,1]<20) & (xyz[:,2]<3) & (xyz[:,2]>-2) ]
        #self.xyz = xyz[(xyz[:,0]>0)]
        #rospy.loginfo(self.xyz)
        #rospy.loginfo(self.xyz.shape)

    def pointclud2_to_img(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # plt.clf()
            # plt.scatter(self.xyz[:,1]*-1, self.xyz[:,0], linestyle='-') 
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.grid(True)
            # plt.pause(0.01)
            rate.sleep()

    def calc_dist_forward(self):
        r1_bool = self.pc_np[:,5] > -30/180*np.pi
        r2_bool = self.pc_np[:,5] < 30/180*np.pi

        d1 = self.pc_np[r1_bool,4]
        d2 = self.pc_np[r2_bool,4]

        d_list = np.concatenate([d1,d2])
        
        return np.min(d_list)
   
# Gps와 Imu를 이용해 odom topic pub
class MoraiOdom():
    def __init__(self):        
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id='/odom'
        self.odom_msg.child_frame_id='/base_link1'
        self.transformer = Transformer.from_crs('epsg:4326', 'epsg:5178')
        
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)

        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.gps_sub = rospy.Subscriber('/gps', GPSMessage, self.gps_callback)
        
        rospy.wait_for_message('/imu',Imu)
        rospy.wait_for_message('/gps',GPSMessage)
        
        self.br = tf.TransformBroadcaster()
        self.x, self.y = None, None
        self.is_imu = False
        self.gps_imu = False      
        
    def gps_callback(self, gps_msg):
        self.gps_imu = True
        self.lat = gps_msg.latitude 
        self.lon = gps_msg.longitude
        
        self.x, self.y = self.transformer.transform(self.lat, self.lon)
       # print(self.x, self.y)
        br = tf.TransformBroadcaster()
        br.sendTransform((self.x, self.y, 0.),
                            quaternion_from_euler(0,0,0.),
                            rospy.Time.now(),
                            "base_link",
                            "map")
        
        self.utm_msg = Float32MultiArray()
        self.utm_msg.data = [self.x, self.y]
        self.odom_msg.header.stamp = rospy.get_rostime()

        self.odom_msg.pose.pose.position.x = self.x
        self.odom_msg.pose.pose.position.y = self.y
        self.odom_msg.pose.pose.position.z = 0
        
        # rospy.loginfo(self.odom_msg.pose)
        self.odom_pub.publish(self.odom_msg)
          
    def imu_callback(self, data):
        self.is_imu = True
        
        self.odom_msg.pose.pose.orientation.x = data.orientation.x
        self.odom_msg.pose.pose.orientation.y = data.orientation.y
        self.odom_msg.pose.pose.orientation.z = data.orientation.z
        self.odom_msg.pose.pose.orientation.w = data.orientation.w
        
        quaternion = (data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w)
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        self.roll_deg = roll/pi*180
        self.pitch_deg = pitch/pi*180
        self.yaw_deg = yaw/pi*180

        self.prev_time=rospy.get_rostime()
        
if __name__ == '__main__':
    rospy.init_node('sensor_test')
    # cam = MoraiCamera(1)
    # lidar = MoraiLidar()
    odom = MoraiOdom()
    
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        # cv2.imshow('Image', cam.img)
        # cv2.waitKey(1)
        rate.sleep()
