# 2023_Autonomous-vehicle-competition

This repo is the code for the 2023 autonomous vehicle competition.

## Missions
# 1. Auto parking by 3D LiDAR and IMU sensors
On the competition map, there is a parking space. So, autonomous vehicle needs to recognize a parking area and park there.
First, I established a ROI for parking area based on the map. 

![image](https://github.com/gigohe2/2023_Autonomous-vehicle-competition/assets/59073888/b909d99a-b38e-46c0-8ef1-2f957243935f)

Second, I projected the point cloud which obtained by 3D LiDAR to 2D local space(xy plane).
If there is an empty space over a certain length, I set that space as a parking area.
And then, we need to stop the car beside that parking area. To stop, I applied PID controller by the input of distance between car and that parking area.
After the car stopped at desired point, I used IMU sensor to obtain the orientation of the car. By using that info, do auto parking.

![image](https://github.com/gigohe2/2023_Autonomous-vehicle-competition/assets/59073888/68dbd95b-9aa2-4c82-9b4f-0bb046fe1507)

# 2. Obstacle avoidance by 3D LiDAR and IMU sensors
There are gps shaded areas on the map. So, I have to use other sensors rather than GNSS.
To get a stable point cloud data by 3D LiDAR, I did LiDAR-IMU calibration.
The 3D LiDAR operates at 10Hz and the imu operates at 50Hz. So we have to synchronize two sensor datas. 
Synchronization is implemented by ROS method 'ApproximateTimeSynchronizer'. It is synchronizer which two datas are not given at same time.
And then I used the point cloud data to avoid the obstacles. 

# 3. Traffic light and person detection by camera sensor
We trained yolo to detect traffic light and person. 
There is a sub classes which named 'Person' and 'TrafficLight'. They send traffic light information to main program by ROS topic. 


You can watch our pratice run at the below YOUTUBE link.

https://youtu.be/VVdpXSxjZ6c
