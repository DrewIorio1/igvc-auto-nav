The following code was developed for the 2025 IGVC auto navigation competition. It commenced in March 2025, with the objective of completion by the end of May 2025. The code is implemented in Python as a fusion node to perform lane following, object detection, map finding utilizing RRT*, waypoint navigation, and velocity commands for two motors. It was written at the University of Bridgeport with support and assistance from Ali Hamadeen and Rudra Mitra. 

![image](https://github.com/user-attachments/assets/9a785ec3-88bb-4d29-98a0-2ad5f02fa4b9)

The vehicle's autonomous navigation system, based on the Robot Operating System (ROS2), is regulated by the Fusion Controller node, which oversees data integration from various sources, including localization, LiDAR, cameras, GPS, path planning, and control velocity. The localization system, as detailed in Section 6.2, acquires the x and y coordinates, as well as the angle at which the vehicle is heading, through information obtained from the motors and the Inertial Measurement Unit (IMU). Subsequently, the process node determines, based on established thresholds, whether to utilize the camera or LiDAR system. The camera is employed in scenarios where lanes are present, and only a single obstacle is detected; conversely, LiDAR is utilized when an obstacle is identified within a range of four to five meters. Based on the camera input and the established thresholds, the system assesses the availability of lanes and ascertains whether waypoints or obstacles necessitate navigation. These inputs identify the goal points, after which the start and goal points are provided to the RRT* algorithm for the establishment of the optimal path. Ultimately, the speed commands are generated by the control velocity controller, which determines the appropriate speed—ranging from one to five miles per hour—to dispatch to each motor. Each motor operates independently; therefore, the control velocity must account for the differential slip of each tire for every command issued.  

Python: Python 3.10.12

Pip Freeze:

cv-bridge==3.2.1fastimport==0.9.14
fs==2.4.12
fsspec==2025.5.0
genmsg==0.5.16
genpy==0.6.16
geographic-msgs==1.0.6
geometry-msgs==4.8.0
gpg==1.16.0
logging-demo==0.20.5
map-msgs==2.1.0
matplotlib==3.5.1
message-filters==4.3.7
nav-msgs==4.8.0
numpy==1.22.0
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-cupti-cu12==12.6.80
nvidia-cuda-nvrtc-cu12==12.6.77
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.5.1.17
nvidia-cufft-cu12==11.3.0.4
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.7.77
nvidia-cusolver-cu12==11.7.1.2
nvidia-cusparse-cu12==12.5.4.2
nvidia-cusparselt-cu12==0.6.3
nvidia-nccl-cu12==2.26.2
nvidia-nvjitlink-cu12==12.6.85
nvidia-nvtx-cu12==12.6.77
olefile==0.46
opencv-python==4.11.0.86
osrf-pycommon==2.1.6
packaging==25.0
pandas==2.2.3
pcl-msgs==1.0.0
pendulum-msgs==0.20.5
Pillow==9.0.1
ply==3.11
psutil==5.9.0
py==1.10.0
py-cpuinfo==9.0.0
rcl-interfaces==1.2.1
rclpy==3.3.16
rcutils==5.1.6
requests==2.25.1
requests-toolbelt==0.9.1
resource-retriever==3.1.3
right_node @ file:///home/ubuntu/ros2_ws/src/right_node
rmw-dds-common==1.6.0
robot-localization==3.5.3
roman==3.3
ros2action==0.18.12
ros2bag==0.15.14
ros2cli==0.18.12
ros2component==0.18.12
ros2doctor==0.18.12
ros2interface==0.18.12
ros2launch==0.19.9
ros2lifecycle==0.18.12
ros2multicast==0.18.12
ros2node==0.18.12
ros2param==0.18.12
ros2pkg==0.18.12
ros2run==0.18.12
ros2service==0.18.12
ros2topic==0.18.12
rosbag2-interfaces==0.15.14
rosbag2-py==0.15.14
rqt-action==2.0.1
rqt-bag==1.1.5
rqt-bag-plugins==1.1.5
rqt-console==2.0.3
rqt-graph==1.3.1
rqt-gui==1.1.7
rqt-gui-py==1.1.7
rqt-msg==1.2.0
rqt-plot==1.1.5
rqt-publisher==1.5.0
rqt-py-common==1.1.7
rqt-py-console==1.0.2
rqt-reconfigure==1.1.2
rqt-service-caller==1.0.5
rqt-shell==1.0.2
rqt-srv==1.0.3
rqt-topic==1.5.0
scipy==1.8.0
SecretStorage==3.3.1
semver==2.10.2
sensor-msgs==4.8.0
sensor-msgs-py==4.8.0
service-identity==18.1.0
shape-msgs==4.8.0
sros2==0.10.6
statistics-msgs==1.2.1
std-msgs==4.8.0
std-srvs==4.8.0
stereo-msgs==4.8.0
svgwrite==1.4.3
sympy==1.14.0
tf-transformations==1.1.0
tf2-geometry-msgs==0.25.12
tf2-kdl==0.25.12
tf2-msgs==0.25.12
tf2-py==0.25.12
tf2-ros-py==0.25.12
tf2-tools==0.25.12
toml==0.10.2
topic-monitor==0.20.5
torch==2.7.0
torchvision==0.22.0
tqdm==4.67.1
trajectory-msgs==4.8.0
transforms3d==0.3.1
ultralytics==8.3.143
ultralytics-thop==2.0.14


Folder structure 
-	Build: The assembly of the directory for ROS2.
-	fusion_node: The primary execution of the fusion process.  
-	Install: Install from the Robot Operating System (ROS) process.
-	Launch: Housing ROS launch files
-	Models: The models that will be executed during the final implementation. 

