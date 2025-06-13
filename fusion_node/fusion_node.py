# Fusion-2.py
"""
File: Fusion-2.py - Fused Controller for Robot Navigation

This file implements a fused controller for a robot that integrates
computer vision, LIDAR, and GPS data to navigate autonomously.
It uses YOLO for object detection, RRT* for path planning,
and various sensors to maintain localization and control.
It also includes methods for detecting lane markings,
    adjusting goal points, and selecting next goals based on drivable areas.

    Required Libraries:
        math - Mathematical functions
        numpy - Numerical operations
        cv2 - OpenCV for image processing
        time - Time functions
        rclpy - ROS 2 Python client library
        sensor_msgs - ROS 2 message types for sensors
        geometry_msgs - ROS 2 message types for geometry
        nav_msgs - ROS 2 message types for navigation
        cv_bridge - ROS 2 bridge for OpenCV images
        controller - Webots controller API
        ultralytics - YOLO model for object detection   
        tf_transformations - ROS 2 transformations
        tf2_ros - ROS 2 TF2 for broadcasting transforms
        RRTStar - RRT* path planning algorithm
        RobotLocalization - Robot localization module
        RRTPathToGPSConverter - Converts RRT paths to GPS coordinates
        DrivableAreaController - Detects drivable areas
        WaypointNavigator - Navigates waypoints

Methods:
    detect_lane1 - Detects lane markings in an image
    adjust_goal_point - Adjusts goal point to stay within drivable area
    get_next_goal_lane - Computes next goal point based on lane detection
    select_goal_point - Selects the best goal point within the drivable area
    yolo_to_obstacle_map - Converts YOLO detections to obstacle map
    yolo_track_to_obstacle_map - Converts YOLO track results to obstacle map
    pad_goal_from_obstacles - Pads goal point away from obstacles
    lidar_to_occupancy - Converts LIDAR scan to occupancy grid
    gps_to_pixel - Converts GPS coordinates to pixel coordinates

History 
     5/10/2025 AI - Updated hyper parameters for path planning and next goal
     5/26/2025 AI - Updated goal selection to use drivable area and lane detection
     5/28/2025 AI - Moved Helper functions to Utility.py, added cost map publishing to ros2/nav2
                    Incorporated Kalman filter for IMU integration

Todo:
    Need to check if message are in a relevent time frame not an old subscription
"""
import math
from pkgutil import get_loader
from venv import logger
from warnings import catch_warnings
import numpy as np
import cv2
import time
import torch
import numpy as np
import utm
import logging
# Ros2 imports for sensor data and messages
from sensor_msgs.msg import Image, LaserScan, Imu, NavSatFix, JointState
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from cv_bridge import CvBridge

#from controller import Robot
import rclpy
from rclpy.node import Node
from ultralytics import YOLO

import tf_transformations
from tf2_ros import TransformBroadcaster

from tf_transformations import euler_from_quaternion
from rclpy.qos import qos_profile_sensor_data

from RRTStar import RRTStar
from RobotLocalization import RobotLocalization


# Import custom modules for robot control, path conversion, drivable area detection, and waypoint navigation
from robot_controller import WebotsRosController
from RRTPathToGPSConverter import RRTPathToGPSConverter
from DrivableAreaController import DrivableAreaDetector
from WaypointNavigator import WaypointNavigator
from Utility import *



class KalmanFilter1D:
    """
    A simple 1D Kalman filter for estimating position (x) and velocity (v_x).

    State vector: [position, velocity]^T

    Variables:
        dt: Time step for the filter.
        x: State vector (position and velocity).
        P: Error covariance matrix.
        F: State transition matrix.
        B: Control input matrix.
        Q: Process noise covariance.
        H: Measurement matrix.
        R: Measurement noise covariance.

    """
    def __init__(self, dt, measurement_noise, process_noise):
        """
            Constructor for the KalmanFilter1D class.

            Args:
                dt: Time step for the filter.
                measurement_noise: Measurement noise covariance.
                process_noise: Process noise covariance.
        """
        self.dt = dt
        # Initial state: zero position and velocity.
        self.x = np.array([[0.0], [0.0]])
        # Initial error covariance.
        self.P = np.eye(2)
        # State transition matrix (will update with dt).
        self.F = np.array([[1, dt],
                           [0, 1]])
        # Control input matrix.
        self.B = np.array([[0.5 * dt ** 2],
                           [dt]])
        # Process noise covariance.
        self.Q = process_noise * np.eye(2)
        # Measurement matrix: here, we assume we can observe the position.
        self.H = np.array([[1, 0]])
        # Measurement noise covariance.
        self.R = np.array([[measurement_noise]])

    def predict(self, u, dt):
        """
            Predict the next state and update the error covariance.

            Args:
                u: Control input (acceleration).
                dt: Time step for the prediction.

            Returns:
                x: Predicted state vector (position and velocity).
        """

        # Update dt-dependent matrices.
        self.F = np.array([[1, dt],
                           [0, 1]])
        self.B = np.array([[0.5 * dt ** 2],
                           [dt]])
        # Predict the state: x = F * x + B * u
        self.x = np.dot(self.F, self.x) + self.B * u
        # Predict error covariance.
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        """
            Update the state with a new measurement.

            Args:
                z: Measurement (position).

            Returns:
                x: Updated state vector (position and velocity).
        """
        y = z - np.dot(self.H, self.x)
        # Innovation covariance.
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Kalman gain.
        K = np.dot(self.P, self.H.T) @ np.linalg.inv(S)
        # Update state estimate.
        self.x = self.x + np.dot(K, y)
        # Update error covariance.
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x


class IMUIntegrator:
    """
        IMU Integrator Node for ROS2.

        This node subscribes to IMU data and integrates the linear acceleration
        to estimate the position and velocity of the robot. It can use a Kalman

        Variables:
            use_kf: Boolean to choose between Kalman filter and direct integration.
            last_msg_time: Timestamp of the last received IMU message.
            theta: Orientation (yaw in radians).
            x_position: Estimated x position.
            velocity_x: Estimated x velocity.

    """

    def __init__(self):
        """
            Constructor for the IMUIntegrator class.

            Initializes the node, subscribes to IMU data, and sets up the publisher.


        """
        
        # Declare a parameter to choose the integration method.
        #self.declare_parameter('use_kalman', True)
        self.use_kf =True #self.get_parameter('use_kalman').value
        #self.get_logger().info("Using Kalman Filter: " + str(self.use_kf))
        
        
        # Variables for tempo-spatial integration.
        self.last_msg_time = None
        self.theta = 0.0  # Orientation (yaw in radians).
        
        # Initialization for x position integration.
        if self.use_kf:
            # Initialize the Kalman filter with nominal dt and noise parameters.
            self.kf = KalmanFilter1D(dt=0.1, measurement_noise=0.1, process_noise=0.01)
        else:
            self.x_position = 0.0  # Direct integration of position.
            self.velocity_x = 0.0  # Direct integration of velocity.

    def update_imu(self, msg: Imu):
        """
            Callback function for IMU data.

            Args:
                msg: The IMU message containing linear acceleration and angular velocity.
        """
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_msg_time is None:
            self.last_msg_time = current_time
            return
        dt = current_time - self.last_msg_time
        self.last_msg_time = current_time

        # --- Orientation Integration ---
        # Update orientation (theta) from the angular velocity (assumed around the z-axis).
        omega_z = msg.angular_velocity.z
        self.theta += omega_z * dt

        # --- x Position Integration ---
        acceleration_x = msg.linear_acceleration.x
        if self.use_kf:
            # Use the Kalman filter to integrate x acceleration.
            state_pred = self.kf.predict(u=acceleration_x, dt=dt)
            # For demonstration, simulate a measurement by using the prediction.
            # In a real system, replace 'z' with an actual measurement.
            z = np.array([[state_pred[0, 0]]])
            state_est = self.kf.update(z)
            filtered_x = state_est[0, 0]
        else:
            # Direct integration: first update velocity then position.
            self.velocity_x += acceleration_x * dt
            self.x_position += self.velocity_x * dt
            filtered_x = self.x_position

        # --- Prepare the PoseStamped Message ---
        new_msg = Imu()
        new_msg.header = msg.header
        new_msg.linear_acceleration = msg.linear_acceleration
        new_msg.angular_velocity = msg.angular_velocity
        # Assign the computed x position.
        """
        new_msg.pose.position.x = filtered_x
        new_msg.pose.position.y = 0.0  # Assuming movement is along x.
        new_msg.pose.position.z = 0.0
        
        # Convert the yaw (theta) to a quaternion representation.
        new_msg.pose.orientation.x = 0.0
        new_msg.pose.orientation.y = 0.0
        new_msg.pose.orientation.z = math.sin(self.theta / 2.0)
        new_msg.pose.orientation.w = math.cos(self.theta / 2.0)
	
	imu_msg.angular_velocity.x = 0.0
	imu_msg.angular_velocity.y = 0.0
	imu_msg.angular_velocity.z = 0.0
	imu_msg.linear_acceleration.x = filtered_x
	imu_msg.linear_acceleration.y = 0.0
	imu_msg.linear_acceleration.z = 0.0
	imu_msg.orientation_covariance = [0.0] * 9
	imu_msg.angular_velocity_covariance = [0.0] * 9
	imu_msg.linear_acceleration_covariance = [0.0] * 9
	"""
        #tf_transformations.quaternion_from_euler()
        # Publish the pose.
        #self.pose_publisher.publish(pose_msg)
        logger.warn(
            f"dt: {dt:.4f}s, θ: {self.theta:.4f} rad, x: {filtered_x:.4f}"
        )
        return new_msg

        



class FusionNode(Node):
    """
        FusionNode is a ROS2 node that integrates various sensors and algorithms
        for autonomous robot navigation. It subscribes to camera, LIDAR, IMU, 
        GPS, and motor state topics, processes the data for perception and
        planning, and publishes command velocities and paths.
        It uses YOLO for object detection, RRT* for path planning, 
        and maintains localization using IMU and GPS data.

        Variables:
            camera (Camera): The camera sensor instance.
            lidar (LaserScan): The LIDAR sensor instance.
            imu (Imu): The IMU sensor instance.
            gps (NavSatFix): The GPS sensor instance.
            left_motor (JointState): The left motor joint state.
            right_motor (JointState): The right motor joint state.
            localizer (RobotLocalization): Instance for robot localization.
            bridge (CvBridge): Bridge for converting ROS images to OpenCV format.
            br (TransformBroadcaster): Broadcasts transforms for the robot's pose.
            pub_cmd (Publisher): Publisher for command velocities.
            pub_path (Publisher): Publisher for RRT paths.
            map_pub (Publisher): Publisher for cost maps used in Nav2.
            w (int): Width of the camera frame.
            h (int): Height of the camera frame.
            frame (numpy.ndarray): The current camera frame.
            daselector (DrivableAreaDetector): Instance for detecting drivable areas.
            navigator (WaypointNavigator): Instance for waypoint navigation.
        
        Methods:
            __init__: Initializes the FusionNode and its subscriptions.
            left_motor_cb: Callback for left motor joint state.
            right_motor_cb: Callback for right motor joint state.
            image_cb: Callback for camera image messages.
            lidar_cb: Callback for LIDAR scan messages.
            imu_cb: Callback for IMU data messages.
            gps_cb: Callback for GPS fix messages.
            cmd_cb: Callback for command velocity messages.
            run_perception_and_planning: Main loop for perception and planning.
            publish_cmd_vel: Publishes command velocities to control the robot.
            publish_rrt_path: Publishes RRT path as a Path message.
            publish_cost_map: Publishes cost map as an OccupancyGrid message.
            plan_rrt: Plans a path using RRT* algorithm based on obstacles and goals.
    """
    def __init__(self, waypoint_file) :#, robot, camera, lidar, imu, gps, left_motor, right_motor):
        super().__init__('fusion_node')

        logging.basicConfig(filename='app.log')
        logger.setLevel(logging.DEBUG)
        #self.robot = robot
        #self.timestep = int(robot.getBasicTimeStep())
        self.imu_integrator = IMUIntegrator()
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_cb, 10)
        self.imu_sub = self.create_subscription(Imu, '/camera/camera/imu', self.imu_cb, qos_profile_sensor_data)
        self.gps_sub = self.create_subscription(NavSatFix, '/gps/fix', self.gps_cb, 10)
        # self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        self.left_sub = self.create_subscription(JointState, '/left_motor', self.left_motor_cb, 10)
        self.right_motor_sub = self.create_subscription(JointState, '/right_motor', self.right_motor_cb, 10)

        self.left_motor = None #left_motor; self.left_motor.setPosition(float('inf'))
        self.right_motor= None #right_motor;self.right_motor.setPosition(float('inf'))
        #self.left_motor.setVelocity(1.0); self.right_motor.setVelocity(1.0)
        self.localizer = None
        self.bridge = CvBridge()
        self.br = TransformBroadcaster(self)
        
        self.pub_cmd = self.create_publisher(Twist,'/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/rrt_path', 10)
        self.map_pub = self.create_publisher(
            OccupancyGrid, '/cost_map', 10)  # publish cost map for Nav2

        self.linear = 1.0
        self.angular= 0.0
        self.last_cmd = time.time()
        self.w = 1080
        self.h = 1920
        self.frame = None
        self.daselector = DrivableAreaDetector(None, self.h, self.w)
	
        self.imu = None
        self.navigator = None
        self.waypoint_file = waypoint_file
        self.model = YOLO("./best.pt")
        print("YOLO model loaded.")
        self.logmsg('Fusion Node Initialized', False)


    def left_motor_cb(self, msg: JointState):
        """
            left motor encoder callback

            Args:
                msg - left motor as a joint state
        """
        self.left_motor = msg
        #self.left_motor.setPosition(float('inf'))

    def right_motor_cb(self, msg: JointState):
        """
            Right motor encoder call back function

            Args:
                msg - For the left motor encoder
        """
        self.right_motor = msg

    def image_cb(self, msg: Image):
        """
            Image call back from the subscription of the current image 
          

            Note the current code is expecting 30 frames a second
        """
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #TODO apply any stablizations that is need. 
        self.run_perception_and_planning()

    def lidar_cb(self, msg: LaserScan):
        """
            Lidar call back function for calculating the current points

            Args:
                
                msg - as the current scan object as lidar scan
        """
        self.lidar = msg

    def imu_cb(self, msg: Imu):
        """
            Call back for the imu 

            Arg:
                msg - as a type of imu
        """
        self.imu = self.imu_integrator.update_imu(msg)
     
     
    def imu_callback(self, msg: Imu):
        orientation_q = msg.orientation
        q = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(q)
        self.heading = np.degrees(yaw)
        self.get_logger().info(f"Heading (yaw): {self.heading:.2f} degrees")
        print(f'Heading (yaw): {self.heading:.2f} degrees')
        
        #self.localizer.update_imu(self.imu)
        

    def gps_cb(self, msg: NavSatFix):
        """
            GPS call back frunction 

            Args:
                msg - GPS message as a type NavSatFix
        """
        #self.localization.update_gps(msg)
        self.gps = msg

    #def cmd_cb(self, msg: Twist):
    #    self.controller.update_velocity(msg.linear.x, msg.angular.z)
    

    def logmsg(self, message, warning):
        """
            Helper function to log message to command window and logging

            Args:
                message - message to be outputted
                warring - true to log as was warning other wise info
        """

        print(message)
        if warning:
            logging.warning(message)
        else:
            logging.warning(message)

    def run_perception_and_planning(self):
        """
        Main loop for perception and planning.

        This function processes the camera frame, detects lanes, computes

        Remakes:
            Code does a check on the camera frame, imu, right motor (encoder) , left motor (encoder) to ensure all 
            the sensors are responding before calculating a path and command velocity

        """
        if self.frame is None or self.imu is None or self.right_motor is None or self.left_motor is None:
            is_frame = self.frame is None
            is_imu = self.imu is None
            is_right_encoder = self.right_motor is None
            is_left_encoder = self.left_motor is None
            message = f'Initializing Null State Camera Frame Null : {is_frame} and imu state is Null : {is_imu} Right Encoder is Null {is_right_encoder} Left Encoder is Null {is_left_encoder}'
            self.logmsg(message, True)
            return

        #w = camera.getWidth(); h = camera.getHeight()
        if self.localizer is None:
            self.localizer = RobotLocalization(1/1000.0, 0.33, 0.0975, self.left_motor, self.right_motor, imu=self.imu)
        else:
            self.localizer.set_location(self.imu, self.left_motor, self.right_motor)

        position = None
        if not self.gps is None:        
            position = self.gps.getValues()
        latitude = 42.6682354 #41.129718 #position[0]  # X-coordinate (latitude)
        longitude = -83.2174689 #-73.552733  #position[1]  # Y-coordinate (longitude)
        print(f"latitude : {latitude} longitude {longitude}")
        gpsconverter = RRTPathToGPSConverter(latitude, longitude)
    
        # # the known GPS of your Engineering Center NW corner:
        # corner_lat =  42.442123   # for example
        #corner_lon = -76.501234
        #
        # Setup the gps cornate

        if self.navigator is None:
            self.logmsg('Initializing Waypoint Navigation with file ' + self.waypoint_file, False)
            self.navigator = WaypointNavigator(
                yaml_file= self.waypoint_file, #"waypoints.yaml",
                origin_lat=latitude,
                origin_lon=-longitude,
                pixels_per_meter=10  
            )
    
    
        
        #buf = camera.getImage()
        #frame = np.frombuffer(buf, np.uint8).reshape((h, w, 4))
        bgr = cv2.cvtColor(self.frame, cv2.COLOR_BGRA2BGR)
        

        res = model.track(bgr, persist=True)
        vis = res[0].plot() if res and res[0].boxes else bgr.copy()
        #lane_mask, lines, drivable_mask, num_lanes = detect_lane(bgr)
        #drivable_area = get_drivable_area(yolov2, bgr)
        #overlay_and_display(bgr, drivable_area)
        drivable_area = self.daselector.process_frame(self.frame)
        
            # display_overlay=True
        lane_mask_raw_valid, lines_valid = detect_lane1(bgr.copy())
        
        if lines_valid is not None:
            for x1,y1,x2,y2 in lines_valid[:,0]:
                cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("Perception", lane_mask_raw_valid)
        
        self.localizer.update()
        x, y, heading = self.localizer.get_pose()
        
        
        #position = self.gps.getValues()
        latitude = self.gps.latitude  # X-coordinate (latitude)
        longitude = self.gps.longitude  # Y-coordinate (longitude)
        
        local = False
        
        """
            This code is expecting field of view, which means all the points orgin are stating from
            the center bottom
        """
        x = (w/2)
        y = y + h- 1
        print(f"X:{x} Y:{y} heading:{heading}")
        
        checklidar = False
        ranges = None
        occ = None
        # Lidar is not require for the fusion to run ensure that lidar is available to processe
        if not self.lidar is None:
            ranges = self.lidar.getRangeImage()   # returns a Python list or numpy array of floats
            occ = lidar_to_occupancy(
                ranges,
                angle_min=0.0,        # or lidar.getFov() offsets
                angle_max=2*math.pi,
                grid_size=400
            )
        
        
        counter, distances =  get_obstacle_info(res, bgr.shape[:2])
        
        print(f"counter {counter} and distance {distances}")
        
        num_on = np.sum(drivable_area == 255)
        num_off = np.sum(drivable_area == 0)
        print(f"drivable area on {num_on} vs off {num_off}")
        
        #bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        fow = 100
        obstacle_map = yolo_track_to_obstacle_map(res, self.frame.shape[:2],
                                        conf_thresh=0.1,
                                        classes=None)
        
        if not self.lidar is None and check_obj_distances(distances, 350) and num_on < 600:
            """
            drivable_area = lidar_to_drivable_area(ranges,
            angle_min=0.0,        # or lidar.getFov() offsets
            angle_max=2*math.pi,
            grid_size=400)
            obstacle_map = occ
            x,  y = drive_map.shape[:2]
            y = y // 2
            x = x // 2
            """
            #obstacle_map = occ
            if occ is not None:
                checklidar = True
             
        else:
            cost_map =  np.logical_not(drivable_mask)#, _, _, _ = update_cost_map(bgr)
            self.publish_cost_map(cost_map)
             

        #cv2.imshow("Obstacle map", obstacle_map)
        
        grid_size = occ.shape[0]       # 400
        robot_px = grid_size // 2      # 200
        robot_py = grid_size // 2      # 200
        
        
        goal_forward_dist = 150
        goal_roi_width = 200
        goal_roi_height = 100
        
        # Initialize navigation state
        goal = None
        navigating = False
        
        # 1. Check if waypoint navigation is active
        if self.navigator.active:
            
            self.logmsg('Waypoint navigation is active', False)
            # Convert current GPS to pixel coordinates
            current_pixel = gps_to_pixel(latitude, longitude, self.navigator)  
            
            # Update position and get next waypoint goal
            navigating = self.navigator.update_position(current_pixel)
            goal = self.navigator.get_goal_point()
        
        # 2. Fallback to lane navigation when waypoints are complete
        if not goal and not self.navigator.active:
            if self.navigator.check_near_first_waypoint((x, y), threshold=200):
                # Reset waypoint navigation if near start
                self.navigator.current_idx = 0
                self.navigator.path_completed = False
                self.navigator.isactive = True
            else:
                # Use lane-based navigation
                goal = get_next_goal_lane(
                    robot_xy=(x, y),
                    robot_heading=heading,
                    lane_mask=lane_mask_raw_valid,
                    obstacle_map=obstacle_map,
                    drivable_mask=drivable_area,
                    forward_dist=150,
                    roi_width=200,
                    roi_height=160
                )
        
        

        #4. if not goal & check lidar
        if checklidar and not goal:
            # 1. Find safe retreat direction using occupancy map
            vec_goal = get_safe_retreat_vec(occ, drivable_area, lane_mask_raw_valid)

            goal = convert_to_world(vec_goal, grid_resolution=0.1)
            #if goal:
            #    goal = convert_to_world(goal)
            self.logmsg(f"Emergency retreat to {goal}", False)
           
        # 3. Apply obstacle padding to ANY valid goal
        if goal:
            x_goal, y_goal = pad_goal_from_obstacles(
                goal[0], goal[1], 
                obstacle_map, 
                pad_px=30
            )
            x_goal, y_goal = pad_goal_from_lines(x_goal, y_goal, lines_valid, pad_px=30)
            goal = (x_goal, y_goal)
        print(f"goal {goal} after padding")
        path = self.plan_rrt(drivable_area, (x, y), goal) if goal else None
        if path:
            pathgps = gpsconverter.convert_path(path)
            print(f"Path: {path}")
            pathimg =  bgr.copy()
            h, w = bgr.shape[:2]
            converted = [(int(x), int(h - y)) for x, y in path]

            pathimg = bgr.copy()

            for i in range(len(path) - 1):
                cv2.line(pathimg, tuple(map(int, path[i])), tuple(map(int, path[i + 1])), (0, 255, 0), 2)
            
            self.publish_rrt_path(path)
            cv2.imshow("Tru's Path", pathimg)
            controller = WebotsRosController(self.left_motor, self.right_motor,  gps=self.gps, imu=self.imu, origin_lat=None, origin_lon=None, localize=self.localizer, img_h=self.h, img_w= self.w)
            controller.path_set(path)
            v_cmd, w_cmd = controller.update()
            self.publish_cmd_vel(self, v_cmd, w_cmd)
            print("after update motors")
        else:
            self.logmsg(f'No path planned for starting points {latitude} and {longitude}', True)
           
            #self.publish_rrt_path(path)


      



    def publish_cmd_vel(self, linear, angular):
        """
            Publishes a command velocity message to control the robot.

            Args:

            linear (float): Linear velocity in m/s.
            angular (float): Angular velocity in rad/s.
        """
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.pub_cmd.publish(msg)
        self.last_cmd = time.time()

    def publish_rrt_path(self, path_points):
        """
            Pushilisher function for the rrt path

            Args:
                The points in the path


        """
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"  #  changed from "map"
        for x, y in path_points:
            pose = PoseStamped()
            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = "odom"
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.pub_path.publish(msg)

    def publish_cost_map(self, cost_map):
        """
        Publishes the cost map as an OccupancyGrid message.

        Args:
            cost_map (numpy.ndarray): The cost map to publish.
        """
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = 0.1
        msg.info.width = cost_map.shape[1]
        msg.info.height = cost_map.shape[0]
        msg.header.stamp = self.get_clock().now().to_msg()
        # TODO: set grid.info (resolution, width, height, origin)
        msg.data = list((cost_map * 100).astype(np.int8).flatten())
        self.map_pub.publish(msg)



    def plan_rrt(self, obst, start, goal):
        margin = 100
        ra = [min(start[0], goal[0]) - margin, max(start[0], goal[0]) + margin]
        rrt = RRTStar(start, goal, obst, rand_area=ra, max_iter=300)
        return rrt.planning(False)
    
    
def main(args=None):
    print("Fusion Node is running!")
    rclpy.init(args=args)
    node = FusionNode('waypoints_test.yaml')
    try:
        
        rclpy.spin(node)
    except Exception as e:
        print(f"Error Running fusion node {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

