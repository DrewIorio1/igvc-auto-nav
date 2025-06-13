"""
    File: Utility.py - Conains supporting functions for the fusion_node

    Required Libaries:
        numpy - for numerical operations
        cv2 -  for computer vision tasks, like image processing and drawing and line detection
        time - for time-related functions
        math - for mathematical operations like sin , cos

    History 
        
        5/30/2025 - Added Logging supporting function for all to use
"""

import numpy as np
import cv2
import time
import math

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

def detect_lane1(image,
                roi_vertical_start=0.5,
                canny_thresh1=50, canny_thresh2=150,
                hough_rho=1, hough_theta=np.pi/180,
                hough_thresh=50, min_line_len=50, max_line_gap=10):
    """
    Detect white/yellow lane markings in a forward-facing image.

    Args:
        image:    input image (BGR)
        roi_vertical_start: fraction of image height to start ROI
        canny_thresh1: Canny edge detection threshold 1
        canny_thresh2: Canny edge detection threshold 2
        hough_rho: Hough transform rho resolution
        hough_theta: Hough transform theta resolution
        hough_thresh: Hough transform threshold
        min_line_len: minimum line length to be considered a line
        max_line_gap: maximum gap between line segments to be connected

    Returns:
      lane_mask: single-channel binary mask of lane pixels
      lines:    list of line segments [[x1,y1,x2,y2],…] or None
    """
    H, W = image.shape[:2]
    # 1) Restrict to lower half (road ahead)
    y0 = int(H * roi_vertical_start)
    roi = image[y0:, :]

    # 2) Color threshold in HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 180), (180, 25, 255))
    yellow = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))
    mask = cv2.bitwise_or(white, yellow)

    # 3) Morphological closing to seal corners, then blur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blurred = cv2.GaussianBlur(closed, (5, 5), 0)

    # 4) Edge detection
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

    # 5) Probabilistic Hough
    lines = cv2.HoughLinesP(edges,
                            rho=hough_rho,
                            theta=hough_theta,
                            threshold=hough_thresh,
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # lines are in ROI coords; shift them back to full image
    if lines is not None:
        lines[:, 0, 1] += y0
        lines[:, 0, 3] += y0

    # 6) Expand mask back to full image size (for masking later)
    lane_mask = np.zeros((H, W), dtype=np.uint8)
    lane_mask[y0:, :] = closed

    return lane_mask, lines





def adjust_goal_point(original_goal, lane_mask, drivable_mask, robot_xy, pullback_ratio=0.7):
    """
    Adjust goal point to stay within drivable area while pulling it closer to the robot.
    
    Args:
        original_goal (tuple): Initial goal coordinates (x, y)
        lane_mask: Binary lane mask (white=lane)
        drivable_mask: Binary drivable area mask (white=drivable)
        robot_xy (tuple): Robot's current position (x, y)
        pullback_ratio: How much to pull back towards robot (0.0-1.0)
        
    Returns:
        tuple: Adjusted (x, y) coordinates
    """
    robot_x, robot_y = robot_xy
    orig_x, orig_y = original_goal
    
    # Calculate vector from robot to original goal
    dx = orig_x - robot_x
    dy = orig_y - robot_y
    
    # Create pullback point closer to robot
    pullback_x = robot_x + dx * pullback_ratio
    pullback_y = robot_y + dy * pullback_ratio
    
    # Find drivable points near the pullback location
    ys, xs = np.where(drivable_mask > 0)
    if len(xs) == 0:
        return original_goal  # No adjustment possible
    
    # Convert to numpy array for efficient calculations
    drivable_pts = np.column_stack((xs, ys))
    
    # Find nearest drivable point to pullback location
    distances = np.linalg.norm(drivable_pts - [pullback_x, pullback_y], axis=1)
    closest_idx = np.argmin(distances)
    
    return (drivable_pts[closest_idx][0], drivable_pts[closest_idx][1])


def get_next_goal_lane(
    robot_xy,
    robot_heading,
    obstacle_map,
    lane_mask,
    drivable_mask, 
    forward_dist=200,
    roi_width=100,
    roi_height=50,
    min_area=10,
    single_lane_margin=3
):
    """
    Compute the next goal point:
     - If two lanes are detected in the lane_mask ROI, goal = midpoint between lanes.
     - If one lane is detected, goal = farthest right within drivable area (with margin).
     - Else, find the largest obstacle in obstacle_map ROI (masked by lane_mask) and goal = its centroid.

    Returns:
        (x_goal, y_goal) in pixel coords, or None if nothing found.
    """
    """
    1) TWO lanes → midpoint between them.
    2) ONE lane  → middle of drivable_mask in ROI.
    3) ZERO lanes → largest obstacle in obstacle_map.
    """
    H, W = obstacle_map.shape
    # 1) ROI in front of robot
    cx = int(robot_xy[0] + math.cos(robot_heading)*forward_dist)
    cy = int(robot_xy[1] + math.sin(robot_heading)*forward_dist)
    x0 = max(0, min(W - roi_width,  cx - roi_width//2))
    y0 = max(0, min(H - roi_height, cy - roi_height//2))
    x1, y1 = x0 + roi_width, y0 + roi_height

    lane_roi      = lane_mask[y0:y1, x0:x1]
    driv_roi      = drivable_mask[y0:y1, x0:x1]
    obstacle_roi  = obstacle_map[y0:y1, x0:x1]

    def find_centroids(mask):
        cnts,_ = cv2.findContours((mask>0).astype(np.uint8),
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            A = cv2.contourArea(c)
            if A<min_area: continue
            M = cv2.moments(c)
            if M['m00']==0: continue
            cx_r = M['m10']/M['m00']
            cy_r = M['m01']/M['m00']
            out.append((cx_r, cy_r, A, c))
        return out

    lane_cents = find_centroids(lane_roi)

    # --- 1) TWO lanes → midpoint -------------------
    if len(lane_cents) >= 2:
        print("2 Lanes")
        lane_cents.sort(key=lambda x: x[2], reverse=True)  # Sort by area/confidence
    
        # Get lane centroids in ROI coordinates
        xA, yA, _, _ = lane_cents[0]
        xB, yB, _, _ = lane_cents[1]
    
        # Calculate lateral midpoint between lanes
        mid_x = (xA + xB) / 2
    
        # Project forward using robot heading and forward_dist
        proj_y = forward_dist  # Use full forward distance for projection
    
        # Convert to image coordinates (assuming ROI is centered at (cx, cy))
        goal_x = x0 + mid_x
        goal_y = cy  # Maintain original forward distance center
    
        # Alternative: If needing to add to robot's original position
        # goal_x = robot_x + math.cos(robot_heading) * forward_dist + mid_x
        # goal_y = robot_y + math.sin(robot_heading) * forward_dist
    
        return float(goal_x), float(goal_y)

    # --- 2) ONE lane → middle of drivable_mask ----
    if len(lane_cents) == 1:
        # fetch bias from parameters
        print("One Lane")
        params =  0.7#self.get_parameters(['drivable_bias'])
        bias = 0.7# params[0].value

        # crop drivable area
        driv = drivable_mask[y0:y1, x0:x1]
        ys, xs = np.nonzero(driv > 0)
        if len(xs) > min_area:
            left_edge  = float(xs.min())
            right_edge = float(xs.max())

            # weighted midpoint
            mid_local_x = left_edge + bias * (right_edge - left_edge)
            front_local_y = float(ys.min())
            print(f"Initial Goal: {float(x0 + mid_local_x)} Y: {float(y0 + front_local_y)}")
            
            
            adjusted_goal = adjust_goal_point((float(x0 + mid_local_x), 
                float(y0 + front_local_y)) , lane_mask, drivable_mask, robot_xy)
            adjusted_goal = adjusted_goal
            print(f"Adjusted Goal: {adjusted_goal}")
         
            return float(adjusted_goal[0]), float(adjusted_goal[1]) # Step 3: Fall back to original one-lane logic if not enough drivable space
        else: # ELSE TO len(xs) > min_area:
            goal_x, goal_y = select_goal_point( lane_mask, drivable_mask,
                robot_x=robot_xy[0], robot_y=robot_xy[1],
                robot_heading=robot_heading, forward_dist=forward_dist,
                roi_width=roi_width,roi_height=roi_height)
            if goal_x is not None and  goal_x != (None, None):
                print(f"goal x {goal_x} ")
                if isinstance(goal_x, tuple):
                    # Assuming the numeric value is the first element of the tuple.
                    goal_x = goal_x[0]
                    goal_y = goal_y[0] if isinstance(goal_y, tuple) else goal_y
                print(f"Found goal at ({goal_x}, {goal_y})")
                return float(goal_x), float(goal_y)
            else:
                print("No valid goal point found")
    
    ys, xs = np.nonzero(driv_roi > 0)
    """
    if len(xs) > min_area:
        front_y = ys.min()
        goal_x = (xs.min() + xs.max()) / 2  # Midpoint of drivable area
        goal_x = np.clip(goal_x, single_lane_margin, roi_width - single_lane_margin)
        goal_y = float(y0 + front_y)

        return float(x0 + goal_x), float(y0 + goal_y)
    """
    # --- 3) ZERO lanes → obstacle fallback -------
    obs_cents = find_centroids(obstacle_roi)
    if obs_cents:
        obs_cents.sort(key=lambda x: x[2], reverse=True)
        xo, yo, _, _ = obs_cents[0]
        return float(x0 + xo), float(y0 + yo)
    
    
    ys, xs = np.nonzero(driv_roi > 0)
    
   
    
    if len(xs) > min_area:
        print("Using Drivable Area Mid Point")
        # Compute the centroid of the largest drivable region
        mid_x = (xs.min() + xs.max()) / 2
        mid_y = (ys.min() + ys.max()) / 2
        return float(x0 + mid_x), float(y0 + mid_y)
    
    
    
    # --- 5)  fallback find midpoint of drivable area at half of forward_dist ------
    half_forward_dist = forward_dist // 2
    cx_half = int(robot_xy[0] + math.cos(robot_heading) * half_forward_dist)
    cy_half = int(robot_xy[1] + math.sin(robot_heading) * half_forward_dist)

    x0_half = max(0, min(W - roi_width, cx_half - roi_width // 2))
    y0_half = max(0, min(H - roi_height, cy_half - roi_height // 2))
    x1_half, y1_half = x0_half + roi_width, y0_half + roi_height

    driv_half_roi = drivable_mask[y0_half:y1_half, x0_half:x1_half]
    ys_half, xs_half = np.nonzero(driv_half_roi > 0)

    if len(xs_half) > min_area:
        print("Using Half the hight and drivable area")
        mid_x_half = (xs_half.min() + xs_half.max()) / 2
        mid_y_half = (ys_half.min() + ys_half.max()) / 2
        return float(x0_half + mid_x_half), float(y0_half + mid_y_half)
    
    
    # --- 6)  fallback → midpoint of drivable area at 1/4 forward_dist ------
    quarter_forward_dist = forward_dist // 4
    cx_quarter = int(robot_xy[0] + quarter_forward_dist)
    cy_quarter = int(robot_xy[1])  # Ignore heading, stay in same Y axis

    x0_quarter = max(0, min(W - roi_width, cx_quarter - roi_width // 2))
    y0_quarter = max(0, min(H - roi_height, cy_quarter - roi_height // 2))
    x1_quarter, y1_quarter = x0_quarter + roi_width, y0_quarter + roi_height

    driv_quarter_roi = drivable_mask[y0_quarter:y1_quarter, x0_quarter:x1_quarter]
    ys_quarter, xs_quarter = np.nonzero(driv_quarter_roi > 0)

    if len(xs_quarter) > min_area:
        print("Using a quarter of the drivable area")
        mid_x_quarter = (xs_quarter.min() + xs_quarter.max()) / 2
        mid_y_quarter = (ys_quarter.min() + ys_quarter.max()) / 2
        return float(x0_quarter + mid_x_quarter), float(y0_quarter + mid_y_quarter)

    
    print("no goal point")
    # nothing found for testing purpose i like to see the robot move even in a minimal way
    return float(robot_xy[0] - 20), float( robot_xy[1])
    
    

def select_goal_point( lane_mask_raw, drivable_mask, robot_x, robot_y, robot_heading, 
                     forward_dist, roi_width, roi_height):
    """
    Selects the best goal point within the drivable area considering lane boundaries.
    
    Args:
        bgr: Original color image (for dimensions)
        lane_mask_raw: Binary lane mask (white=lane)
        drivable_mask: Binary drivable area mask (white=drivable)
        robot_x, robot_y: Robot position in image coordinates
        robot_heading: Robot heading in radians
        forward_dist: Distance forward to look for goal
        roi_width, roi_height: Size of search region
        
    Returns:
         (goal_x, goal_y) coordinates or (None, None) if no valid goal found
        str: Source of goal point ("Drivable", "Lane", or "None")
    """
    H, W = drivable_mask.shape[:2]
    
    # Calculate center of search region
    cx = int(robot_x + math.cos(robot_heading) * forward_dist)
    cy = int(robot_y + math.sin(robot_heading) * forward_dist)
    
    # Define ROI boundaries (clamped to image)
    x0 = max(cx - roi_width//2, 0)
    x1 = min(cx + roi_width//2, W)
    y0 = max(cy - roi_height//2, 0)
    y1 = min(cy + roi_height//2, H)
    
    # Initialize return values
    goal_x, goal_y = None, None
    source = "None"
    
    if y0 >= y1 or x0 >= x1:
        return (None, None), "Invalid ROI"
    
    # Get ROI slices
    drivable_roi = drivable_mask[y0:y1, x0:x1]
    lane_roi = lane_mask_raw[y0:y1, x0:x1]
    
    # Find all drivable points in ROI
    drivable_pts = np.column_stack(np.where(drivable_roi > 0))
    if len(drivable_pts) > 0:
        # Convert to absolute coordinates
        drivable_pts[:, 0] += y0  # y-coordinate
        drivable_pts[:, 1] += x0  # x-coordinate
        
        # Find point closest to center of ROI
        center_pt = np.array([cy, cx])
        distances = np.linalg.norm(drivable_pts - center_pt, axis=1)
        closest_idx = np.argmin(distances)
        goal_y, goal_x = drivable_pts[closest_idx]
        source = "Drivable"
    else:
        # Fallback to lane points if no drivable area
        lane_pts = np.column_stack(np.where(lane_roi > 0))
        if len(lane_pts) > 0:
            lane_pts[:, 0] += y0
            lane_pts[:, 1] += x0
            distances = np.linalg.norm(lane_pts - center_pt, axis=1)
            closest_idx = np.argmin(distances)
            goal_y, goal_x = lane_pts[closest_idx]
            source = "Lane"
    print(f"source {source}")
    return  goal_x, goal_y #, source


    
def yolo_to_obstacle_map(detections, image_shape, conf_thresh=0.5):
    """
    Converts YOLO detections (list of [x1,y1,x2,y2,conf,class_id])
    into a binary obstacle map of shape image_shape (H, W).
    Any pixel inside a kept bbox is marked 1.
    """
    H, W = image_shape
    obs_map = np.zeros((H, W), dtype=np.uint8)

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf < conf_thresh:
            continue
        # clamp coords to image bounds
        x1i = max(0, min(W-1, int(x1)))
        y1i = max(0, min(H-1, int(y1)))
        x2i = max(0, min(W-1, int(x2)))
        y2i = max(0, min(H-1, int(y2)))
        # fill the rectangle in the obstacle map
        cv2.rectangle(obs_map,
                      (x1i, y1i),
                      (x2i, y2i),
                      color=1,
                      thickness=-1)

    return obs_map








def yolo_track_to_obstacle_map(res, image_shape, conf_thresh=0.0, classes=None, debug=False):
    """
    Convert YOLOv8 track() results into a binary obstacle map.
    Any bbox with conf >= conf_thresh and (cls in classes) will be filled.

    Args:
      res:          Results or list of Results from model.track(..., persist=True)
      image_shape:  (H, W) of your input
      conf_thresh:  include anything above this confidence (set to 0.0 to debug)
      classes:      optional set of class IDs to include
      debug:        if True, print and visualize each box

    Returns:
      obs_map: uint8 H×W, with 255 inside each kept bbox
    """
    H, W = image_shape
    obs_map = np.zeros((H, W), dtype=np.uint8)

    results = res if isinstance(res, (list,tuple)) else [res]
    for r in results:
        # r.boxes.xyxy is a tensor Nx4, r.boxes.conf & r.boxes.cls are Nx1
        bboxes = r.boxes.xyxy.cpu().numpy()   # shape (N,4)
        confs  = r.boxes.conf.cpu().numpy()   # shape (N,)
        clss   = r.boxes.cls.cpu().numpy().astype(int)  # shape (N,)
        for (x1,y1,x2,y2), conf, cls in zip(bboxes, confs, clss):
            if conf < conf_thresh:
                continue
            if classes is not None and cls not in classes:
                continue

            # convert to int pixel coords & clamp
            x1i, y1i = max(0,int(x1)), max(0,int(y1))
            x2i, y2i = min(W-1,int(x2)), min(H-1,int(y2))
            if x2i <= x1i or y2i <= y1i:
                continue

            # draw filled rectangle with value=255
            cv2.rectangle(obs_map, (x1i,y1i), (x2i,y2i), color=255, thickness=-1)

            if debug:
                print(f"DBG: box cls={cls} conf={conf:.2f} -> roi {x1i,y1i,x2i,y2i}")
                #v2.rectangle(vis, (x1i,y1i), (x2i,y2i), (0,0,255), 2)

    if debug:
        print("Obstacle pixels:", np.count_nonzero(obs_map))
    return obs_map


def pad_goal_from_obstacles(x_goal, y_goal, obstacle_map, pad_px=40):
    """
    If there are any obstacle pixels within a pad_px×pad_px box around (x_goal,y_goal),
    compute the centroid of those pixels and push the goal *away* by exactly pad_px pixels.
    """
    H, W = obstacle_map.shape
    # Define a small square ROI around the goal
    x0 = max(0, int(x_goal) - pad_px)
    x1 = min(W, int(x_goal) + pad_px)
    y0 = max(0, int(y_goal) - pad_px)
    y1 = min(H, int(y_goal) + pad_px)
    roi = obstacle_map[y0:y1, x0:x1]
    ys, xs = np.nonzero(roi > 0)
    if len(xs) == 0:
        return x_goal, y_goal  # no obstacles, no shift

    # Centroid of obstacles in the ROI (in full‐image coords)
    cx_obs = x0 + xs.mean()
    cy_obs = y0 + ys.mean()

    # Vector from obstacle towards goal
    vx = x_goal - cx_obs
    vy = y_goal - cy_obs
    norm = math.hypot(vx, vy)
    if norm < 1e-3:
        # goal dead‐on top of obstacles: just move straight back along Y
        return x_goal, y_goal - pad_px

    # Normalize and scale to pad_px
    vx *= pad_px / norm
    vy *= pad_px / norm

    return x_goal + vx, y_goal + vy

def check_obj_distances(distances, threshold=200):
    """
    Check if any distance measurement is below a threshold (in pixels).
    
    Args:
        distances: List of distance measurements
        threshold: Distance threshold in pixels (default: 200)
    
    Returns:
        Tuple of (boolean, list):
        - True if any distance < threshold
        - List of indices where distance < threshold
    """
    warnings = []
    for i, dist in enumerate(distances):
        if dist < threshold:
            warnings.append(i)
    
    return (len(warnings) > 1)







def lidar_to_occupancy(ranges, angle_min, angle_max, 
                      grid_size=400, meters_per_pixel=0.1,
                      max_range=10.0, debug=False):
    """
    Convert LiDAR ranges to binary obstacle map similar to YOLO version
    
    Args:
        ranges: List of distance measurements (meters)
        angle_min/max: Start/end angles of scan (radians)
        grid_size: Size of square output map (pixels)
        meters_per_pixel: Map resolution (m/pixel)
        max_range: Ignore measurements beyond this distance
        debug: Print debug info
        
    Returns:
        obs_map: uint8 grid_size×grid_size, 255=obstacle
    """
    # Initialize empty map (robot at center)
    obs_map = np.zeros((grid_size, grid_size), dtype=np.uint8)
    center = grid_size // 2
    
    # Convert polar to Cartesian and mark obstacles
    for i, dist in enumerate(ranges):
        if dist > max_range or np.isinf(dist):
            continue
            
        # Calculate angle for this measurement
        angle = angle_min + i*(angle_max - angle_min)/len(ranges)
        
        # Convert to grid coordinates
        x = dist * np.cos(angle) / meters_per_pixel
        y = dist * np.sin(angle) / meters_per_pixel
        u = int(center + x)
        v = int(center - y)  # Flip Y-axis for image coordinates
        
        # Mark obstacle if within bounds
        if 0 <= u < grid_size and 0 <= v < grid_size:
            cv2.circle(obs_map, (u, v), radius=2, color=255, thickness=-1)
            
            if debug:
                print(f"LIDAR @ ({u},{v}) - {dist:.2f}m")

    if debug:
        cv2.imshow("Debug Lidar", obs_map)
        cv2.waitKey(1)
        
    return obs_map

def get_obstacle_info(res, image_shape, pixel_thresh=400):
    """
    Args:
        res: YOLOv11n `model.track().
        image_shape: tuple (H, W) of image dimensions.
        pixel_thresh: vertical pixel distance from bottom up (default 400).

    Returns:
        count: number of valid obstacle boxes within the given pixel_thresh.
        Distances: list of vertical distances (in pixels) from the robot origin to the box centers.
    """
    H, W = image_shape
    origin_y = H
    results = res if isinstance(res, (list, tuple)) else [res]
   
    counter = 0
    distances = []

    for r in results:
        for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = box.tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            vertical_dist = origin_y - cy  
            if vertical_dist > 0 and vertical_dist <= pixel_thresh:
                counter += 1
                distances.append(vertical_dist)

    return counter, distances


def gps_to_pixel(lat, lon, navigator):
        """Convert GPS to pixel coordinates using your UTM setup"""
        easting, northing, _, _ = utm.from_latlon(lat, lon)
        dx = easting - navigator.origin_easting
        dy = northing - navigator.origin_northing
        return (
            int(dx * navigator.pixels_per_meter),
            int(-dy * navigator.pixels_per_meter))  





def get_safe_retreat(occ_map, fov_mask, lane_mask,
                     forward_pixels=200, roi_width=400):
    """
    Returns the absolute (row, col) in the full image that maximizes
    occ_map * fov_mask * lane_mask within a 200px window in front.
    """
    H, W = occ_map.shape

    # Crop last `forward_pixels` rows
    occ_roi = occ_map[H - forward_pixels : H, :]
    fov_roi = fov_mask[H - forward_pixels : H, :]
    lane_roi = lane_mask[H - forward_pixels : H, :]

    # Center‐horizontal crop
    center = W // 2
    half = roi_width // 2
    c0 = max(center - half, 0)
    c1 = min(center + half, W)

    occ_roi   = occ_roi[:, c0:c1]
    fov_roi   = fov_roi[:, c0:c1]
    lane_roi  = lane_roi[:, c0:c1]

    # Combined safety score
    score = occ_roi * fov_roi * lane_roi

    # Best pixel in ROI
    rel_row, rel_col = np.unravel_index(np.argmax(score), score.shape)

    # Convert back to full‐image coords
    abs_row = (H - forward_pixels) + rel_row
    abs_col = c0 +       rel_col

    # Sanity check
    assert H - forward_pixels <= abs_row < H, (
        f"abs_row={abs_row} outside [{H-forward_pixels}, {H})"
    )

    return abs_row, abs_col


def get_safe_retreat_vec(occ_map, fov_mask, lane_mask,
                         forward_pixels=200, roi_width=400):
    """
    Returns (dx, dy) in pixels from the vehicle (bottom center):
      - dy > 0 means "forward" up to `forward_pixels`
      - dx > 0 means to the right
    """
    H, W = occ_map.shape

    # Same ROI cropping
    occ_roi = occ_map[H - forward_pixels : H, :]
    fov_roi = fov_mask[H - forward_pixels : H, :]
    lane_roi = lane_mask[H - forward_pixels : H, :]

    center = W // 2
    half   = roi_width // 2
    c0 = max(center - half, 0)
    c1 = min(center + half, W)

    occ_roi   = occ_roi[:, c0:c1]
    fov_roi   = fov_roi[:, c0:c1]
    lane_roi  = lane_roi[:, c0:c1]

    score = occ_roi * fov_roi * lane_roi
    rel_row, rel_col = np.unravel_index(np.argmax(score), score.shape)

    # Pixel‐space vector from bottom‐center:
    #   vehicle at (row=H-1, col=center)
    #   target at (row=(H-forward_pixels)+rel_row, col=c0+rel_col)
    target_row = (H - forward_pixels) + rel_row
    target_col = c0 + rel_col

    dy = (H - 1) - target_row      # positive = backward in image coords
    dx = target_col - center       # positive = right

    # But we want "forward" positive, so flip the sign of dy:
    dy = -dy

    return dx, dy


def convert_to_world(point, grid_resolution=0.1, origin=(0, 0)):
    """
    Converts a grid‐pixel vector (dx, dy) into world meters.
    """
    x_pix, y_pix = point
    x_m = x_pix * grid_resolution + origin[0]
    y_m = y_pix * grid_resolution + origin[1]
    return x_m, y_m



def pad_goal_from_lines(x_goal, y_goal, lines, pad_px=40):
    """
    If any endpoints of line segments in `lines` fall within a pad_px×pad_px window
    around (x_goal, y_goal), compute their centroid and push the goal
    *away* by exactly pad_px pixels along the vector from that centroid.

    Parameters
    ----------
    x_goal : float
      Goal column (horizontal) index.
    y_goal : float
      Goal row (vertical) index.
    lines : array-like
      Array (or list) of line segments, where each line is [x1, y1, x2, y2].
      (The array can be of shape (N,4) or (N,1,4); if the latter, it will be squeezed.)
    pad_px : int
      The minimum distance (in pixels) enforced from any line endpoint.

    Returns
    -------
    (x_new, y_new) : tuple of floats
      The adjusted goal, guaranteed to lie at least pad_px away from the nearby line endpoints.
    """
    # If no lines were found, return the goal unchanged.
    if lines is None or len(lines) == 0:
        return x_goal, y_goal

    # Ensure lines is a NumPy array and squeeze in case of shape (N,1,4)
    lines = np.asarray(lines)
    if lines.ndim == 3:
        lines = np.squeeze(lines, axis=1)  # now shape (N, 4)

    endpoints = []
    # Loop over each line and check both endpoints.
    for line in lines:
        x1, y1, x2, y2 = line
        # Check if the first endpoint is within the pad window
        if (x_goal - pad_px <= x1 <= x_goal + pad_px) and (y_goal - pad_px <= y1 <= y_goal + pad_px):
            endpoints.append((x1, y1))
        # Check if the second endpoint is within the pad window
        if (x_goal - pad_px <= x2 <= x_goal + pad_px) and (y_goal - pad_px <= y2 <= y_goal + pad_px):
            endpoints.append((x2, y2))

    # 2) If no endpoints are nearby, leave the goal as-is.
    if len(endpoints) == 0:
        return x_goal, y_goal

    # 3) Compute the centroid of the nearby endpoints (full-image coordinates).
    endpoints = np.array(endpoints)  # shape (M, 2)
    cx = endpoints[:, 0].mean()
    cy = endpoints[:, 1].mean()

    # 4) Compute vector from the centroid to the goal.
    vx = x_goal - cx
    vy = y_goal - cy
    dist = math.hypot(vx, vy)

    # 5) If the centroid is essentially on the goal, just move straight “up” (–y)
    if dist < 1e-3:
        return x_goal, y_goal - pad_px

    # 6) Normalize the vector and scale it to pad_px.
    vx *= pad_px / dist
    vy *= pad_px / dist

    # 7) Return the “padded” goal.
    return x_goal + vx, y_goal + vy
