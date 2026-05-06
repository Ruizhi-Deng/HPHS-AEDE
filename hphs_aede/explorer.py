#!/usr/bin/env python3
import rclpy
import rclpy.time
from rclpy.node import Node
import cv2
import numpy as np
import math
import tf2_ros
import sensor_msgs_py.point_cloud2 as pc2
from numpy import inf
from hphs_aede.utils import distance, orientation, diagonal_distance, PolarToCartesian
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped
from collections import deque, defaultdict
from copy import copy
from itertools import permutations
from hphs_aede.dtw import dtw
import time as time_module


_MARKER_SPHERE = 2
_MARKER_SPHERE_LIST = 7
_MARKER_CUBE = 1
_MARKER_CUBE_LIST = 6
_MARKER_LINE_STRIP = 4
_MARKER_ADD = 0


def _euler_from_quaternion(q):
    """q = [x, y, z, w].  Returns (roll, pitch, yaw)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


class Explorer(Node):
    def __init__(self):
        super().__init__("exploration")
        # Basic parameter in the paper
        self.r_gap = 1.2
        self.theta_inf = 15

        self.gap_frontier_vicinity = 0.3
        self.inf_frontier_vicinity = 2.0
        self.img_frontier_vicinity = 0.5
        self.total_frontier_vicinity = 1.5

        self.lamda_1 = 0.1
        self.lamda_2 = 0.3
        self.lamda_3 = 3

        self.k_size = 4
        self.su = 1
        self.sf = 3
        self.so = 5
        self.gamma_1 = 0.8
        self.gamma_2 = 0.1
        self.gamma_3 = 0.1

        self.obs_dist_threshold = 0.2
        self.check_obs_radius = 0.4

        self.angle_increment = np.deg2rad(1.0) # lidar parameter
        self.range_min = 0.2
        self.range_max = 10
        self.angle_min = -np.pi
        self.angle_max = np.pi

        # Subregions
        self.n_w = 3 # number of subregions in width
        self.n_h = 3 # number of subregions in height
        self.subregion_max_width = 10
        self.subregion_max_height = 10
        self.subregions = []
        self.subregion_center = []
        self.classflied_frontiers = []
        self.selected_subregion = 0
        self.optimal_global_path = []
        self.last_global_path = []

        self.h1_max = 5 # using for nomalization in heuristic function
        self.h2_max = 5
        self.h3_max = 10

        # Odom data
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.angle = 0.0
        # Laser data
        self.laser_data = []
        # Map data
        self.map_data = ()
        self.map_resolution = 0.05
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_width = 0 # unit (pix)
        self.map_height = 0
        self.map_width_resized = 0.0 # unit (meter)
        self.map_height_resized = 0.0
        self.map_origin_x_resized = 0.0
        self.map_origin_y_resized = 0.0

        # Note: all frontiers are located on the map frame
        self.total_frontiers = deque()
        self.gap_frontiers   = deque()
        self.inf_frontiers   = deque()
        self.img_frontiers   = deque()

        self.local_goal = [0.0, 0.0]
        self.end_exploration = False

        # Publisher
        self.total_frontiers_pub = self.create_publisher(Marker, "/total_frontiers", 1)
        self.gap_frontiers_pub = self.create_publisher(Marker, "/gap_frontiers", 1)
        self.img_frontiers_pub = self.create_publisher(Marker, "/img_frontiers", 1)
        self.inf_frontiers_pub = self.create_publisher(Marker, "/inf_frontiers", 1)
        
        self.subregions_pub = self.create_publisher(Marker, "/subregions", 1)
        self.selected_subregion_pub = self.create_publisher(Marker, "/selected_subregion", 1)
        self.global_path_pub = self.create_publisher(Marker, "/global_path", 1)
        self.local_goal_pub = self.create_publisher(Marker, "/local_goal", 1)
        self.waypoint_pub = self.create_publisher(PointStamped, "/way_point", 1)
        self.runtime_pub = self.create_publisher(Float32, "/runtime", 1)
        
        # Subscriber
        self.cloud_sub = self.create_subscription(PointCloud2, "/sensor_scan", self.cloud_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, "/state_estimation", self.odom_callback, 1)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 1)
        
        # Transform listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def cloud_callback(self, cloud_msgs):
        cloud_list = list(pc2.read_points(cloud_msgs, field_names=["x", "y", "z"], skip_nans=True))
        cloud_data = []
        for point in cloud_list:
            if -0.2 <= point[2] <= 0.2 and self.range_min <= np.hypot(point[0], point[1]) <= self.range_max:
                cloud_data.append(point)
        
        # Record the minimum polar radius under each polar angle
        cloud_dict = defaultdict(lambda: float('inf'))
        for point in cloud_data:
            radius = np.hypot(point[0], point[1])
            angle = np.arctan2(point[1], point[0])
            angle_deg = np.degrees(angle)
            
            angle_bin = round(angle_deg)
            if radius < cloud_dict[angle_bin]:
                cloud_dict[angle_bin] = radius
        
        self.laser_data = [cloud_dict[angle] if cloud_dict[angle] != float('inf') else self.range_max for angle in range(-180, 181)]

    def odom_callback(self, odom_data):
        self.odom_x = odom_data.pose.pose.position.x
        self.odom_y = odom_data.pose.pose.position.y
        quaternion = [
            odom_data.pose.pose.orientation.x,
            odom_data.pose.pose.orientation.y,
            odom_data.pose.pose.orientation.z,
            odom_data.pose.pose.orientation.w,
        ]
        euler = _euler_from_quaternion(quaternion)
        self.angle = round(euler[2], 4)

    def map_callback(self, map_data):
        self.map_data = map_data.data
        self.map_resolution = map_data.info.resolution
        self.map_width = map_data.info.width
        self.map_height = map_data.info.height
        self.map_origin_x = map_data.info.origin.position.x
        self.map_origin_y = map_data.info.origin.position.y
        self.resizeMap()

    def _plan_path_bfs(self, start_world, goal_world):
        """BFS path on occupancy grid – ROS2 equivalent of move_base global plan."""
        if self.map_width == 0 or len(self.map_data) == 0:
            return []
        OBSTACLE_THRESH = 50
        MAX_NODES = 50000
        sx, sy = self.CoordToIndex(start_world)
        gx, gy = self.CoordToIndex(goal_world)
        if self.map_data[gx + gy * self.map_width] >= OBSTACLE_THRESH:
            return []
        queue = deque([(sx, sy)])
        parent = {(sx, sy): None}
        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        found = False
        while queue and len(parent) < MAX_NODES:
            cx, cy = queue.popleft()
            if cx == gx and cy == gy:
                found = True
                break
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    if (nx, ny) not in parent:
                        if self.map_data[nx + ny * self.map_width] < OBSTACLE_THRESH:
                            parent[(nx, ny)] = (cx, cy)
                            queue.append((nx, ny))
        if not found:
            return []
        path = []
        node = (gx, gy)
        while node is not None:
            path.append(list(self.IndexToCoord(node)))
            node = parent[node]
        path.reverse()
        return path

    def IndexToCoord(self, idx):
        x = (idx[0] + 0.5) * self.map_resolution + self.map_origin_x
        y = (idx[1] + 0.5) * self.map_resolution + self.map_origin_y
        return x, y

    def CoordToIndex(self, coord):
        id_x = min(max(int((coord[0] - self.map_origin_x) / self.map_resolution), 0), self.map_width - 1)
        id_y = min(max(int((coord[1] - self.map_origin_y) / self.map_resolution), 0), self.map_height - 1)
        return id_x, id_y

    def transform(self, source_frame, target_frame, source_loc):
        target_loc = [0, 0]
        t = None
        while t is None:
            try:
                t = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except Exception:
                self.get_logger().info("Cannot find transform from %s to %s" % (source_frame, target_frame))
                time_module.sleep(0.01)

        trans = t.transform.translation
        rot = t.transform.rotation
        eul = _euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
        cos_y = math.cos(eul[2])
        sin_y = math.sin(eul[2])
        target_loc[0] = cos_y * source_loc[0] - sin_y * source_loc[1] + trans.x
        target_loc[1] = sin_y * source_loc[0] + cos_y * source_loc[1] + trans.y
        return target_loc
    
    def resizeMap(self):
        min_x = np.inf
        min_y = np.inf
        max_x = 0
        max_y = 0
        for i in range(self.map_width):
            for j in range(self.map_height):
                map_idx = i + j * self.map_width
                if 0 <= self.map_data[map_idx] <= 10:
                    x = float(i + 0.5) * self.map_resolution + self.map_origin_x
                    y = float(j + 0.5) * self.map_resolution + self.map_origin_y
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
        self.map_origin_x_resized = min_x
        self.map_origin_y_resized = min_y
        width_resized = int(round((max_x - min_x) / self.map_resolution))
        height_resized = int(round((max_y - min_y) / self.map_resolution))
        self.map_width_resized = width_resized * self.map_resolution
        self.map_height_resized = height_resized * self.map_resolution

    # Sampling frontier points based on image processing
    def getImagFrontiers(self, robot_pos, search_range):
        # Calculate detection area
        id_x_start, id_y_start = self.CoordToIndex([robot_pos[0] - search_range, robot_pos[1] - search_range])
        id_x_end, id_y_end     = self.CoordToIndex([robot_pos[0] + search_range, robot_pos[1] + search_range])

        img_width = id_x_end - id_x_start 
        img_height = id_y_end - id_y_start 
        img = np.zeros((img_height, img_width, 1), np.uint8)

        for i in range(id_y_start, id_y_end):
            for j in range(id_x_start, id_x_end):
                if self.map_data[i * self.map_width + j] == 100:
                    img[i - id_y_start, j - id_x_start] = 0
                elif self.map_data[i * self.map_width + j] == 0:
                    img[i - id_y_start, j - id_x_start] = 255
                elif self.map_data[i * self.map_width + j] == -1:
                    img[i - id_y_start, j - id_x_start] = 205

        o = cv2.inRange(img, 0, 1)
        edges = cv2.Canny(img, 0, 255)
        contours, hierarchy = cv2.findContours(o, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, contours, -1, (255,255,255), 5)
        o = cv2.bitwise_not(o) 
        res = cv2.bitwise_and(o, edges)

        frontier = copy(res)
        contours, hierarchy = cv2.findContours(frontier, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frontier, contours, -1, (255,255,255), 2)

        contours, hierarchy = cv2.findContours(frontier, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        frontier_points = []
        if len(contours) > 0:
            i = 0    
            for i in range(0, len(contours)):
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    xr = (id_x_start + cx) * self.map_resolution + self.map_origin_x
                    yr = (id_y_start + cy) * self.map_resolution + self.map_origin_y
                    point = [np.array([xr, yr])]
                    if len(frontier_points) > 0:
                        frontier_points = np.vstack([frontier_points, point])
                    else:    
                        frontier_points = point
        return frontier_points
    
    def getNearestObsDist(self, frontier):
        nearest_obs_dist = np.inf
        id_x_start, id_y_start = self.CoordToIndex([frontier[0] - self.check_obs_radius, frontier[1] - self.check_obs_radius])
        id_x_end, id_y_end     = self.CoordToIndex([frontier[0] + self.check_obs_radius, frontier[1] + self.check_obs_radius])

        for k in range(id_y_start, id_y_end):
            for j in range(id_x_start, id_x_end):
                if self.map_data[k * self.map_width + j] >= 20:
                    obs_coord = self.IndexToCoord([j, k])
                    obs_dist = distance(frontier, obs_coord)
                    if obs_dist < nearest_obs_dist:
                        nearest_obs_dist = obs_dist
        return nearest_obs_dist
   
    ## -----------------Subregion Segmentation and Selection Module -------------------##
    def setSubregion(self):
        self.subregions.clear()
        self.subregion_center.clear()
        subregion_width = self.map_width_resized / self.n_w
        subregion_height = self.map_height_resized / self.n_h

        if subregion_width  > self.subregion_max_width:
            self.n_w = self.n_w + 1
            subregion_width = self.map_width_resized / self.n_w
        if subregion_height > self.subregion_max_height:
            self.n_h = self.n_h + 1
            subregion_height = self.map_height_resized / self.n_h

        for i in range(self.n_w * self.n_h):
            frontier_inside = False
            frontier_count = 0
            center = [0.0, 0.0]
            # Center of the subregion
            center[0] = self.map_origin_x_resized + int(i % self.n_w) * subregion_width + subregion_width / 2
            center[1] = self.map_origin_y_resized + int(i / self.n_w) * subregion_height + subregion_height / 2
            self.subregion_center.append(center)

            for frontier in self.total_frontiers:
                if self.isInside(center, frontier):
                    frontier_count += 1
                if(frontier_count >= 1):
                    frontier_inside = True
                    break
            # Note: Subregions that don't have frontiers will not be added into the subregion set
            if(frontier_inside):
                self.subregions.append(i)

    def isInside(self, center, point):
        subregion_width = self.map_width_resized / self.n_w
        subregion_height = self.map_height_resized / self.n_h
        x = center[0]
        y = center[1]
        x1 = x - subregion_width / 2
        x2 = x + subregion_width / 2
        y1 = y - subregion_height / 2
        y2 = y + subregion_height / 2
        if x1 < point[0] < x2 and y1 < point[1] < y2:
            return True
        else:
            return False

    # Assign the frontiers to their respective subregions by location
    def classflyFrontiers(self):
        self.classflied_frontiers = []
        if len(self.subregions) > 0:
            for i in range(self.n_w * self.n_h):
                frontiers_in_subregion = []
                for frontier in self.total_frontiers:
                    if(self.isInside(self.subregion_center[i], frontier)):
                        frontiers_in_subregion.append(frontier)
                self.classflied_frontiers.append(frontiers_in_subregion)
    
    # Arrange the access order of the subregions
    def arrangeSubregion(self):
        pair_lst = []
        if len(self.subregions) > 0:
            for i in range(len(self.subregions)):
                subregion_idx = self.subregions[i]
                # Distance between the robot and the center of the subregion
                # Todo 1: The distance should be calculated by A* algorithm using the grid map
                # Todo 2: Use the centroid of the frontiers within the subregion, instead of the center of the subregion
                dist = distance([self.odom_x, self.odom_y], self.subregion_center[subregion_idx])
                dist_index_pair = (dist, subregion_idx)
                pair_lst.append(dist_index_pair)
            
            sorted_dist_index_pairs = sorted(pair_lst)
            # Choose the 5 nearest subregions at most, 
            # as the distant subregions contribute less to the overall revenue
            if len(self.subregions) > 5:
                top_indices = [pair[1] for pair in sorted_dist_index_pairs[:5]]
            else:
                top_indices = [pair[1] for pair in sorted_dist_index_pairs]

            # List all the possible of the arrangement of subregions
            # Todo 3: Use heuristic algorithm to optimize the arrangement, instead of the brute force method
            permutations_lst = list(permutations(top_indices, len(top_indices)))
            best_rev = -inf
            for i in range(len(permutations_lst)):
                option_arrangment = permutations_lst[i]
                # Calculate the revenue of an optional arangement of subregions
                total_rev = 0
                cumulative_dist = 0
                for j in range(len(option_arrangment)):
                    cur_idx = option_arrangment[j]

                    if j == 0:
                        dist = distance([self.odom_x, self.odom_y], self.subregion_center[cur_idx]) * self.lamda_3
                    else:
                        last_idx = option_arrangment[j - 1]
                        dist = distance(self.subregion_center[last_idx], self.subregion_center[cur_idx])
                        
                    cumulative_dist += dist
                    rev = np.exp(-self.lamda_1 * cumulative_dist)
                    total_rev += rev

                # Calculate DTW similarity between the last path sequence and the current path sequence
                if len(self.last_global_path) != 0:
                    seq1 = []
                    seq2 = []
                    for k in range(len(option_arrangment)):
                        seq1.append(self.subregion_center[option_arrangment[k]])
                    for l in range(len(self.last_global_path)):
                        seq2.append(self.subregion_center[self.last_global_path[l]])
                    dtw_sim = dtw(seq1, seq2)
                    
                else:
                    dtw_sim = 0

                total_rev = total_rev * np.exp(-self.lamda_2 * dtw_sim)

                if total_rev > best_rev:
                    best_rev = total_rev
                    self.optimal_global_path = option_arrangment

            self.selected_subregion = self.optimal_global_path[0]
            self.last_global_path = self.optimal_global_path
    ## ------------------------------------------------------------------------- ##

    ## --------------------Hybrid Frontiers Sampling Module-------------------- ##
    def isFrontier(self, frontier):
        is_frontier = False
        unknow_grid = 0
        obs_grid = 0
        check_radius = 0.3
        total_grid = 0

        id_x_start, id_y_start = self.CoordToIndex([frontier[0] - check_radius, frontier[1] - check_radius])
        id_x_end, id_y_end     = self.CoordToIndex([frontier[0] + check_radius, frontier[1] + check_radius])

        for i in range(id_x_start, id_x_end + 1):
            for j in range(id_y_start, id_y_end + 1):
                total_grid += 1
                if self.map_data[i + j * self.map_width] == -1:
                    unknow_grid += 1
                elif self.map_data[i + j * self.map_width] >= 10:
                    obs_grid += 1

        if total_grid > 0 and 0.2 < unknow_grid / total_grid < 0.9 and obs_grid / total_grid < 0.3:
            is_frontier = True

        return is_frontier
    
    def isObstacle(self, frontier):
        is_obstacle = False
        check_radius = 0.1

        id_x_start, id_y_start = self.CoordToIndex([frontier[0] - check_radius, frontier[1] - check_radius])
        id_x_end, id_y_end     = self.CoordToIndex([frontier[0] + check_radius, frontier[1] + check_radius])

        for i in range(id_x_start, id_x_end + 1):
            for j in range(id_y_start, id_y_end + 1):
                if self.map_data[i + j * self.map_width] >= 10:
                    is_obstacle = True
                    break

        return is_obstacle


    def addGapFrontiers(self):
        for i in range(len(self.laser_data) - 1):
            if i + 1 >= len(self.laser_data):
                break
            # If the distance difference between two consecutive LiDAR points is greater than r_gap (m), 
            # it is considered that there is an unexplored passable area
            if abs(self.laser_data[i + 1] - self.laser_data[i]) >= self.r_gap:
                radius = (self.laser_data[i] + self.laser_data[i + 1]) / 2
                angle = self.angle_min + (i + 0.5) * self.angle_increment
                x, y = PolarToCartesian(radius, angle)

                # Note: the point is locate on the sensor frame
                new_frontier = [x, y]
                # tranform to the "map" frame
                new_frontier = self.transform('sensor_at_scan', 'map', new_frontier)

                add_bool = True
                if not self.isFrontier(new_frontier) or self.isObstacle(new_frontier):
                    add_bool = False

                if add_bool:
                    if(len(self.gap_frontiers) > 0):
                        for j in range(len(self.gap_frontiers) - 1, -1, -1):
                            d = distance(new_frontier, self.gap_frontiers[j])
                            if d < self.gap_frontier_vicinity:
                                add_bool = False
                                break

                if add_bool:
                    if len(self.total_frontiers) > 0:
                        # Check if the newly added frontier is too close to the existing frontiers
                        for j in range(len(self.total_frontiers) - 1, -1, -1):
                            d = distance(new_frontier, self.total_frontiers[j])
                            if d < self.total_frontier_vicinity:
                                add_bool = False
                                break

                if add_bool:
                    self.total_frontiers.append(new_frontier)
                    self.gap_frontiers.append(new_frontier)

    def addInfFrontiers(self):
        count = 0
        for i in range(90, len(self.laser_data)-90):
            go = False
        # If there theta of inf (exceeds the perceptual range), consider adding a frontier
            if  self.laser_data[i] == self.range_max:
                count += 1
                continue

            if count >= (self.theta_inf / 180 * np.pi / self.angle_increment):
                go = True
            else:
                count = 0

            if go:
                radius = 10
                angle = self.angle_min + (i - (count / 2)) * self.angle_increment
                x, y = PolarToCartesian(radius, angle)
                count = 0
                
                # Note: New frontier is locate on the sensor frame
                new_frontier = [x, y]
                # Tranform to map frame
                new_frontier = self.transform('sensor_at_scan', 'map', new_frontier)

                add_bool = True
                if not self.isFrontier(new_frontier) or self.isObstacle(new_frontier):
                    add_bool = False

                if add_bool:
                    if(len(self.inf_frontiers) > 0):
                        for j in range(len(self.inf_frontiers) - 1, -1, -1):
                            d = distance(new_frontier, self.inf_frontiers[j])
                            if d < self.inf_frontier_vicinity:
                                add_bool = False
                                break

                if add_bool:
                    if len(self.total_frontiers) > 0:
                        for j in range(len(self.total_frontiers) - 1, -1, -1):
                            d = distance(new_frontier, self.total_frontiers[j])
                            if d < self.total_frontier_vicinity:
                                add_bool = False
                                break

                if add_bool:
                    self.total_frontiers.append(new_frontier)
                    self.inf_frontiers.append(new_frontier)

    def addImagFrontiers(self, search_range = 4):
        robot_pos = [self.odom_x, self.odom_y]
        frontiers_num = 0
        # Frontiers is locate on the map frame
        image_frontiers = self.getImagFrontiers(robot_pos, search_range)
        for new_frontier in image_frontiers:
            add_bool = True
            if(len(self.img_frontiers) > 0):
                for j in range(len(self.img_frontiers) - 1, -1, -1):
                    d = distance(new_frontier, self.img_frontiers[j])
                    if d < self.img_frontier_vicinity:
                        add_bool = False
                        break

            if add_bool:
                if len(self.total_frontiers) > 0:
                    for j in range(len(self.total_frontiers) - 1, -1, -1):
                        d = distance(new_frontier, self.total_frontiers[j])
                        if d < self.total_frontier_vicinity:
                            add_bool = False
                            break

            if add_bool:
                self.total_frontiers.append(new_frontier)
                self.img_frontiers.append(new_frontier)
                frontiers_num += 1
        return frontiers_num

    def checkFrontiers(self):
        if len(self.total_frontiers) > 0:
            for i in range(len(self.total_frontiers) - 1, -1, -1):
                remove_bool = False
                # Remove frontiers that have already been approached
                d1 = distance(self.total_frontiers[i], [self.odom_x, self.odom_y])
                if self.total_frontiers[i][0] == self.local_goal[0] and self.total_frontiers[i][1] == self.local_goal[1]:
                    if d1 < 1.5:
                        remove_bool = True
                else:
                    if d1 < 2:
                        remove_bool = True

                # Remove frontiers that are too close to obstacles
                obs_num = 0
                unknow_num = 0
                check_radius = 0.6

                id_x_start, id_y_start = self.CoordToIndex([self.total_frontiers[i][0] - check_radius, self.total_frontiers[i][1] - check_radius])
                id_x_end, id_y_end     = self.CoordToIndex([self.total_frontiers[i][0] + check_radius, self.total_frontiers[i][1] + check_radius])
                for k in range(id_y_start, id_y_end):
                    for j in range(id_x_start, id_x_end):
                        if self.map_data[k * self.map_width + j] >= 1:
                            obs_num += 1
                        if self.map_data[k * self.map_width + j] == -1:
                            unknow_num += 1
                if obs_num >= 5 or unknow_num <= 1:
                    remove_bool = True

                if remove_bool:
                    del self.total_frontiers[i]

    def getFrontiers(self):
        self.addGapFrontiers()
        self.addInfFrontiers()
        self.addImagFrontiers()
    ## -------------------------------------------------------------------- ##

    ## ---------------------Frontier Selection Module---------------------- ##
    def heurisitic(self, frontier):
        # calculate h1
        h1 = diagonal_distance(frontier, [self.odom_x, self.odom_y])
        # update h1_max
        if (h1 > self.h1_max):
            self.h1_max = h1

        # calculate h2
        theta = orientation(self.odom_x, self.odom_y, frontier[0], frontier[1], self.angle)
        h2 = np.exp(2 * (2 * abs(theta) / np.pi - 1))
        # update h2_max
        if (h2 > self.h2_max):
            self.h2_max = h2

        # calculate h3
        id_x_start, id_y_start = self.CoordToIndex([frontier[0] - (self.k_size / 2), 
                                                    frontier[1] - (self.k_size / 2)])
        id_x_end, id_y_end     = self.CoordToIndex([frontier[0] + (self.k_size / 2), 
                                                    frontier[1] + (self.k_size / 2)])
        score = 0
        total_grid = (id_x_end - id_x_start) * (id_y_end - id_y_start)
        for j in range(id_y_start, id_y_end):
            for i in range(id_x_start, id_x_end):
                if self.map_data[j * self.map_width + i] >= 20:
                    score += self.so
                elif 0 <= self.map_data[j * self.map_width + i] <= 20:
                    score += self.sf
                elif self.map_data[j * self.map_width + i] == -1:
                    score += self.su
        score = score / total_grid
        h3 = np.exp(score)
        # update h3_max
        if (h3 > self.h3_max):
            self.h3_max = h3

        # Normalization
        h1_normalized = (h1 - 0) / (self.h1_max - 0)
        h2_normalized = (h2 - 0) / (self.h2_max - 0)
        h3_normalized = (h3 - 0) / (self.h3_max - 0)

        h = self.gamma_1 * h1_normalized  + self.gamma_2 * h2_normalized + self.gamma_3 * h3_normalized
        return h

    def selectLocalGoal(self):
        min_cost = inf
        local_goal = []
        if len(self.classflied_frontiers) > 0:
            frontiers_in_selected_subregion = self.classflied_frontiers[self.selected_subregion]
            for frontier in frontiers_in_selected_subregion:
                cost = self.heurisitic(frontier)
                if cost < min_cost:
                    min_cost = cost
                    local_goal = frontier
            self.local_goal = local_goal

    def sendLocalGoal(self):
        if not self.local_goal:
            return
        # Plan a collision-free path (equivalent to move_base global plan)
        path = self._plan_path_bfs([self.odom_x, self.odom_y], self.local_goal)
        waypoint = PointStamped()
        waypoint.header.frame_id = "map"
        waypoint.header.stamp = self.get_clock().now().to_msg()
        waypoint.point.z = 0.75
        if len(path) > 15:
            wp = path[15]
        elif path:
            wp = path[-1]
        else:
            # No reachable path – fall back to direct goal
            wp = self.local_goal
        waypoint.point.x = float(wp[0])
        waypoint.point.y = float(wp[1])
        self.waypoint_pub.publish(waypoint)
        # Note: Prevents frequent sending of targets
        time_module.sleep(0.1)
    ## ------------------------------------------------------------------------- ##
    
    ## ---------------------------For Visualization--------------------- ##
    def pubFrontierMarkers(self):
        # Publish total frontier markers
        total_frontiers = Marker()
        total_frontiers.header.frame_id = "map"
        total_frontiers.type = _MARKER_SPHERE_LIST
        total_frontiers.action = _MARKER_ADD
        total_frontiers.scale.x = 0.3
        total_frontiers.scale.y = 0.3
        total_frontiers.scale.z = 0.3
        total_frontiers.color.a = 0.6
        total_frontiers.color.r = 0.0
        total_frontiers.color.g = 0.0
        total_frontiers.color.b = 1.0
        total_frontiers.pose.orientation.w = 1.0

        total_points = []
        for frontier in self.total_frontiers:
            p = Point()
            p.x = frontier[0]
            p.y = frontier[1]
            p.z = 0.75
            total_points.append(p)
        total_frontiers.points = total_points
        self.total_frontiers_pub.publish(total_frontiers)

        # Publish gap_frontiers markers
        gap_frontiers = Marker()
        gap_frontiers.header.frame_id = "map"
        gap_frontiers.type = _MARKER_SPHERE_LIST
        gap_frontiers.action = _MARKER_ADD
        gap_frontiers.scale.x = 0.3
        gap_frontiers.scale.y = 0.3
        gap_frontiers.scale.z = 0.3
        gap_frontiers.color.a = 0.6
        gap_frontiers.color.r = 1.0
        gap_frontiers.color.g = 0.0
        gap_frontiers.color.b = 0.0
        gap_frontiers.pose.orientation.w = 1.0

        gap_points = []
        for frontier in self.gap_frontiers:
            p = Point()
            p.x = frontier[0]
            p.y = frontier[1]
            p.z = 0.75
            gap_points.append(p)
        gap_frontiers.points = gap_points
        self.gap_frontiers_pub.publish(gap_frontiers)

        # Publish image_frontiers markers
        img_frontiers = Marker()
        img_frontiers.header.frame_id = "map"
        img_frontiers.type = _MARKER_SPHERE_LIST
        img_frontiers.action = _MARKER_ADD
        img_frontiers.scale.x = 0.3
        img_frontiers.scale.y = 0.3
        img_frontiers.scale.z = 0.3
        img_frontiers.color.a = 0.6
        img_frontiers.color.r = 0.0
        img_frontiers.color.g = 1.0
        img_frontiers.color.b = 0.0
        img_frontiers.pose.orientation.w = 1.0
        img_points = []
        for frontier in self.img_frontiers:
            p = Point()
            p.x = frontier[0]
            p.y = frontier[1]
            p.z = 0.75
            img_points.append(p)
        img_frontiers.points = img_points
        self.img_frontiers_pub.publish(img_frontiers)

        # Publish inf_frontiers markers
        inf_frontiers = Marker()
        inf_frontiers.header.frame_id = "map"
        inf_frontiers.type = _MARKER_SPHERE_LIST
        inf_frontiers.action = _MARKER_ADD
        inf_frontiers.scale.x = 0.3
        inf_frontiers.scale.y = 0.3
        inf_frontiers.scale.z = 0.3
        inf_frontiers.color.a = 0.6
        inf_frontiers.color.r = 0.5
        inf_frontiers.color.g = 0.5
        inf_frontiers.color.b = 0.5
        inf_frontiers.pose.orientation.w = 1.0
        inf_points = []
        for frontier in self.inf_frontiers:
            p = Point()
            p.x = frontier[0]
            p.y = frontier[1]
            p.z = 0.75
            inf_points.append(p)
        inf_frontiers.points = inf_points
        self.inf_frontiers_pub.publish(inf_frontiers)

        # Publish local_goal markers
        local_goal = Marker()
        local_goal.header.frame_id = "map"
        local_goal.type = _MARKER_SPHERE
        local_goal.action = _MARKER_ADD
        local_goal.scale.x = 1.0
        local_goal.scale.y = 1.0
        local_goal.scale.z = 1.0
        local_goal.color.a = 0.6
        local_goal.color.r = 0.0
        local_goal.color.g = 1.0
        local_goal.color.b = 0.0
        local_goal.pose.orientation.w = 1.0
        local_goal.pose.position.x = self.local_goal[0]
        local_goal.pose.position.y = self.local_goal[1]
        local_goal.pose.position.z = 0.8
        self.local_goal_pub.publish(local_goal)

    def pubSubregionMarkers(self):
        if self.map_width_resized == 0.0:
            return

        # Publish marker for subregions
        subregions = Marker()
        subregions.header.frame_id = "map"
        subregions.header.stamp = self.get_clock().now().to_msg()
        subregions.ns = "subregions"
        subregions.type = _MARKER_CUBE_LIST
        subregions.action = _MARKER_ADD
        subregions.pose.orientation.w = 1.0
        subregions.scale.x = self.map_width_resized / self.n_w
        subregions.scale.y = self.map_height_resized / self.n_h
        subregions.scale.z = 1.0
        subregions.color.a = 0.6
        subregions.color.r = 0.0
        subregions.color.g = 0.3
        subregions.color.b = 0.0
        subregion_width = self.map_width_resized / self.n_w
        subregion_height = self.map_height_resized / self.n_h

        subregion_points = []
        for index in self.subregions:
            if index == self.selected_subregion:
                continue
            position = Point()
            position.x = self.map_origin_x_resized + int(index % self.n_w) * subregion_width + subregion_width / 2
            position.y = self.map_origin_y_resized + int(index / self.n_w) * subregion_height + subregion_height / 2
            position.z = 0.75
            subregion_points.append(position)
        subregions.points = subregion_points
        self.subregions_pub.publish(subregions)

        # Publish marker for selected subregion
        selected_subregion = Marker()
        selected_subregion.header.frame_id = "map"
        selected_subregion.header.stamp = self.get_clock().now().to_msg()
        selected_subregion.ns = "selected_subregion"
        selected_subregion.type = _MARKER_CUBE
        selected_subregion.action = _MARKER_ADD
        selected_subregion.pose.orientation.w = 1.0

        selected_subregion.scale.x = self.map_width_resized / self.n_w
        selected_subregion.scale.y = self.map_height_resized / self.n_h
        selected_subregion.scale.z = 1.0
        selected_subregion.color.a = 0.6
        selected_subregion.color.r = 0.0
        selected_subregion.color.g = 0.0
        selected_subregion.color.b = 0.3

        selected_subregion.pose.position.x = self.map_origin_x_resized + int(self.selected_subregion % self.n_w) * subregion_width + subregion_width / 2
        selected_subregion.pose.position.y = self.map_origin_y_resized + int(self.selected_subregion / self.n_w) * subregion_height + subregion_height / 2
        selected_subregion.pose.position.z = 0.75
        self.selected_subregion_pub.publish(selected_subregion)
    
    def drawGlobalPath(self):
        global_path = Marker()
        global_path.header.frame_id = "map"
        global_path.header.stamp = self.get_clock().now().to_msg()
        global_path.ns = "global_path"
        global_path.type = _MARKER_LINE_STRIP
        global_path.action = _MARKER_ADD
        global_path.pose.orientation.w = 1.0
        global_path.scale.x = 0.2
        global_path.color.a = 0.6
        global_path.color.r = 1.0
        global_path.color.g = 0.0
        global_path.color.b = 0.0

        global_points = []
        for index in self.optimal_global_path:
            p1 = Point()
            p1.x = self.subregion_center[index][0]
            p1.y = self.subregion_center[index][1]
            p1.z = 0.0
            global_points.append(p1)
        global_path.points = global_points
        self.global_path_pub.publish(global_path)
    ## ------------------------------------------------------------------------- ##

    def run(self):
        if len(self.laser_data) == 0 or self.map_width == 0:
            return
        
        start_time = time_module.time()
        ## ------------Main Executive Procedure------------ ## 
        self.getFrontiers()
        self.checkFrontiers()

        self.setSubregion()
        self.classflyFrontiers()
        self.arrangeSubregion()

        self.selectLocalGoal()
        self.sendLocalGoal()
        ## ------------------------------------------------ ##
        total_time = time_module.time() - start_time
        runtime_msg = Float32()
        runtime_msg.data = float(total_time)
        self.runtime_pub.publish(runtime_msg)

        if len(self.total_frontiers) == 0:
            num_frontiers = self.addImagFrontiers(50)
            self.get_logger().info('finding frontiers!')
            if num_frontiers == 0:
                self.get_logger().info('no more frontiers, end exploration!')
                self.end_exploration = True

        # Publish markers for visualization
        self.pubFrontierMarkers()
        self.pubSubregionMarkers()
    
def main(args=None):
    rclpy.init(args=args)
    explorer = Explorer()
    while rclpy.ok() and not explorer.end_exploration:
        rclpy.spin_once(explorer, timeout_sec=0.1)
        explorer.run()
    explorer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
