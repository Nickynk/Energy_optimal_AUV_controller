#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Software License Agreement (BSD License)
#
#  Copyright (c) 2014, Ocean Systems Laboratory, Heriot-Watt University, UK.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of the Heriot-Watt University nor the names of
#     its contributors may be used to endorse or promote products
#     derived from this software without specific prior written
#     permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  Original authors:
#   Valerio De Carolis, Marian Andrecki, Corina Barbalata, Gordon Frost
#
#  Modified 2017, DROP Lab, university of Michigan, USA
#  Author:
#     Corina Barbalata, Eduardo Iscar, Atulya Shree

from __future__ import division

import numpy as np
np.set_printoptions(precision=10, suppress=False)

import scipy as sci
import scipy.interpolate
import math
import ast

from planning.path import tools as tt
from simulator.util import conversions as cnv
from utils import NED

# default config
DEFAULT_TOLERANCES = np.array([
    1,                # meters
    1,                # meters
    1,                # meters
    np.deg2rad(45),     # radians
    np.deg2rad(45),     # radians
    np.deg2rad(6)       # radians
])

WAYPOINT_SPEED = 0.5 # m/s
WAYPOINT_LOOKAHEAD = 5.0    # meters
LINE_SPACING = 10 # meters
ALTITUDE_OFFSET = 10.0
FLAG_MAP = False
FLAG_GEO = False


class PathGeneration(object):

    def __init__(self, points, position, **kwargs):
        # internal status
        self.cnt = 1  # start always from one (default to first requested waypoint)
        self.des_pos = position  # initialize with current position
        self.des_vel = np.ones(6)  # initialize with normalized speeds
        self.dis_axis = np.zeros(6)  # initialize with empty disable mask

        self.points = np.concatenate((position.reshape(1, 6), points), axis=0)
        self.points = tt.remove_repeated_points(self.points)
        self.points[:, 3:6] = cnv.wrap_pi(self.points[:, 3:6])

        # calculate distances
        self.cum_distances = tt.cumulative_distance(self.points, spacing_dim=3)
        self.total_distance = self.cum_distances[-1]
        self.altitude_offset = ALTITUDE_OFFSET
        self.flag_map = FLAG_MAP
        self.flag_geo = FLAG_GEO

        # NOTE: position tolerances should be passed as a numpy array (or single scalar)
        #   thus it should be parsed by upper modules and not provided directly from the ROS messages/services as strings
        self.tolerances = kwargs.get('tolerances', DEFAULT_TOLERANCES)

        # path status
        self.path_completed = False


    def distance_left(self, position=None):
        return self.cum_distances[-1] - self.distance_completed(position)

    def distance_completed(self, position=None):
        """Calculate the distance along the trajectory covered so far. The distance between the last point (A)
        and current position (B) is calculated by projecting the displacement vector (A-B) on vector representing
        the distance between last point (A) and target point (C). The maximum of this distance is |AC|. Then
        distance from the start of the path to point A is added.
        :param position: numpy array of shape (6)
        :return: float - distance in meters
        """
        current_wp = self.points[self.cnt - 1]
        prev_wp = self.points[self.cnt - 2]
        added_distance = 0

        if position is not None:
            vehicle_direction = (position[0:3] - prev_wp[0:3])
            trajectory_direction = (current_wp[0:3] - prev_wp[0:3])

            if np.linalg.norm(trajectory_direction) != 0:
                added_distance = np.dot(vehicle_direction, trajectory_direction) / np.linalg.norm(trajectory_direction)

        return self.cum_distances[self.cnt - 2] + added_distance

    def calculate_position_error(self, current_position, desired_position):
        error = current_position - desired_position
        error[3:6] = cnv.wrap_pi(error[3:6])
        return error

    def update(self, position, velocity):
        pass

    def __str__(self):
        pass


class MovingWaypointStrategy(PathGeneration):

    def __init__(self, points, position, **kwargs):
        super(MovingWaypointStrategy, self).__init__(points, position, **kwargs)

        # mode config:
        #   - set a target speed for the path navigation
        #   - adjust the time if vehicle is far away from the point
        self.look_ahead = float(kwargs.get('look_ahead', WAYPOINT_LOOKAHEAD))
        self.target_speed = float(kwargs.get('target_speed', WAYPOINT_SPEED))
        self.kind = kwargs.get('interpolation_method', 'linear')
        self.altitude_offset = float(kwargs.get('altitude_offset', 1))
        self.flag_map = kwargs.get('map')

        # trajectory time
        self.dt = 0.2
        self.a = 0.2
        self.v = 0.0
        self.t_interp = self.cum_distances / self.target_speed      # time at the end of each leg
        self.t_end = self.t_interp[-1]

        # interpolating assuming constant speed	
        self.fc = self.points

        # set disable axis mask (fast mode is using few dofs)
        self.dis_axis = np.zeros(6)

        # MPC waypoint init param
        self.n = 1
        self.des_pos = np.zeros(6)
        self.ini_pos = np.zeros(6)
        self.path_x = self.points[:, 0]
        self.path_y = self.points[:, 1]
        self.r_a = 2
        self.ini_sign = 1
        self.dist2path = 0.2

    def update(self, position, velocity):
        # fast motion doesn't require all dofs
        self.dis_axis[1] = 1

        # use real vehicle speed to initialize the virtual speed (this takes into account the previous path inertia)
        if self.v == 0 and velocity[0] > 0.0:
            self.v = velocity[0]

        # MPC waypoint switching
        flag = 0 
        if self.n == 1:
            self.des_pos[0] = self.path_x[self.n]
            self.des_pos[1] = self.path_y[self.n]
            self.ini_pos[0] = self.path_x[self.n-1]
            self.ini_pos[1] = self.path_y[self.n-1]

        if np.mod(self.n,2) == self.ini_sign:
            if np.sqrt((self.des_pos[0]-position[0])**2 + (self.des_pos[1]-position[1])**2) <= self.r_a:
                self.n = self.n + 1; flag = 1
        else:
            d_ct = tt.ct_error(position[0],position[1],self.ini_pos[0],self.ini_pos[1],self.des_pos[0],self.des_pos[1])
            if d_ct <= self.dist2path:
                self.n = self.n + 1; flag = 1

        if flag == 1:
            self.des_pos[0] = self.path_x[self.n]
            self.des_pos[1] = self.path_y[self.n]
            self.ini_pos[0] = self.path_x[self.n-1]
            self.ini_pos[1] = self.path_y[self.n-1]

        self.des_vel = self.ini_pos



class LineStrategy(PathGeneration):
    """LineStrategy provides a line-of-sight navigation mode where the vehicle visit all the path waypoint achieving the
    requested attitude (RPY) but travelling using forward navigation among them, thus implementing a rotate-first approach.
    This mode is useful for its good trade-off between precise positioning and efficient navigation due to
    forward navigation. It make use of the concept of waypoint proximity to adjust the requested attitude when close to
    each requested waypoint.
    """

    def __init__(self, points, position, **kwargs):
        super(LineStrategy, self).__init__(points, position, **kwargs)

        # config
        self.spacing = float(kwargs.get('spacing', LINE_SPACING))
        self.points = tt.interpolate_trajectory(self.points, self.spacing, face_goal=True)
        self.proximity = False

        # calculate distances (after interpolating the points)
        self.cum_distances = tt.cumulative_distance(self.points, spacing_dim=3)
        self.total_distance = self.cum_distances[-1]

        # set disable axis mask (lines mode is using some dofs)
        self.dis_axis = np.zeros(6)

    def update(self, position, velocity):
        if self.cnt >= len(self.points):
            self.des_pos = self.points[-1]
            return

        # select next waypoint
        self.des_pos = self.points[self.cnt]
        self.dis_axis[1] = 1

        # calculate the error using the point in the trajectory list
        error_position = self.calculate_position_error(position, self.des_pos)

        # check if proximity condition (within two tolerances) is reached and latched:
        #   once proximity is triggered the vehicle will adjust for the final yaw
        #   until reaching the requested waypoint (this takes into account disturbances)
        if np.all(np.abs(error_position[0:2]) < 2 * self.tolerances[0:2]):
            self.proximity = True

        # if waypoint is far adjust vehicle orientation towards the next waypoint
        if not self.proximity:
            self.des_pos[5] = tt.calculate_orientation(position, self.des_pos)

        # if waypoint is reached also reset proximity flag (re-enable yaw adjustments)
        if np.all(np.abs(error_position) < self.tolerances):
            self.cnt += 1
            self.proximity = False

            if self.cnt >= len(self.points):
                self.des_pos = self.points[-1]
                self.path_completed = True
