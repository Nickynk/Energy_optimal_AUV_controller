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
#     Corina Barbalata


from __future__ import division

import traceback
import numpy as np
import PyKDL
import math
import time

np.set_printoptions(precision=3, suppress=True)

from simulator.util import conversions as cnv
from controller import cascade_controller as cc
from simulator.model import mathematical_model as mm

import rospy
import roslib
import tf2_ros as tf2
import tf2_geometry_msgs
import tf

from auv_msgs.msg import NavSts
from std_srvs.srv import Empty
from std_msgs.msg import Float64MultiArray, Int32
from nav_msgs.msg import Odometry

from vehicle_interface.msg import PilotStatus, PilotRequest, Vector6Stamped
from vehicle_interface.srv import BooleanService, FloatService
from vehicle_interface.srv import CtrlServiceResponse
from vehicle_interface.srv import StartController, RestartController, StopController, LoadController
from utils.msg import ThrustersData
from utils.msg import PressureSensor
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from utils.msg import Altitude
from diagnostic_msgs.msg import KeyValue

#from controller.srv import LoadController, StartController, StopController, RestartController, CtrlServiceResponse

# general config
DEFAULT_RATE = 10.0  # pilot loop rate (Hz)
STATUS_RATE = 2.0  # pilot report rate (Hz)
TIME_REPORT = 0.5  # console logging (sec)

# default config
#   this values can be overridden using external configuration files
#   please see the reload_config() functions for more details
MAX_SPEED = np.array([0.5, 0.0, 0.5, 0.5, 0.0, 0.5])  # default max speed (m/s and rad/s)
THROTTLE_SHAKE = np.array([0.5, 0.5, 0.57, 0.43])

MAX_THROTTLE = 0.5
MAX_THRUST = 7.86

FORCE_X = 0.0

# controller status
CTRL_DISABLED = 0
CTRL_ENABLED = 1
ALTITUDE= False
OFFSET_ALTITUDE = 2.0
MAPPING = False
SHAKE = False

# node states
S_STOP = -1
S_RESET = 0
S_START = 1
S_STAY = 2
S_ALT = 3
S_GO = 4
S_SHAKE = 5

STATUS_CTRL = {
    CTRL_DISABLED: PilotStatus.PILOT_DISABLED,
    CTRL_ENABLED: PilotStatus.PILOT_ENABLED
}

STATUS_MODE = {
    cc.MODE_POSITION: PilotStatus.MODE_POSITION,
    cc.MODE_VELOCITY: PilotStatus.MODE_VELOCITY,
    cc.MODE_STATION: PilotStatus.MODE_STATION
}

# ros topics
TOPIC_ALT = '/sphere_a/nav_sensors/altitude'
TOPIC_NAV = '/sphere_a/nav/pose_estimation'
TOPIC_THR = 'sphere_hw_iface/thruster_data'
TOPIC_PRESSURE = '/sphere_a/nav_sensors/pressure_sensor'
TOPIC_IMU = '/sphere_a/nav_sensors/imu'
TOPIC_FRC = 'forces/sim/body'
TOPIC_STATUS = 'pilot/status'
TOPIC_FORCES = 'pilot/forces'
TOPIC_GAINS = 'controller/gains'
TOPIC_USER = '/user/forces'
TOPIC_JOY_MODE = 'sphere_a/user/joy_mode'
TOPIC_DIS_AXIS = 'sphere_a/user/dis_axis'

TOPIC_POS_REQ = 'pilot/position_req'
TOPIC_BODY_REQ = 'sphere_a/pilot/body_req'
TOPIC_VEL_REQ = 'pilot/velocity_req'
TOPIC_STAY_REQ = 'pilot/stay_req'
TOPIC_OPT_SOL = 'mpc/opt_sol'

SRV_SWITCH = 'sphere_a/pilot/switch'
SRV_RELOAD = 'pilot/reload'
SRV_DIS_AXIS = 'pilot/disable_axis'
SRV_THRUSTERS = 'thrusters/switch'
SRV_START = 'pilot/start'
SRV_STOP = 'pilot/stop'
SRV_RESTART = 'pilot/restart'
SRV_CTRL_FULL = 'pilot/ctrl_full'


# console output
CONSOLE_STATUS = """pilot:
  pos: %s
  vel: %s
  des_p: %s
  des_v: %s
  tau: %s
  dis: %s
"""


class VehicleControl(object):
    """VehicleControl class represent the ROS interface for the pilot subsystem.
        This class implements all the requirements to keep the vehicle within operative limits and requested
        behaviour. The pilot doesn't implement a low-level controller itself but uses one of the Controller implementations
        available in the vehicle_control module, thus making the vehicle control strategy easy to swap and separated from the main
        interfacing logic.
    """

    def __init__(self, name, pilot_rate, **kwargs):
        self.name = name
        self.pilot_rate = pilot_rate

        self.topic_output = kwargs.get('topic_output', TOPIC_FRC)

        self.state = S_RESET

        # timing
        self.dt = 1.0 / self.pilot_rate
        self.pilot_loop = rospy.Rate(self.pilot_rate)

        self.pilot_status = rospy.Timer(rospy.Duration(1.0 / STATUS_RATE), self.send_status)
        self.ctrl_status = CTRL_DISABLED
        self.ctrl_mode = cc.MODE_POSITION

        self.disable_axis = np.zeros(6)  # works at global level
        self.disable_axis_user = np.zeros(6)
        self.max_speed = MAX_SPEED

        # pilot state
        self.pos = np.zeros(6)
        self.pos_prev = np.zeros(6)
        self.vel = np.zeros(6)
        self.vel_prev = np.zeros(6)
        self.des_pos = np.zeros(6)
        self.des_vel = np.zeros(6)
        self.err_pos = np.zeros(6)
        self.err_vel = np.zeros(6)
        self.depth = 0  # directly coming from the pressure sensor
        self.roll = 0  # directly coming from the IMU sensor
        self.pitch = 0
        self.yaw = 0
        self.acc = np.zeros(6)
        self.joy_mode = 0
        self.cnt = 0
        self.shake_cnt = 0
        self.force_x = 0
        self.altitude = 0.0
        self.flag_altitude = ALTITUDE
        self.offset_altitude = OFFSET_ALTITUDE
        self.flag_go = MAPPING
        self.flag_shake = SHAKE

        # plannner 
        self.wp_ind = 0
        self.r_coa = 1.5 
        self.wpx_list = [0.9,-1.0,-3.0,0.9]#[0.4,-0.5,-1.5,0.4] #[0.4,0.4,-2.0]
        self.wpy_list = [-3.5,-0.3,-2.0,-3.5]#[-3.5,-1.5,-2.5,-3.5] #[-2.5,-1.5,-1.5]
        self.wp_num = len(self.wpx_list)
        self.des_pos[0] = self.wpx_list[self.wp_ind]
        self.des_pos[1] = self.wpy_list[self.wp_ind]
        self.des_pos[2] = 1.0 #-0.61
        self.des_pos_pre = np.zeros(6)
        self.des_pos_pre[0] = -4.0
        self.des_pos_pre[1] = -1.65
        self.opt_sol = np.zeros(23)
        self.cput = 0.0

        # controller placeholder
        self.controller = None
        self.ctrl_type = None

        # speeds limits
        self.lim_vel_user = self.max_speed
        self.lim_vel_ctrl = self.max_speed

        # load configuration (this updates limits and modes)
        self.reload_config()

        # outputs
        self.tau_ctrl = np.zeros(6, dtype=np.float64)  # u = [X, Y, Z, K, M, N]
        self.tau = np.zeros(6, dtype=np.float64)  # u = [X, Y, Z, K, M, N]
        self.tau_user = np.zeros(6, dtype=np.float64)
        self.thr = np.zeros(4, dtype=np.float64)
        self.throttle = np.zeros(4, dtype=np.float64)
       # self.throttle_prev =  np.zeros(4, dtype=np.float64)

        self.throttle_prev = THROTTLE_SHAKE

        self.tfBuffer = tf2.Buffer()
        self.listener = tf2.TransformListener(self.tfBuffer)

        # limits on outputs
        self.thr_limit = MAX_THRUST * np.ones(4, dtype=np.float64)

        # thruster mapping matrix
        self.MT = np.zeros((6, 4))

        self.act_on_command = {
            'start': self.cmd_start,
            'stop': self.cmd_stop,
            'reset': self.cmd_reset,
            'stay': self.cmd_stay,
            'alt': self.cmd_alt,
            'go': self.cmd_go,
            'shake': self.cmd_shake
        }

        # ros interface
        self.sub_depth = rospy.Subscriber(TOPIC_PRESSURE, PressureSensor, self.handle_depth, tcp_nodelay=True,
                                          queue_size=1)
        self.sub_imu = rospy.Subscriber(TOPIC_IMU, Imu, self.handle_imu, tcp_nodelay=True, queue_size=1)
        self.sub_nav = rospy.Subscriber(TOPIC_NAV, Odometry, self.handle_nav, tcp_nodelay=True, queue_size=1)
        self.pub_status = rospy.Publisher(TOPIC_STATUS, PilotStatus, tcp_nodelay=True, queue_size=1)
        self.pub_forces = rospy.Publisher(TOPIC_FORCES, Vector6Stamped, tcp_nodelay=True, queue_size=1)
        self.pub_opt_sol = rospy.Publisher(TOPIC_OPT_SOL, Float64MultiArray, tcp_nodelay=True, queue_size=1)
        self.pub_thr = rospy.Publisher(TOPIC_THR, ThrustersData, tcp_nodelay=True, queue_size=1)

        # pilot requests
        self.sub_pos_req = rospy.Subscriber(TOPIC_POS_REQ, PilotRequest, self.handle_pos_req, tcp_nodelay=True,
                                            queue_size=1)
        self.sub_body_req = rospy.Subscriber(TOPIC_BODY_REQ, PilotRequest, self.handle_body_req, tcp_nodelay=True,
                                             queue_size=1)
        self.sub_vel_req = rospy.Subscriber(TOPIC_VEL_REQ, PilotRequest, self.handle_vel_req, tcp_nodelay=True,
                                            queue_size=1)
        self.sub_alt = rospy.Subscriber(TOPIC_ALT, Altitude, self.handle_altitude, queue_size=1)
        # self.sub_stay_req = rospy.Subscriber(TOPIC_STAY_REQ, PilotRequest, self.handle_stay_req, tcp_nodelay=True, queue_size=1)

        # joystick commands
        self.sub_user = rospy.Subscriber(TOPIC_USER, Vector6Stamped, self.handle_user, tcp_nodelay=True, queue_size=1)
        self.disable_axis_user = rospy.Subscriber(TOPIC_DIS_AXIS, Vector6Stamped, self.handle_axis_user,
                                                  tcp_nodelay=True, queue_size=1)
        self.sub_joy_mode = rospy.Subscriber(TOPIC_JOY_MODE, Int32, self.handle_joy_mode, tcp_nodelay=True,
                                             queue_size=1)

        self.s_switch = rospy.Service(SRV_SWITCH, BooleanService, self.srv_switch)
        self.s_reload = rospy.Service(SRV_RELOAD, Empty, self.srv_reload)
        self.s_start = rospy.Service(SRV_START, StartController, self.srv_start_controller)
        self.s_stop = rospy.Service(SRV_STOP, StopController, self.srv_stop_controller)
        self.s_restart = rospy.Service(SRV_RESTART, RestartController, self.srv_restart_controller)
        self.s_ctrl_full = rospy.Service(SRV_CTRL_FULL, LoadController, self.srv_load_controller)


    def reload_config(self, user_config=None):
        """This functions parses the configuration for the pilot and the low-level controller.
        It uses a configuration dictionary if provided, otherwise it queries the ROS param server for loading the latest
        version of the configuration parameters.
        :param user_config: user configuration dictionary (optional)
        """

        try:
            pilot_config = rospy.get_param('pilot', None)
        except Exception:
            tb = traceback.format_exc()
            rospy.logerr('%s: error in loading the configuration from ROS param server:\n%s', self.name, tb)
            return
        # print pilot_config


        # pilot params
        self.prioritize_axis = bool(pilot_config.get('prioritize_axis', False))

        # speed limits
        self.max_speed = np.array(pilot_config.get('max_speed', MAX_SPEED.tolist()), dtype=np.float64)
        self.max_speed = np.clip(self.max_speed, -MAX_SPEED, MAX_SPEED)

        # update controller params
        self.ctrl_config = rospy.get_param('pilot/controller', dict())
        self.model_config = rospy.get_param('vehicle/model', dict())

        # controller selection (if a new controller has been requested by user)
        if self.ctrl_type != self.ctrl_config.get('type', 'cascaded'):
            # store new type
            # print self.ctrl_type
            self.ctrl_type = self.ctrl_config.get('type', 'cascaded')

            if self.ctrl_type == 'cascaded':
                self.controller = cc.CascadeController(self.dt, self.ctrl_config, self.model_config,
                                                       lim_vel=self.max_speed)
            elif self.ctrl_type == 'auto':
                self.controller = cc.AutoTuneController(self.dt, self.ctrl_config, self.model_config,
                                                        lim_vel=self.max_speed)
                self.force_x = self.ctrl_config.get('x_force', float(FORCE_X))
            elif self.ctrl_type == 'pid':
                self.controller = cc.PIDController(self.dt, self.ctrl_config, self.model_config, lim_vel=self.max_speed)
            elif self.ctrl_type == 'mpc':
                self.controller = cc.MPCSurgeController(self.dt, self.ctrl_config, self.model_config,
                                                        lim_vel=self.max_speed)
            elif self.ctrl_type == 'sliding_mode':
                self.controller = cc.SlidingModeController(self.dt, self.ctrl_config, self.model_config,
                                                           lim_vel=self.max_speed)
                self.force_x = self.ctrl_config.get('x_force', float(FORCE_X))
                self.offset_altitude = self.ctrl_config.get('offset_altitude', float(OFFSET_ALTITUDE))
                if self.offset_altitude > 0.0:
                    self.flag_altitude = True
                else:
                    self.flag_altitude = False
            else:
                rospy.logfatal('controller type [%s] not supported', self.ctrl_type)
                raise ValueError('controller type [%s] not supported', self.ctrl_type)

            # notify the selection
            rospy.loginfo('%s: enabling %s controller ...', self.name, self.ctrl_type)

            # load or reload controller configuration
            self.controller.update_config(self.ctrl_config, self.model_config)
            # safety switch service

    def _check_inputs(self):
        # position input checks:
        #   prevent negative depths (out of the water)
        #   remove roll
        #   wrap angles if necessary
        self.des_pos[2] = np.maximum(self.des_pos[2], 0)
        self.des_pos[3:6] = cnv.wrap_pi(self.des_pos[3:6])
        self.des_vel = np.clip(self.des_vel, -self.max_speed, self.max_speed)  # m/s and rad/s



        # limits input checks:
        #   prevent this go above the maximum rated speed
        self.lim_vel_user = np.clip(self.lim_vel_user, -self.max_speed, self.max_speed)
        self.lim_vel_ctrl = np.clip(self.lim_vel_ctrl, -self.max_speed, self.max_speed)

    def handle_depth(self, data):
        self.depth = data.depth
       #self.pos[2] = self.depth

    def handle_altitude(self, data):
        self.altitude = data.altitude


    def handle_imu(self, data):
        dt_int = 0.0001
        #quaternion = (data.orientation.x,
        #              data.orientation.y,
        #              data.orientation.z,
        #              data.orientation.w)
        #euler = tf.transformations.euler_from_quaternion(quaternion)
        #self.roll = euler[0]
        #self.pitch = euler[1]
        #self.yaw = euler[2]

     #   poseMsg = PoseStamped()
     #   poseMsg.pose.position.x = data.linear_acceleration.x
     #   poseMsg.pose.position.y = data.linear_acceleration.y
     #   poseMsg.pose.position.z = data.linear_acceleration.z
     #   poseMsg.pose.orientation.x = data.orientation.x
     #   poseMsg.pose.orientation.y = data.orientation.y
     #   poseMsg.pose.orientation.z = data.orientation.z
     #   poseMsg.pose.orientation.z = data.orientation.w

      #  try:
      #      transform = self.tfBuffer.lookup_transform('sphere_a_ned', 'sphere_a/imu_frame', rospy.Time(0))
      #      (t_r, t_p, t_y) = tf.transformations.euler_from_quaternion([transform.transform.rotation.x, \
      #                                                                  transform.transform.rotation.y, \
      #                                                                  transform.transform.rotation.z, \
      #                                                                  transform.transform.rotation.w])

       # except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException):
       #     rospy.logerr("Could not find transform between sphere_a and imu frame")
       #     return

       # correctedPose = tf2_geometry_msgs.do_transform_pose(poseMsg, transform)

       # self.acc[0] = correctedPose.pose.position.x
       # self.acc[1] = correctedPose.pose.position.y
       # self.acc[2] = correctedPose.pose.position.z

        # self.vel[0:3] = self.vel_prev[0:3] + dt_int*self.acc[0:3]
        # self.pos[0:3] = self.pos_prev[0:3] + dt_int*self.vel[0:3] + dt_int*dt_int*self.acc[0:3]

        #quaternion = (correctedPose.pose.orientation.x,
        #              correctedPose.pose.orientation.y,
        #              correctedPose.pose.orientation.z,
        #              correctedPose.pose.orientation.w)
        #euler = tf.transformations.euler_from_quaternion(quaternion)
        #self.pos[3] = self.pitch  # euler[0]
        #self.pos[4] = self.roll  # euler[1]
        #self.pos[5] = self.yaw  # euler[2]
        #self.pos[3:6] = cnv.wrap_pi(self.pos[3:6])


        # self.pos_prev[0:3] = self.pos[0:3]
        # self.vel_prev[0:3] = self.vel[0:3]

    def handle_nav(self, data):
        # parse navigation data

        quaternion = (data.pose.pose.orientation.x,
                      data.pose.pose.orientation.y,
                      data.pose.pose.orientation.z,
                      data.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)

        self.vel = np.array([
            data.twist.twist.linear.x,
            data.twist.twist.linear.y,
            data.twist.twist.linear.z,
            data.twist.twist.angular.x,
            data.twist.twist.angular.y,
            data.twist.twist.angular.z
        ])
        self.roll = euler[0]
        self.pitch = euler[1]
        self.yaw = euler[2]

        self.pos[0] = data.pose.pose.position.x 
        self.pos[1] = data.pose.pose.position.y  
        self.pos[2] = data.pose.pose.position.z
        self.pos[3] = self.pitch 
        self.pos[4] = self.roll  
        self.pos[5] = self.yaw  
        self.pos[3:6] = cnv.wrap_pi(self.pos[3:6])

        # populate errors (used for info only)
        self.err_pos = self.pos - self.des_pos
        self.err_vel = self.vel - self.des_vel

    def handle_pos_req(self, data):
        try:
            # global referenced request
            self.des_pos = np.array(data.position[0:6])
            self.des_vel = np.zeros(6)
            self.lim_vel_user = self.max_speed
            self.ctrl_status = CTRL_ENABLED
            self._check_inputs()
            self.ctrl_mode = cc.MODE_POSITION
        except Exception:
            tb = traceback.format_exc()
            rospy.logerr('%s: bad position request\n%s', self.name, tb)

    def handle_body_req(self, data):
        try:
            # body referenced request
            J = mm.compute_jacobian(0, 0, self.pos[5])
            body_request = data.position[0:6] #np.dot(J, np.array(data.position[0:6]))
            # print "Body requests ", body_request
            print "Position requested ", self.pos

            self.des_pos = self.pos + body_request
            self.des_vel = np.zeros(6)
            self.lim_vel_user = self.max_speed

            # optionally apply speeds limits if requested by the user
            if len(data.limit_velocity) == 6:
                idx_vel = np.array(data.limit_velocity) > 0
                self.lim_vel_user[idx_vel] = np.array(data.velocity)[idx_vel]

            self._check_inputs()
            self.ctrl_mode = cc.MODE_POSITION
            self.ctrl_status = CTRL_ENABLED
        except Exception:
            tb = traceback.format_exc()
            rospy.logerr('%s: bad body request\n%s', self.name, tb)

    def handle_vel_req(self, data):
        try:
            self.des_pos = np.zeros(6)
            self.des_vel = np.array(data.velocity[0:6])
            self.lim_vel_user = self.max_speed

            self._check_inputs()
            self.ctrl_mode = cc.MODE_VELOCITY
            self.ctrl_status = CTRL_ENABLED
        except Exception:
            tb = traceback.format_exc()
            rospy.logerr('%s: bad velocity request\n%s', self.name, tb)


    def handle_user(self, data):
        # read user input
        if len(data.values) == 6:
            self.tau_user = np.array(data.values)
        else:
            self.tau_user = np.zeros(6)

    def handle_axis_user(self, data):
        self.disable_axis_user = np.array(data.values)

    def handle_joy_mode(self, data):
        # read user input
        self.joy_mode = int(data.data)
        self.cnt = 0

    def send_forces_thr(self):
        ns = ThrustersData()
        ns.header.stamp = rospy.Time.now()
        ns.ul = self.throttle[2]
        ns.ur = self.throttle[3]
        ns.fl = 1 - self.throttle[0]
        ns.fr = self.throttle[1]

        self.pub_thr.publish(ns)

    def send_status(self, event=None):
        ps = PilotStatus()
        ps.header.stamp = rospy.Time.now()

        ps.status = STATUS_CTRL[self.ctrl_status]
        ps.mode = STATUS_MODE[self.ctrl_mode]
        ps.des_pos = self.des_pos.tolist()
        ps.des_vel = self.des_vel.tolist()
        ps.err_pos = self.err_pos.tolist()
        ps.err_vel = self.err_vel.tolist()

        ps.lim_vel_ctrl = self.lim_vel_ctrl.tolist()
        ps.lim_vel_user = self.lim_vel_user.tolist()

        self.pub_status.publish(ps)

    def srv_start_controller(self):
        """This function handles the switch service.
        This will start the low-level controller, leaving the pilot parsing only the user input if disabled.
        Upon enabling this function reloads the configuration from the ROS param server, providing a quick way to change
        parameters in order to adjust the behaviour of the pilot or the low-level controller.
        Upon re-enabling the pilot will use the last known values for mode, desired position and velocities.
        """
        # enable the low-level controller
        self.ctrl_status = CTRL_ENABLED
        return True

    def srv_stop_controller(self):
        """This function handles the switch service.
        This will start the low-level controller, leaving the pilot parsing only the user input if disabled.
        Upon enabling this function reloads the configuration from the ROS param server, providing a quick way to change
        parameters in order to adjust the behaviour of the pilot or the low-level controller.
        Upon re-enabling the pilot will use the last known values for mode, desired position and velocities.
        """
        # enable the low-level controller
        self.ctrl_status = CTRL_DISABLED
        return True

    def srv_restart_controller(self):
        """This function handles the switch service.
        This will start the low-level controller, leaving the pilot parsing only the user input if disabled.
        Upon enabling this function reloads the configuration from the ROS param server, providing a quick way to change
        parameters in order to adjust the behaviour of the pilot or the low-level controller.
        Upon re-enabling the pilot will use the last known values for mode, desired position and velocities.
        """
        # enable the low-level controller
        if (self.ctrl_status == CTRL_ENABLED):
            self.ctrl_status = CTRL_DISABLED
            rospy.sleep(2)
            self.ctrl_status = CTRL_ENABLED
        return True

    # safety switch service
    def srv_switch(self, req):
        """This function handles the switch service.
        This will enable/disable the low-level controller, leaving the pilot parsing only the user input if disabled.
        Upon enabling this function reloads the configuration from the ROS param server, providing a quick way to change
        parameters in order to adjust the behaviour of the pilot or the low-level controller.
        Upon re-enabling the pilot will use the last known values for mode, desired position and velocities.
        """
        # self.ctrl_status = CTRL_ENABLED
        # return True
        if req.request is True:
            # reload configuration
            self.reload_config()

            # enable the low-level controller
            self.ctrl_status = CTRL_ENABLED
            # reset axis disable of last request
            #   this doesn't change the global axis disable request
            self.disable_axis_user = np.zeros(6)

            return True
        else:
            self.ctrl_status = CTRL_DISABLED
            return False

    def srv_load_controller(self, req):
        """This function handles the full controller reload with similar functionalities as the joystick.
               This will enable/disable the low-level controller, leaving the pilot parsing only the user input if disabled.
               Upon enabling this function reloads the configuration from the ROS param server, providing a quick way to change
               parameters in order to adjust the behaviour of the pilot or the low-level controller.
               Furthermore, it sets if the vehicle uses depth station-keeping or altitude station-keeping in the full control mode.
               Upon re-enabling the pilot will use the last known values for mode, desired position and velocities.
               """

        packed_request = {
            'command': req.command
        }

        for option in req.options:
            packed_request[option.key] = option.value

        info = {}
        #res = BooleanService #CtrlServiceResponse()

        try:
            cmd = packed_request['command']
            response = self.act_on_command[cmd](**packed_request)
            info.update(response)
        except Exception:
            info['error'] = 'unspecified command'

        #res.result = True
        #res.state = STATUS_CTRL[self.state]

        return True


    def cmd_start(self, **kwargs):
            try:
                self.ctrl_status = CTRL_ENABLED
                rospy.logdebug('%s: switching on controller: %s', self.name, self.ctrl_status)
                self.state = S_START

            except rospy.ServiceException:
                tb = traceback.format_exc()
                rospy.logerr('%s: controller service error:\n%s', self.name, tb)
                return {'error': 'controller switch failed'}

            return {}

    def cmd_stop(self, **kwargs):
        try:
            self.ctrl_status = CTRL_DISABLED
            self.joy_mode = 0
            rospy.logdebug('%s: switching off, but maintianing the current params: %s', self.name, self.ctrl_status)
            self.state = S_STOP
            self.cnt = 0
        except rospy.ServiceException:
            tb = traceback.format_exc()
            rospy.logerr('%s: controller service error:\n%s', self.name, tb)
            return {'error': 'controller switch failed'}

        return {}

    def cmd_reset(self, **kwargs):
        try:
            self.ctrl_status = CTRL_DISABLED
            self.joy_mode = 0
            self.reload_config()
            self.flag_go = False
            self.flag_altitude = False
            rospy.logdebug('%s: reset the controller: %s', self.name, self.ctrl_status)
            self.state = S_RESET

        except rospy.ServiceException:
            tb = traceback.format_exc()
            rospy.logerr('%s: controller service error:\n%s', self.name, tb)
            return {'error': 'controller switch failed'}

        return {}

    def cmd_stay(self, **kwargs):
        try:
            self.ctrl_status = CTRL_DISABLED
            self.joy_mode = 2
            self.state = S_STAY
            self.cnt = 0
            self.des_pos = self.pos
            rospy.logdebug('%s: swithching to depth keeping: %s', self.name, self.state)
        except rospy.ServiceException:
            tb = traceback.format_exc()
            rospy.logerr('%s: controller service error:\n%s', self.name, tb)
            return {'error': 'controller switch failed'}

        return {}

    def cmd_alt(self, **kwargs):
        try:
            self.ctrl_status = CTRL_ENABLED
            self.flag_altitude = True
            self.state = S_ALT
            self.flag_go = False
            self.offset_altitude = self.altitude
            rospy.logdebug('%s: swithching to station keeping: %s', self.name, self.state)
        except rospy.ServiceException:
            tb = traceback.format_exc()
            rospy.logerr('%s: controller service error:\n%s', self.name, tb)
            return {'error': 'controller switch failed'}

        return {}

    def cmd_go(self, **kwargs):
        try:
            self.ctrl_status = CTRL_ENABLED
            self.flag_go = True
            self.state = S_GO
            rospy.logdebug('%s: swithching to mapping mode: %s', self.name, self.state)
        except rospy.ServiceException:
            tb = traceback.format_exc()
            rospy.logerr('%s: controller service error:\n%s', self.name, tb)
            return {'error': 'controller switch failed'}
        return {}

    def cmd_shake(self, **kwargs):
        try:
            self.ctrl_status = CTRL_ENABLED
            self.flag_shake = True
            self.state = S_SHAKE
            rospy.logdebug('%s: swithching to mapping mode: %s', self.name, self.state)
        except rospy.ServiceException:
            tb = traceback.format_exc()
            rospy.logerr('%s: controller service error:\n%s', self.name, tb)
            return {'error': 'controller switch failed'}
        return {}

    def srv_reload(self, req):
        """This function handle the reload service.
            This will make the pilot reloading the shared configuration on-the-fly from the ROS parameter server.
            Be careful with this approach with the using with the real vehicle cause any error will reflect on the vehicle
            behaviour in terms of control, speeds, and so on ...
        """
        rospy.logwarn('%s: reloading configuration ...', self.name)
        self.reload_config()

        return []

    # disable axis (per service request)
    # def srv_disable_axis(self, data):
    #     if len(data.request) == 6:
    #         self.disable_axis_pilot = np.array(data.request[0:6])
    #     else:
    #         rospy.logwarn('%s: resetting disabled axis', self.name)
    #         self.disable_axis_pilot = np.zeros(6)
    #
    #     rospy.logwarn('%s: set disabled axis mask: %s', self.name, self.disable_axis_pilot)
    #
    #     return True

    def load_alloc_matrix(self):

        # from cad model column1 - bow (front), column2 - stern (back), column 3 - port (left), column 4 -stabord()
        #self.MT = np.array([
        #    [0, 0, 1, 1],
        #    [0, 0, 0, 0],
        #    [1, 1, 0, 0],
        #    [0.0, 0.0, 0.0, 0.0],
        #    [0.279, -0.279, 0.0, 0.0],
        #    [0.0, 0.0, 0.169, -0.169]
        # ])
        self.MT = np.array([
            [self.ctrl_config['x_alloc']['a'],self.ctrl_config['x_alloc']['f'],self.ctrl_config['x_alloc']['l'],self.ctrl_config['x_alloc']['r']],
            [self.ctrl_config['y_alloc']['a'],self.ctrl_config['y_alloc']['f'],self.ctrl_config['y_alloc']['l'],self.ctrl_config['y_alloc']['r']], 
            [self.ctrl_config['z_alloc']['a'],self.ctrl_config['z_alloc']['f'],self.ctrl_config['z_alloc']['l'],self.ctrl_config['z_alloc']['r']],
            [self.ctrl_config['p_alloc']['a'],self.ctrl_config['p_alloc']['f'],self.ctrl_config['p_alloc']['l'],self.ctrl_config['p_alloc']['r']],
            [self.ctrl_config['q_alloc']['a'],self.ctrl_config['q_alloc']['f'],self.ctrl_config['q_alloc']['l'],self.ctrl_config['q_alloc']['r']],
            [self.ctrl_config['r_alloc']['a'],self.ctrl_config['r_alloc']['f'],self.ctrl_config['r_alloc']['l'],self.ctrl_config['r_alloc']['r']],
            ])
        #print (self.MT)   


    def compute_thrust_throttle(self):
        self.load_alloc_matrix()  # load thruster allocation matrix
        self.MT_inv = np.linalg.pinv(self.MT)  # mapp to thrusters
        self.thr = np.dot(self.MT_inv, self.tau)
        self.thr = np.clip(self.thr, -self.thr_limit, self.thr_limit).astype(float)

        # thrust to throttle direct mapping from graph
        # neutral position = 0.5
        # full forward = 1.0
        # full back = 0.0
        # MAX_THROTTLE = 0.5
        # MAX_THRUST = 7.86
        # if motors act different
        #MAX_THRUST_VECT = np.array([3.86, 5.86, 7.86, 7.86]) 
        MAX_THRUST_VECT = np.array([self.ctrl_config['max_force']['Ta'],self.ctrl_config['max_force']['Tf'],self.ctrl_config['max_force']['Tl'],self.ctrl_config['max_force']['Tr']])
        #print (MAX_THRUST_VECT)
        self.throttle = 0.5 + self.thr * MAX_THROTTLE / MAX_THRUST_VECT
        # self.throttle = 0.5 + self.thr*MAX_THROTTLE/MAX_THRUST
        self.throttle = np.clip(self.throttle, 0, 1).astype(float)

    def loop(self):
        # init commands
        print("This is a test")
        self.tau = np.zeros(6)
        #print("X force ", self.force_x)
        # run the low-level control only if enabled explicitly
        if (self.flag_altitude==True):
            rospy.loginfo("Altitude control")
            self.des_pos[2] = self.pos[2] + (self.altitude - self.offset_altitude)

        # simple waypoint switching logic
        self.controller.r_coa = self.r_coa
        self.controller.wp_ind = self.wp_ind
        if np.sqrt((self.des_pos[0]-self.pos[0])**2+(self.des_pos[1]-self.pos[1])**2) <= self.r_coa:
            self.wp_ind = self.wp_ind + 1
            self.des_pos[0] = self.wpx_list[self.wp_ind]
            self.des_pos[1] = self.wpy_list[self.wp_ind]
            self.des_pos_pre[0] = self.wpx_list[self.wp_ind-1]
            self.des_pos_pre[1] = self.wpy_list[self.wp_ind-1]

        self.path_slope = math.atan2(self.des_pos[1]-self.des_pos_pre[1],self.des_pos[0]-self.des_pos_pre[0])
        self.path_rot = np.array([[math.cos(self.path_slope),math.sin(self.path_slope)],[-math.sin(self.path_slope),math.cos(self.path_slope)]])
        #print(self.path_slope,self.path_rot)
        self.controller.des_pos_pre = self.des_pos_pre
        self.controller.path_rot = self.path_rot
        self.controller.path_slope = self.path_slope
        #print (self.wp_ind)

        if (self.ctrl_status == CTRL_ENABLED) and (self.flag_shake == True):
            self.controller.des_pos = self.pos
            self.controller.des_vel = self.vel
            self.controller.ctrl_mode = cc.MODE_POSITION
            # set directly thruster values no controller
            self.shake_cnt = self.shake_cnt + 1
            if (self.shake_cnt<20):    # keep doing the same movement for 1 second
                self.throttle = THROTTLE_SHAKE
                #self.throttle_prev = self.throttle
            elif (self.shake_cnt>=20) and (self.shake_cnt<40):
                self.throttle = 1.0-THROTTLE_SHAKE
                #self.throttle_prev = self.throttle
            else:
		self.throttle = np.array([0.5, 0.5, 0.5, 0.5])
                self.shake_cnt = 0
                self.flag_shake = False
                self.joy_mode = 0
                self.ctrl_status = CTRL_DISABLED
                

            #self.throttle = np.clip(self.throttle, 0, 1).astype(float)
            rospy.loginfo("Mode shake from services")
            rospy.loginfo("Position %s", self.pos)
            rospy.loginfo("Desired position %s", self.pos)
            rospy.loginfo("Forces %s", self.tau)
            rospy.loginfo("Thruster forces %s", self.throttle)

        if (self.ctrl_status == CTRL_ENABLED) and (self.flag_go == False) and (self.flag_shake == False):
            # set controller
            self.controller.des_pos = self.des_pos
            self.controller.des_vel = self.des_vel
            self.controller.ctrl_mode = cc.MODE_POSITION
                #  get computed forces
            self.tau = self.controller.update(self.pos, self.vel)
            #self.tau[0] = self.force_x   # open loop in X
            self.tau[1] = 0.0
            #self.tau[2] = 0.0
            #self.tau[5] = 0.0
            self.compute_thrust_throttle()
            rospy.loginfo("Mode autonomous from services")
            rospy.loginfo("Position %s", self.pos)
            rospy.loginfo("Velocity %s", self.vel)
            rospy.loginfo("Des position %s", self.controller.des_pos)
            rospy.loginfo("Forces %s", self.tau)

##################################################################################################################

        if (self.ctrl_status == CTRL_ENABLED) and (self.flag_go == True) and (self.flag_shake == False):
            self.controller.des_pos = self.des_pos
            self.controller.des_vel = self.des_vel
            self.controller.ctrl_mode = cc.MODE_POSITION
            #cpu_st = time.time()
            self.tau, self.opt_sol = self.controller.mpcupdate(self.pos, self.vel)
            #self.cput = time.time()-cpu_st
            #print self.cput
            #self.tau[0] = self.force_x
            #self.tau[0] = 5.0 #0.0
            self.tau[2] = 0.0
            self.tau[4] = 0.0
            self.tau[5] = -self.tau[5]
            self.compute_thrust_throttle()
            #print(self.opt_sol)
            rospy.loginfo("Mode GO and SURVEY")
            rospy.loginfo("Position %s", self.pos)
            rospy.loginfo("Velocity %s", self.vel)
            rospy.loginfo("Desired position %s", self.controller.des_pos)
            rospy.loginfo("Controller force %s", self.tau)
            rospy.loginfo("Thruter force %s", self.thr)
            rospy.loginfo("Thruster throttle %s", self.throttle)

#################################################################################################################
        if (self.ctrl_status == CTRL_DISABLED):
            if (self.ctrl_status == CTRL_DISABLED) and (self.joy_mode == 2) and (self.cnt < 30):
                # set controller
                self.controller.des_pos = self.des_pos
                self.controller.des_vel = self.vel
                self.controller.ctrl_mode = cc.MODE_POSITION
                #  get computed forces
                self.tau = self.controller.update(self.pos, self.vel)
                self.tau[3] = 0.0
                self.compute_thrust_throttle()
                self.cnt = 0
                rospy.loginfo("Mode station keeping")
                rospy.loginfo("Position %s", self.pos)
                rospy.loginfo("Desired position %s", self.controller.des_pos)
                rospy.loginfo("Controller force %s", self.tau)

            if (self.ctrl_status == CTRL_DISABLED) and (self.joy_mode == 1) and (
                self.cnt < 30):  # depth keeping while all the other DOFs are controlled using joystick
                if (self.disable_axis_user[2] == 1):
                    self.controller.des_pos[2] = self.des_pos[2]
                self.cnt = 0
                self.controller.ctrl_mode = cc.MODE_POSITION
                #  get computed forces
                tau_depth = self.controller.update(self.pos, self.vel)
                self.tau[0] = self.tau_user[0]
                self.tau[1] = self.tau_user[1]
                self.tau[2] = tau_depth[2]
                self.tau[3] = self.tau_user[3]
                self.tau[4] = self.tau_user[4]
                self.tau[5] = self.tau_user[5]
                self.compute_thrust_throttle()
                rospy.loginfo("Mode depth keeping")
                rospy.loginfo("Position %s", self.pos)
                rospy.loginfo("Controller force %s", self.tau)

            if (self.ctrl_status == CTRL_DISABLED) and (self.joy_mode == 0) and (self.cnt < 30):
                self.tau = self.tau_user
                self.tau[3] = 0.0
                rospy.loginfo("Position %s", self.pos)	
                rospy.loginfo("Mode joystick")
                rospy.loginfo("Forces joystick %s", self.tau_user)
                self.compute_thrust_throttle()



            if (self.cnt > 30):
                self.tau = np.zeros(6)
                self.compute_thrust_throttle()
                rospy.loginfo("Lost COMMS with the joystick. Surfacing")
                rospy.loginfo("Position %s", self.pos)


        # send force feedback
        pf = Vector6Stamped()
        pf.header.stamp = rospy.Time.now()
        pf.values = self.tau.tolist()
        self.pub_forces.publish(pf)
        self.send_forces_thr()
        # send MPC solution feedback
        optf = Float64MultiArray()
        optf.data = np.append(self.opt_sol,self.cput)
        #print optf.data
        self.pub_opt_sol.publish(optf)

    def report_status(self, event=None):
        if self.ctrl_status == CTRL_ENABLED:
            print(self.controller)

    def run(self):
        # init pilot
        rospy.loginfo('%s: pilot initialized ...', self.name)

        import time

        # pilot loop
        while not rospy.is_shutdown():
            # t_start = time.time()

            # run main pilot code
            self.loop()
            # rospy.loginfo('%s: pilot runing ...', self.name)
            self.cnt = self.cnt + 1

            # t_end = time.time()
            # rospy.loginfo('%s: loop time: %.3f s', self.name, t_end - t_start)

            try:
                self.pilot_loop.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo('%s shutdown requested ...', self.name)

        # graceful shutdown
        self.ctrl_status = CTRL_DISABLED

    def __str__(self):
        return CONSOLE_STATUS % (
            self.pos, self.vel, self.des_pos, self.des_vel,
            self.tau,
            self.disable_axis
        )


def main():
    rospy.init_node('vehicle_pilot')
    name = rospy.get_name()
    rospy.loginfo('%s initializing ...', name)

    # load global parameters
    rate = int(rospy.get_param('~pilot_rate', DEFAULT_RATE))
    topic_output = rospy.get_param('~topic_output', TOPIC_FRC)
    verbose = bool(rospy.get_param('~verbose', False))

    rate = int(np.clip(rate, 1, 100).astype(int))

    # show current settings
    rospy.loginfo('%s pilot rate: %s Hz', name, rate)
    rospy.loginfo('%s topic output: %s', name, topic_output)

    # start vehicle control node
    pilot = VehicleControl(name, rate, topic_output=topic_output, verbose=verbose)

    try:
        pilot.run()
    except Exception:
        tb = traceback.format_exc()
        rospy.logfatal('%s uncaught exception, dying!\n%s', name, tb)


if __name__ == '__main__':
    main()









