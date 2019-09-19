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
import numpy.linalg as npl

np.set_printoptions(precision=3, suppress=True)

from simulator.util import conversions as cnv

import rospy
import roslib
import tf2_ros as tf2
import tf2_geometry_msgs
import tf

from auv_msgs.msg import NavSts
from std_srvs.srv import Empty, Trigger, TriggerResponse
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
DEFAULT_RATE = 100.0  # pilot loop rate (Hz)
STATUS_RATE = 10.0  # pilot report rate (Hz)
TIME_REPORT = 0.1  # console logging (sec)

# node states
S_STOP = -1
S_RESET = 0
S_START = 1

#STATUS_EKF = {
#    EKF_DISABLED: PilotStatus.PILOT_DISABLED,
#    EKF_ENABLED: PilotStatus.PILOT_ENABLED
#}

# ros topics
TOPIC_NAV = '/vrpn_client_1521236411947584390/estimated_odometry'
TOPIC_EKF = '/sphere_a/nav/pose_estimation'
TOPIC_PRESSURE = '/sphere_a/nav_sensors/pressure_sensor'
TOPIC_IMU = '/sphere_a/nav_sensors/imu'

SRV_START = 'ekf/start'
SRV_STOP = 'ekf/stop'
SRV_RESTART = 'ekf/restart'
SRV_EKF_FULL = 'ekf/ekf_full'
EKF_DISABLED = 0
EKF_ENABLED = 1

# console output
CONSOLE_STATUS = """ekf:
  vel: %s
"""


class EKFStateEstimation(object):
    """StateEstimation class represent the ROS interface for estimating the vehicle 
velocity from vehicle poses using extened Kalman filter.
    """

    def __init__(self, name, ekf_rate, **kwargs):
        self.name = name
        self.ekf_rate = ekf_rate
        self.topic_output = kwargs.get('topic_output', TOPIC_EKF)

        # timing
        self.dt = 1.0 / self.ekf_rate
        self.ekf_loop = rospy.Rate(self.ekf_rate)

        # EKF status
        self.tfBuffer = tf2.Buffer()
        self.listener = tf2.TransformListener(self.tfBuffer)
        self.ekf_status = EKF_DISABLED #EKF_ENABLED #
        self.cnt = 0
        self.t_seq = 0 
        self.t_seq_pre = 0 
        self.pos = np.zeros(6)
        self.vel = np.zeros(6)
        self.vel_EKF = np.zeros(6)
        self.pos_EKF = np.zeros(6)

        # ros interface
        self.sub_depth = rospy.Subscriber(TOPIC_PRESSURE, PressureSensor, self.handle_depth, tcp_nodelay=True,
                                          queue_size=1)
        self.sub_imu = rospy.Subscriber(TOPIC_IMU, Imu, self.handle_imu, tcp_nodelay=True, queue_size=1)
        self.sub_nav = rospy.Subscriber(TOPIC_NAV, Odometry, self.handle_nav, tcp_nodelay=True, queue_size=1)
        self.pub_ekf = rospy.Publisher(TOPIC_EKF, Odometry, tcp_nodelay=True, queue_size=1)

       # ros service for command ekf
        self.s_start = rospy.Service(SRV_START, Trigger, self.srv_start_ekf)
        self.s_stop = rospy.Service(SRV_STOP, Trigger, self.srv_stop_ekf)
        self.s_restart = rospy.Service(SRV_RESTART, Trigger, self.srv_restart_ekf)


        # init params for EKF ##################################################
        ########################################################################
        ########################################################################
        self.st_size = 12; self.pose_size = 6; self.vel_size = 6 # dimension of the system
        self.Q = 0.01**2*np.eye(self.st_size) # motion noise
        self.R = 0.01**2*np.eye(self.pose_size) # observation noise
        self.use_u = 0 # use or not use acceleration 
        #self.dt_ratio = 50 # threshold for deciding the data lost
        #self.dt_len = 1 # nominal time step (averaging with the time)
        self.ini_sig = 0.1 # initial niose covariance  
        self.mu = np.zeros((self.st_size,1)) 
        self.Sigma = self.ini_sig*np.eye(self.st_size)
        self.vel_est = np.zeros(6)
        #self.dt_ub = 0.01

        # Compute the Jacobian of the system
        self.sys_w = np.eye(self.st_size) # Jacobian for motion noise
        self.sys_z = np.concatenate((np.eye(self.pose_size),np.zeros((self.pose_size,self.vel_size))),axis=1) # Jacobian for observation model
        self.sys_v = np.eye(self.pose_size) # Jacobian for observastion noise

        # Model parameter for estimating the acceleration
        #self.Xu = 11.686; self.Yv = 21.645; self.Nr = 0.158; self.Zw = 21.645; self.Kp = 0.158; self.Mq = 0.158;
        self.Xu = 48.17; self.Yv = 4.11; self.Nr = 4.11; self.Zw = 4.11; self.Kp = 48.17; self.Mq = 4.11;
        self.D_back = 0.1694; self.m = 20.42; self.L = 1; self.rad = 0.1; self.rho_water = 1025;
        self.xg = 0; self.yg = 0; self.xb = 0; self.yb = 0; self.zb = -0.009; self.zg = 0.00178;
        self.W = 200.116; self.B = 201.586; self.Ixx = 0.12052; self.Iyy = 0.943099; self.Izz = 1.006087; 
        self.Xu_dot = - 0.1 * self.m; 
        self.Yv_dot = - np.pi * self.rho_water * self.rad**2 * self.L;
        self.Zw_dot = - np.pi * self.rho_water * self.rad**2 * self.L;
        self.Kp_dot = - np.pi * self.rho_water * self.rad**4 * 0.25;
        self.Mq_dot = - np.pi * self.rho_water * self.rad**2 * self.L**3 /12;
        self.Nr_dot = - np.pi * self.rho_water * self.rad**2 * self.L**3 /12; 

    # Normalize the angle to -pi to pi
    def wraptopi(self,x):
        angle_rad = x - 2*np.pi*np.floor((x+np.pi)/(2*np.pi))
        return angle_rad

    # Compute the input from thruster command
    def thrust2accel(self,mu,Tsum,Tsub):
        T_surge = Tsum; M_yaw = Tsub
        uu = mu[6]; v = mu[7]; w = mu[8]; p = mu[9]; q = mu[10]; r = mu[11]
        phi = mu[3]; theta = mu[4]; psi = mu[5]; input_u = np.zeros((len(mu),1))
        input_u[6] = (v*r*self.m - self.Xu*np.abs(uu)*uu+ T_surge)/(self.m-self.Xu_dot)
        input_u[7] = (- uu*r*self.m - self.Yv*np.abs(v)*v)/(self.m-self.Yv_dot)
        input_u[11] = (- self.Nr*np.abs(r)*r + M_yaw)/(self.Izz-self.Nr_dot)
        return input_u

    # Compute the rotation matrix from current state
    def euler2rotation(self,phi,theta,psi):
        R_pos = [[np.cos(psi)*np.cos(theta),-np.sin(psi)*np.cos(phi)+np.cos(psi)*np.sin(theta)*np.sin(phi),np.sin(psi)*np.sin(phi)+np.cos(psi)*np.sin(theta)*np.cos(phi)],
                 [np.sin(psi)*np.cos(theta),np.cos(psi)*np.cos(phi)+np.sin(psi)*np.sin(theta)*np.sin(phi),-np.cos(psi)*np.sin(phi)+np.sin(psi)*np.sin(theta)*np.cos(phi)], 
                 [-np.sin(theta),np.cos(theta)*np.sin(phi),np.cos(theta)*np.cos(phi)]]
        R_ori = [[1.0,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)], 
                 [0,np.cos(phi),-np.sin(phi)], 
                 [0,np.sin(phi)/np.cos(theta),np.cos(phi)/np.cos(theta)]]
        return R_pos,R_ori

    # Analytic Jacobian for motion model
    def Jacobian_sys_f(self,mu,R_pos,R_ori):
        phi = mu[3][0]; theta = mu[4][0]; psi = mu[5][0]; u = mu[6][0]; v = mu[7][0]; w = mu[8][0]; p = mu[9][0]; q = mu[10][0]; r = mu[11][0];
        R2 = R_pos
        R4 = R_ori
        R1 = [[(-np.sin(psi)*np.sin(phi)+np.cos(psi)*np.sin(theta)*np.cos(phi))*v+(np.sin(psi)*np.cos(phi)-np.cos(psi)*np.sin(theta)*np.sin(phi))*w,-np.cos(psi)*np.sin(theta)*u+np.cos(psi)*np.cos(theta)*np.sin(phi)*v+np.sin(psi)*np.cos(theta)*np.cos(phi)*w,-np.sin(psi)*np.cos(theta)*u+(-np.cos(psi)*np.cos(phi)-np.sin(psi)*np.sin(theta)*np.sin(phi))*v+(np.cos(psi)*np.sin(phi)-np.sin(psi)*np.sin(theta)*np.cos(phi))*w],
             [(-np.cos(psi)*np.sin(phi)+np.sin(psi)*np.sin(theta)*np.cos(phi))*v+(-np.cos(psi)*np.cos(phi)-np.sin(psi)*np.sin(theta)*np.sin(phi))*w,-np.sin(psi)*np.sin(theta)*u+np.sin(psi)*np.cos(theta)*np.sin(phi)*v+np.sin(psi)*np.cos(theta)*np.cos(phi)*w,-np.cos(psi)*np.cos(theta)*u+(-np.sin(psi)*np.cos(phi)+np.cos(psi)*np.sin(theta)*np.sin(phi))*v+(np.sin(psi)*np.sin(phi)+np.cos(psi)*np.sin(theta)*np.cos(phi))*w],
             [np.cos(theta)*np.cos(phi)*v-np.cos(theta)*np.sin(phi)*w,-np.cos(theta)*u-np.sin(theta)*np.sin(phi)*v-np.sin(theta)*np.cos(phi)*w,0]]
        R3 = [[np.cos(phi)*np.tan(theta)*q-np.sin(phi)*np.tan(theta)*r,np.sin(phi)*q/((np.cos(theta))**2.0)+np.cos(phi)*r/((np.cos(theta))**2.0),0],
             [-np.sin(phi)*q-np.cos(phi)*r,0,0],
             [np.cos(phi)*q/np.cos(theta)-np.sin(phi)*r/(np.cos(theta)),np.sin(phi)*np.sin(theta)*q/((np.cos(theta))**2.0)+np.cos(phi)*np.sin(theta)*r/((np.cos(theta))**2.0),0]]
        F1 = np.concatenate((np.zeros((3,3)),R1,R2,np.zeros((3,3))),axis=1)
        F2 = np.concatenate((np.zeros((3,3)),R3,np.zeros((3,3)),R4),axis=1)
        F = np.concatenate((F1,F2,np.zeros((6,12))),axis=0)
        return F

    def ekf_prediction(self,mu,Sigma,dt,Tsum,Tsub): 
        # EKF prediction
        R_pos,R_ori = self.euler2rotation(mu[3][0],mu[4][0],mu[5][0]) # rotational matrix
        R_rb = np.concatenate((np.concatenate((R_pos,np.zeros((3,3))),axis=1), np.concatenate((np.zeros((3,3)),R_ori),axis=1)), axis=0)
        if self.use_u == 1:
            input_u = self.thrust2accel(mu,Tsum,Tsub) # convert the command to input
            dx = np.dot(np.concatenate((np.concatenate((np.zeros((6,6)),R_rb),axis=1), np.zeros((6,12))),axis=0),mu) + input_u 
        else:
            dx = np.dot(np.concatenate((np.concatenate((np.zeros((6,6)),R_rb),axis=1), np.zeros((6,12))),axis=0),mu) 
        predMu = mu + dx*dt # predicted mean 
        predMu[3] = self.wraptopi(predMu[3]); predMu[4] = self.wraptopi(predMu[4]); predMu[5] = self.wraptopi(predMu[5]) # normalize the angle
        F = self.Jacobian_sys_f(mu,R_pos,R_ori)*dt + np.eye(self.st_size) # compute Jacobian of motion model based on the current state
        W = self.sys_w
        predSigma = np.dot(np.dot(F,Sigma),np.transpose(F)) + np.dot(np.dot(W,self.Q),np.transpose(W)); # predicted covariance
        return predMu, predSigma

    def ekf_correction(self,predMu,predSigma,z): 
        # EKF correction
        H = self.sys_z
        zhat = np.dot(H,predMu) # predicted observation
        nu = z-zhat # innovation
        nu[3] = self.wraptopi(nu[3]); nu[4] = self.wraptopi(nu[4]); nu[5] = self.wraptopi(nu[5]) # normalize the angle
        S = np.dot(np.dot(H,predSigma),np.transpose(H)) + self.R
        S_size = len(S[0])
        K = np.dot(np.dot(predSigma,np.transpose(H)),np.dot(npl.pinv(S),np.eye(S_size))) # Kalman gain
        mu = predMu + np.dot(K,nu) # Corrected mean
        I = np.eye(len(mu))
        Sigma = np.dot(np.dot((I - np.dot(K,H)),predSigma),np.transpose((I - np.dot(K,H)))) + np.dot(np.dot(K,self.R),np.transpose(K)) # Corrected covariance
        return mu, Sigma
        ########################################################################
        ########################################################################


    def handle_depth(self, data):
        self.depth = data.depth
        #self.pos[2] = self.depth

    def handle_imu(self, data):
        dt_int = 0.0001
        quaternion = (data.orientation.x,
                      data.orientation.y,
                      data.orientation.z,
                      data.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        #self.roll = euler[0]
        #self.pitch = euler[1]
        #self.yaw = euler[2]

        #self.pos_IMU[3] = self.pitch 
        #self.pos_IMU[4] = self.roll  
        #self.pos_IMU[5] = self.yaw  
        #self.pos_IMU[3:6] = cnv.wrap_pi(self.pos_IMU[3:6])


    def handle_nav(self, data):
        # parse navigation data
        quaternion = (data.pose.pose.orientation.x,
                      data.pose.pose.orientation.y,
                      data.pose.pose.orientation.z,
                      data.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.t_seq = data.header.seq
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

        self.vel = np.array([
            data.twist.twist.linear.x,
            data.twist.twist.linear.y,
            data.twist.twist.linear.z,
            data.twist.twist.angular.x,
            data.twist.twist.angular.y,
            data.twist.twist.angular.z
        ])


    def send_ekf(self):
        es = Odometry()
        es.header.stamp = rospy.Time.now()
        quaternion = tf.transformations.quaternion_from_euler(self.pos_EKF[3], self.pos_EKF[4], self.pos_EKF[5])
        es.pose.pose.orientation.x = quaternion[0]
        es.pose.pose.orientation.y = quaternion[1]
        es.pose.pose.orientation.z = quaternion[2]
        es.pose.pose.orientation.w = quaternion[3]
        es.pose.pose.position.x = self.pos_EKF[0]
        es.pose.pose.position.y = self.pos_EKF[1]
        es.pose.pose.position.z = self.pos_EKF[2]
        es.twist.twist.linear.x = self.vel_EKF[0]
        es.twist.twist.linear.y = self.vel_EKF[1]
        es.twist.twist.linear.z = self.vel_EKF[2]
        es.twist.twist.angular.x = self.vel_EKF[3]
        es.twist.twist.angular.y = self.vel_EKF[4]
        es.twist.twist.angular.z = self.vel_EKF[5]
 
        self.pub_ekf.publish(es)


    def srv_start_ekf(self,request):
        # enable ekf
        self.ekf_status = EKF_ENABLED
        return TriggerResponse(success=True,message="start ekf successfully")

    def srv_stop_ekf(self,request):
        # stop ekf
        self.ekf_status = EKF_DISABLED
        return TriggerResponse(success=True,message="stop ekf successfully")

    def srv_restart_ekf(self,request):
        # restart
        if (self.ekf_status == EKF_ENABLED):
            self.ekf_status = EKF_DISABLED
            rospy.sleep(2)
            self.ekf_status = EKF_ENABLED
        return TriggerResponse(success=True,message="restart ekf successfully")

    def loop(self):
        # thruster command
        self.tau_prev = np.zeros(6)

        if self.ekf_status == EKF_ENABLED:
                z = np.reshape(self.pos,(6,1))
                self.mu, self.Sigma = self.ekf_prediction(self.mu,self.Sigma,self.dt,self.tau_prev[0],self.tau_prev[5])
                # Perform the ekf correction only when having new OptiiTrack data 
                if self.t_seq > self.t_seq_pre:
                    self.mu, self.Sigma = self.ekf_correction(self.mu,self.Sigma,z)
                    #print self.t_seq, self.t_seq_pre
                    self.t_seq_pre = self.t_seq

                self.vel_EKF = [self.mu[6][0],self.mu[7][0],self.mu[8][0],self.mu[9][0],self.mu[10][0],self.mu[11][0]]
                self.pos_EKF = [self.mu[0][0],self.mu[1][0],self.mu[2][0],self.mu[3][0],self.mu[4][0],self.mu[5][0]]
                rospy.loginfo("EKF for state estimation")
                rospy.loginfo("OptiTrack Position %s", self.pos)
                rospy.loginfo("Estimated position %s", self.pos_EKF)
                rospy.loginfo("Estimated velocity %s", self.vel_EKF)

        if self.ekf_status == EKF_DISABLED:
                self.mu = np.concatenate((np.reshape(self.pos,(6,1)),np.zeros((6,1))),axis=0)
                self.Sigma = self.ini_sig*np.eye(self.st_size)
                self.vel_EKF = np.zeros(6)
                self.cnt = 0
                rospy.loginfo("No ekf estimation")
                rospy.loginfo("Position %s", self.pos)

        # send back state estimate
        self.send_ekf()

    def run(self):
        # init pilot
        rospy.loginfo('%s: ekf initialized ...', self.name)

        # pilot loop
        while not rospy.is_shutdown():
            # run main pilot code
            self.loop()
            # rospy.loginfo('%s: pilot runing ...', self.name)
            self.cnt = self.cnt + 1
            rospy.loginfo("In the EKF loop ####################")

            try:
                self.ekf_loop.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo('%s shutdown requested ...', self.name)

        # graceful shutdown
        self.ekf_status = EKF_DISABLED

    def __str__(self):
        return CONSOLE_STATUS % (
            self.vel_EKF
        )


def main():
    rospy.init_node('nav_ekf')
    name = rospy.get_name()
    rospy.loginfo('%s initializing ...', name)

    # load global parameters
    rate = int(rospy.get_param('~ekf_rate', DEFAULT_RATE))
    topic_output = rospy.get_param('~topic_output', TOPIC_EKF)
    verbose = bool(rospy.get_param('~verbose', False))

    rate = int(np.clip(rate, 1, 100).astype(int))

    # show current settings
    rospy.loginfo('%s ekf rate: %s Hz', name, rate)
    rospy.loginfo('%s topic output: %s', name, topic_output)

    # start vehicle control node
    EKF = EKFStateEstimation(name, rate, topic_output=topic_output, verbose=verbose)

    try:
        EKF.run()
    except Exception:
        tb = traceback.format_exc()
        rospy.logfatal('%s uncaught exception, dying!\n%s', name, tb)


if __name__ == '__main__':
    main()









