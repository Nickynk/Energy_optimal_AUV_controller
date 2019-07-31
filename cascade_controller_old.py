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
#  Modified 2018, DROP Lab, university of Michigan, USA
#  Author:
#     Corina Barbalata, Eduardo Iscar, Atulya Shree

from __future__ import division

import numpy as np
#import libraries.mpctools as mpc
import rospy
np.set_printoptions(precision=3, suppress=True)
import casadi as ca
import math
import time

from simulator.model import vehicle_model as vm
from simulator.model import mathematical_model as mm
from simulator.util import conversions as cnv


# controller modes
MODE_POSITION = 0
MODE_VELOCITY = 1
MODE_STATION = 2

CONSOLE_STATUS = """controller:
  pos: %s
  des_p: %s
  vel: %s
  des_v: %s
  mode: %s
"""

class Controller(object):
    """Controller class wraps the low-level controllers for velocity and position.
    This class can be used as a parent class used to describe the behaviour of a different vehicle controllers.
    """

    def __init__(self, dt, config, **kwargs):
        # config
        self.dt = dt
        self.config = config

        # mode
        self.ctrl_mode = MODE_POSITION

        # states
        self.pos = np.zeros(6)
        self.vel = np.zeros(6)

        # requests
        self.des_pos = np.zeros(6)
        self.des_vel = np.zeros(6)

        # limits
        self.lim_vel = kwargs.get('lim_vel', 10 * np.ones(6))

    def update_config(self, ctrl_config, model_config):
        pass

    def update(self, position, velocity):
        return np.zeros(6)

    def __str__(self):
        return CONSOLE_STATUS % (
            self.pos, self.des_pos, self.vel, self.des_vel, self.ctrl_mode
        )

class PIDController(Controller):
    """PID controller implemnts the control structure for position.
           The output of this class is the force necessary for the 6 DOFs to move the vehicle so that the required
           characteristics are achieved.
        """

    def __init__(self, dt, ctrl_config, model_config, **kwargs):
        super(PIDController, self).__init__(dt, ctrl_config, **kwargs)

        # init params
        # position gains
        self.pos_Kp = np.zeros(6)
        self.pos_Kd = np.zeros(6)
        self.pos_Ki = np.zeros(6)

        # controller limits
        self.pos_lim = np.zeros(6)
        self.tau = np.zeros(6)

        # errors
        self.err_pos = np.zeros(6)
        self.err_pos_prev = np.zeros(6)
        self.err_pos_der = np.zeros(6)
        self.err_pos_int = np.zeros(6)

        # init jacobians matrices
        self.J = np.zeros((6, 6))  # jacobian matrix (translate velocity from body referenced to Earth referenced)
        self.J_inv = np.zeros((6, 6))  # inverse jacobian matrix

    def update_config(self, ctrl_config, model_config):
        # pid parameters (position)
        self.pos_Kp = np.array([
            ctrl_config['pos_x']['kp'],
            ctrl_config['pos_y']['kp'],
            ctrl_config['pos_z']['kp'],
            ctrl_config['pos_k']['kp'],
            ctrl_config['pos_m']['kp'],
            ctrl_config['pos_n']['kp'],
        ])

        self.pos_Kd = np.array([
            ctrl_config['pos_x']['kd'],
            ctrl_config['pos_y']['kd'],
            ctrl_config['pos_z']['kd'],
            ctrl_config['pos_k']['kd'],
            ctrl_config['pos_m']['kd'],
            ctrl_config['pos_n']['kd'],
        ])

        self.pos_Ki = np.array([
            ctrl_config['pos_x']['ki'],
            ctrl_config['pos_y']['ki'],
            ctrl_config['pos_z']['ki'],
            ctrl_config['pos_k']['ki'],
            ctrl_config['pos_m']['ki'],
            ctrl_config['pos_n']['ki'],
        ])

        self.pos_lim = np.array([
            ctrl_config['pos_x']['lim'],
            ctrl_config['pos_y']['lim'],
            ctrl_config['pos_z']['lim'],
            ctrl_config['pos_k']['lim'],
            ctrl_config['pos_m']['lim'],
            ctrl_config['pos_n']['lim'],
        ])


    def update(self, position, velocity):
        # store nav updates
        self.pos = position
        self.vel = velocity

        # update jacobians
        self.J = mm.update_jacobian(self.J, self.pos[3], self.pos[4], self.pos[5])
        self.J_inv = np.linalg.pinv(self.J)

        # model-free pid cascaded controller
        #   first pid (outer loop on position)
        self.err_pos = self.pos - self.des_pos
        # self.err_pos = np.dot(self.J_inv, self.err_pos.reshape((6, 1))).flatten()
        # wrap angles and limit pitch
        self.err_pos[3:6] = cnv.wrap_pi(self.err_pos[3:6])

        # update errors
        self.err_pos_der = (self.err_pos - self.err_pos_prev) / self.dt
        self.err_pos_int = np.clip(self.err_pos_int + self.err_pos, -self.pos_lim, self.pos_lim)

        # Position integral terms set to zero to avoid oscillations
        # pos_changed = np.sign(self.err_pos) != np.sign(self.err_pos_prev)
        # pos_changed[2] = False
        # self.err_pos_int[pos_changed] = 0.0

        self.err_pos_prev = self.err_pos

        # first pid output (plus speed limits if requested by the user)
        self.tau = (-self.pos_Kp * self.err_pos) + (-self.pos_Kd * self.err_pos_der) + ( -self.pos_Ki * self.err_pos_int)

        # these are set to zero only for depth control testing -- comment the following 5 lines if full control is active
        # self.tau[4] = 0.0
        # self.tau[0] = 0.0
        # self.tau[1] = 0.0
        # self.tau[3] = 0.0
        # self.tau[5] = 0.0

        return self.tau

    def __str__(self):
        return """%s
                 ep: %s
                 ed: %s
                 ei: %s
                 tau_c: %s
               """ % (
            super(PIDController, self).__str__(),
            self.err_pos, self.err_pos_der, self.err_pos_int, self.tau
        )


class CascadeController(Controller):
    """CascadeController implements a cascaded PID controllers for position and velocity.
       The output of this class is the force necessary for the 6 DOFs to move the vehicle so that the required
       characteristics are achieved.
    """

    def __init__(self, dt, ctrl_config, model_config, **kwargs):
        super(CascadeController, self).__init__(dt, ctrl_config, **kwargs)

        # init params
        # position gains
        self.pos_Kp = np.zeros(6)
        self.pos_Kd = np.zeros(6)
        self.pos_Ki = np.zeros(6)

        # velocity gains
        self.vel_Kp = np.zeros(6)
        self.vel_Kd = np.zeros(6)
        self.vel_Ki = np.zeros(6)

        # controller limits
        self.pos_lim = np.zeros(6)
        self.vel_lim = np.zeros(6)

        self.req_vel = np.zeros(6)
        self.tau_ctrl = np.zeros(6)
        self.tau_prev = np.zeros(6)

        # errors
        self.err_pos = np.zeros(6)
        self.err_pos_prev = np.zeros(6)
        self.err_pos_der = np.zeros(6)
        self.err_pos_int = np.zeros(6)
        self.err_vel = np.zeros(6)
        self.err_vel_prev = np.zeros(6)
        self.err_vel_der = np.zeros(6)
        self.err_vel_int = np.zeros(6)

        # init jacobians matrices
        self.J = np.zeros((6, 6))  # jacobian matrix (translate velocity from body referenced to Earth referenced)
        self.J_inv = np.zeros((6, 6))  # inverse jacobian matrix

    def update_config(self, ctrl_config, model_config):
        # main control loop

        # pid parameters (position)
        self.pos_Kp = np.array([
                ctrl_config['pos_x']['kp'],
                ctrl_config['pos_y']['kp'],
                ctrl_config['pos_z']['kp'],
                ctrl_config['pos_k']['kp'],
                ctrl_config['pos_m']['kp'],
                ctrl_config['pos_n']['kp'],
        ])

        self.pos_Kd = np.array([
                ctrl_config['pos_x']['kd'],
                ctrl_config['pos_y']['kd'],
                ctrl_config['pos_z']['kd'],
                ctrl_config['pos_k']['kd'],
                ctrl_config['pos_m']['kd'],
                ctrl_config['pos_n']['kd'],
        ])

        self.pos_Ki = np.array([
                ctrl_config['pos_x']['ki'],
                ctrl_config['pos_y']['ki'],
                ctrl_config['pos_z']['ki'],
                ctrl_config['pos_k']['ki'],
                ctrl_config['pos_m']['ki'],
                ctrl_config['pos_n']['ki'],
        ])

        self.pos_lim = np.array([
                ctrl_config['pos_x']['lim'],
                ctrl_config['pos_y']['lim'],
                ctrl_config['pos_z']['lim'],
                ctrl_config['pos_k']['lim'],
                ctrl_config['pos_m']['lim'],
                ctrl_config['pos_n']['lim'],
        ])

        # pid parameters (velocity)
        self.vel_Kp = np.array([
                ctrl_config['vel_u']['kp'],
                ctrl_config['vel_v']['kp'],
                ctrl_config['vel_w']['kp'],
                ctrl_config['vel_p']['kp'],
                ctrl_config['vel_q']['kp'],
                ctrl_config['vel_r']['kp'],
        ])

        self.vel_Kd = np.array([
                ctrl_config['vel_u']['kd'],
                ctrl_config['vel_v']['kd'],
                ctrl_config['vel_w']['kd'],
                ctrl_config['vel_p']['kd'],
                ctrl_config['vel_q']['kd'],
                ctrl_config['vel_r']['kd'],
        ])

        self.vel_Ki = np.array([
                ctrl_config['vel_u']['ki'],
                ctrl_config['vel_v']['ki'],
                ctrl_config['vel_w']['ki'],
                ctrl_config['vel_p']['ki'],
                ctrl_config['vel_q']['ki'],
                ctrl_config['vel_r']['ki'],
        ])

        self.vel_lim = np.array([
                ctrl_config['vel_u']['lim'],
                ctrl_config['vel_v']['lim'],
                ctrl_config['vel_w']['lim'],
                ctrl_config['vel_p']['lim'],
                ctrl_config['vel_q']['lim'],
                ctrl_config['vel_r']['lim'],
        ])

        self.vel_input_lim = np.array([
                ctrl_config['vel_u']['input_lim'],
                ctrl_config['vel_v']['input_lim'],
                ctrl_config['vel_w']['input_lim'],
                ctrl_config['vel_p']['input_lim'],
                ctrl_config['vel_q']['input_lim'],
                ctrl_config['vel_r']['input_lim'],
        ])


    def update(self, position, velocity):
        # main control loop

        # store nav updates
        self.pos = position
        self.vel = velocity

        # update jacobians
        self.J = mm.update_jacobian(self.J, self.pos[3], self.pos[4], self.pos[5])
        self.J_inv = np.linalg.pinv(self.J)

        # model-free pid cascaded controller
        #   first pid (outer loop on position)
        self.err_pos = self.pos - self.des_pos
        self.err_pos = np.dot(self.J_inv, self.err_pos.reshape((6, 1))).flatten()
        # wrap angles and limit pitch
        self.err_pos[3:6] = cnv.wrap_pi(self.err_pos[3:6])

        # update errors
        self.err_pos_der = (self.err_pos - self.err_pos_prev) / self.dt
        self.err_pos_int = np.clip(self.err_pos_int + self.err_pos, -self.pos_lim, self.pos_lim)

        # Position integral terms set to zero to avoid oscillations
        pos_changed = np.sign(self.err_pos) != np.sign(self.err_pos_prev)
        pos_changed[2] = False
        self.err_pos_int[pos_changed] = 0.0

        # update previous error
        self.err_pos_prev = self.err_pos

        # first pid output (plus speed limits if requested by the user)
        self.req_vel = (-self.pos_Kp * self.err_pos) + (-self.pos_Kd * self.err_pos_der) + (-self.pos_Ki * self.err_pos_int)


        # if running in velocity mode ignore the first pid
        if self.ctrl_mode == MODE_VELOCITY:
            self.req_vel = self.des_vel

        # apply user velocity limits (if any)
        self.req_vel = np.clip(self.req_vel, -self.lim_vel, self.lim_vel)

        # model-free pid cascaded controller
        #   second pid (inner loop on velocity)
        self.err_vel = np.clip(self.vel - self.req_vel, -self.vel_input_lim, self.vel_input_lim)
        self.err_vel_der = (self.err_vel - self.err_vel_prev) / self.dt
        self.err_vel_int = np.clip(self.err_vel_int + self.err_vel, -self.vel_lim, self.vel_lim)

        # velocity integral terms set to zero to avoid oscillations
        # vel_changed = np.sign(self.err_vel) != np.sign(self.err_vel_prev)
        # vel_changed[2] = False
        # self.err_vel_int[vel_changed] = 0.0

        # update previous error
        self.err_vel_prev = self.err_vel

        # second pid output
        self.tau = (-self.vel_Kp * self.err_vel) + (-self.vel_Kd * self.err_vel_der) + (-self.vel_Ki * self.err_vel_int)
        # self.tau[1] = 0.0
        # self.tau[3] = 0.0


        self.tau_prev = self.tau

        return self.tau

    def __str__(self):

        return """%s
             req_v: %s
             lim_v: %s
             ep: %s
             ed: %s
             ei: %s
             evp: %s
             evd: %s
             evi: %s
             tau_c: %s
           """ % (
            super(CascadeController, self).__str__(),
            self.req_vel, self.lim_vel,
            self.err_pos, self.err_pos_der, self.err_pos_int,
            self.err_vel, self.err_vel_der, self.err_vel_int,
            self.tau_ctrl
        )



class AutoTuneController(CascadeController):
    """AutoTuneController implements an auto tuning controllers for position and velocity.
       The output of this class is the force necessary for the 6 DOFs to move the vehicle so that the required
       characteristics are achieved.
    """

    def __init__(self, dt, ctrl_config, model_config, **kwargs):
        super(AutoTuneController, self).__init__(dt, ctrl_config, model_config, **kwargs)

        # init params
        # adaption coefficients for each DOF of vehicle
        self.adapt_coeff_pos = np.zeros(6)  # position
        self.adapt_coeff_vel = np.zeros(6)  # velocity
        self.adapt_limit_pos = np.zeros(3)
        self.adapt_limit_vel = np.zeros(3)
        self.pitch_surge_coeff = 0.0
        self.pitch_rest_coeff = 0.0
        self.tau_ctrl_prev = np.zeros(6)

    def update_config(self, ctrl_config, model_config):
        # load parameters from default controller
        super(AutoTuneController, self).update_config(ctrl_config, model_config)

        # adaptation coefficients
        self.adapt_coeff_pos = np.array([
            ctrl_config['adapt_coeff_pos']['x'],
            ctrl_config['adapt_coeff_pos']['y'],
            ctrl_config['adapt_coeff_pos']['z'],
            ctrl_config['adapt_coeff_pos']['k'],
            ctrl_config['adapt_coeff_pos']['m'],
            ctrl_config['adapt_coeff_pos']['n'],
        ])

        self.adapt_coeff_vel = np.array([
            ctrl_config['adapt_coeff_vel']['u'],
            ctrl_config['adapt_coeff_vel']['v'],
            ctrl_config['adapt_coeff_vel']['w'],
            ctrl_config['adapt_coeff_vel']['p'],
            ctrl_config['adapt_coeff_vel']['q'],
            ctrl_config['adapt_coeff_vel']['r'],
        ])

        self.adapt_limit_pos = np.array([
            ctrl_config['adapt_limit_pos']['p'],
            ctrl_config['adapt_limit_pos']['i'],
            ctrl_config['adapt_limit_pos']['d']
        ])

        self.adapt_limit_vel = np.array([
            ctrl_config['adapt_limit_vel']['p'],
            ctrl_config['adapt_limit_vel']['i'],
            ctrl_config['adapt_limit_vel']['d']
        ])

        # pitch controller parameters
        self.pitch_surge_coeff = float(ctrl_config.get('pitch_surge_coeff', 0.0))
        self.pitch_rest_coeff = float(ctrl_config.get('pitch_rest_coeff', 0.0))

    def update(self, position, velocity):
        # store nav updates
        self.pos = position
        self.vel = velocity

        # update jacobians
        self.J = mm.update_jacobian(self.J, self.pos[3], self.pos[4], self.pos[5])
        self.J_inv = np.linalg.pinv(self.J)

        #   first pid (outer loop on position)
        self.err_pos = self.pos - self.des_pos

        # wrap angles and limit pitch
        self.err_pos[3:6] = cnv.wrap_pi(self.err_pos[3:6])

        # update errors
        self.err_pos_der = (self.err_pos - self.err_pos_prev) / self.dt
        self.err_pos_int = np.clip(self.err_pos_int + self.err_pos, -self.pos_lim, self.pos_lim)

        # adaptive tuning of position gains
        self.pos_Kp += self.adapt_coeff_pos * self.err_pos * np.abs(self.err_pos)
        self.pos_Ki += self.adapt_coeff_pos * self.err_pos * self.err_pos_int
        self.pos_Kd += self.adapt_coeff_pos * self.err_pos * self.err_pos_der

        self.pos_Kp = np.clip(self.pos_Kp, -self.adapt_limit_pos[0], self.adapt_limit_pos[0])
        self.pos_Ki = np.clip(self.pos_Ki, -self.adapt_limit_pos[1], self.adapt_limit_pos[1])
        self.pos_Kd = np.clip(self.pos_Kd, -self.adapt_limit_pos[2], self.adapt_limit_pos[2])

        # update previous error
        self.err_pos_prev = self.err_pos

        # Position integral terms set to zero to avoid oscillations
        pos_changed = np.sign(self.err_pos) != np.sign(self.err_pos_prev)
        pos_changed[2] = False
        self.err_pos_int[pos_changed] = 0.0

        # first pid output (plus speed limits if requested by the user)
        self.req_vel = (-self.pos_Kp * self.err_pos) + (-self.pos_Kd * self.err_pos_der) + (-self.pos_Ki * self.err_pos_int)

        # if running in velocity mode ignore the first pid
        if self.ctrl_mode == MODE_VELOCITY:
            self.req_vel = self.des_vel

        # apply user velocity limits (if any)
        self.req_vel = np.clip(self.req_vel, -self.lim_vel, self.lim_vel)

        #  second pid (inner loop on velocity)
        self.err_vel = np.clip(self.vel - self.req_vel, -self.vel_input_lim, self.vel_input_lim)
        self.err_vel_der = (self.err_vel - self.err_vel_prev) / self.dt
        self.err_vel_int = np.clip(self.err_vel_int + self.err_vel, -self.vel_lim, self.vel_lim)

        # velocity integral terms set to zero to avoid oscillations
        vel_changed = np.sign(self.err_vel) != np.sign(self.err_vel_prev)
        vel_changed[2] = False
        self.err_vel_int[vel_changed] = 0.0

        # update previous error
        self.err_vel_prev = self.err_vel

        # adaptive tuning of velocity gains
        self.vel_Kp += self.adapt_coeff_vel * self.err_vel * np.abs(self.err_vel)
        self.vel_Ki += self.adapt_coeff_vel * self.err_vel * self.err_vel_int
        self.vel_Kd += self.adapt_coeff_vel * self.err_vel * self.err_vel_der

        self.vel_Kp = np.clip(self.vel_Kp, -self.adapt_limit_vel[0], self.adapt_limit_vel[0])
        self.vel_Ki = np.clip(self.vel_Ki, -self.adapt_limit_vel[1], self.adapt_limit_vel[1])
        self.vel_Kd = np.clip(self.vel_Kd, -self.adapt_limit_vel[2], self.adapt_limit_vel[2])

        # Velocity integral terms set to zero to avoid oscillations
        vel_changed = np.sign(self.err_vel) != np.sign(self.err_vel_prev)
        vel_changed[2] = False  # ignore the depth

        self.err_vel_int[vel_changed] = 0.0

        # second pid output
        self.tau = (-self.vel_Kp * self.err_vel) + (-self.vel_Kd * self.err_vel_der) + (-self.vel_Ki * self.err_vel_int)
	print "Force in yaw ", self.tau[5], " error ",self.err_pos[5]

        self.tau_prev = self.tau

        return self.tau

    def __str__(self):
        return """%s
                 pos_kp: %s
                 pos_kd: %s
                 pos_ki: %s
                 vel_kp: %s
                 vel_kd: %s
                 vel_ki: %s
               """ % (
            super(AutoTuneController, self).__str__(),
            self.pos_Kp, self.pos_Kd, self.pos_Ki,
            self.vel_Kp, self.vel_Kd, self.vel_Ki,
        )

class SlidingModeController(Controller):
    """SlidingModeController implements a sliding mode controller for .
           The output of this class is the force necessary for the 6 DOFs to move the vehicle so that the required
           characteristics are achieved.
    """
    def __init__(self, dt, ctrl_config, model_config, **kwargs):
        super(SlidingModeController, self).__init__(dt, ctrl_config, **kwargs)

        # init params
        # position gains
        self.epsilon = np.zeros(6)
        self.k = np.zeros(6)
        self.sigma = np.zeros(6)
        self.limit = np.zeros(6)

        # controller limits
        self.pos_lim = np.zeros(6)

        self.tau_ctrl = np.zeros(6)
        self.tau_prev = np.zeros(6)

        # errors
        self.err_pos = np.zeros(6)
        self.err_pos_prev = np.zeros(6)
        self.err_pos_der = np.zeros(6)
	self.err_pos_int = np.zeros(6)

        # init jacobians matrices
        self.J = np.zeros((6, 6))  # jacobian matrix (translate velocity from body referenced to Earth referenced)
        self.J_inv = np.zeros((6, 6))  # inverse jacobian matrix

    def update_config(self, ctrl_config, model_config):
        # sliding mode parameters
        self.epsilon = np.array([
            ctrl_config['pos_x']['epsilon'],
            ctrl_config['pos_y']['epsilon'],
            ctrl_config['pos_z']['epsilon'],
            ctrl_config['pos_k']['epsilon'],
            ctrl_config['pos_m']['epsilon'],
            ctrl_config['pos_n']['epsilon'],
        ])

        self.k = np.array([
            ctrl_config['pos_x']['k'],
            ctrl_config['pos_y']['k'],
            ctrl_config['pos_z']['k'],
            ctrl_config['pos_k']['k'],
            ctrl_config['pos_m']['k'],
            ctrl_config['pos_n']['k'],
        ])

        self.limit = np.array([
            ctrl_config['pos_x']['lim'],
            ctrl_config['pos_y']['lim'],
            ctrl_config['pos_z']['lim'],
            ctrl_config['pos_k']['lim'],
            ctrl_config['pos_m']['lim'],
            ctrl_config['pos_n']['lim'],
        ])

    def sigmoid(self, x):
        "Numerically-stable sigmoid function."
        if np.all(x) >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
        return z / (1 + z)

    def update(self, position, velocity):
        # store nav updates
        self.pos = position
        self.err_pos = self.des_pos - self.pos
        self.err_pos[3:6] = cnv.wrap_pi(self.err_pos[3:6])
        # update errors
        self.err_pos_der = (self.err_pos - self.err_pos_prev) / self.dt
	self.err_pos_int = np.clip(self.err_pos_int + self.err_pos, -self.limit, self.limit)
        self.sigma = self.err_pos_der + self.k*self.err_pos
        self.tau = - self.epsilon*self.sigmoid(self.sigma)
	self.tau[5] = -self.k[5]*self.err_pos[5] - self.epsilon[5]*self.err_pos_int[5]
	print "Force in yaw ", self.tau[5], " error in yaw ", self.err_pos[5]
        return self.tau

    def __str__(self):
        return """%s
                epsilon: %s
                k: %s
                limit: %s
                """ % (
            super(SlidingModeController, self).__str__(),
            self.epsilon, self.k, self.limit,
        )

############# Function for user defined terminiation############################
# Callback function

class MyCallback(ca.Callback):
    def __init__(self, name, nx, ng, np, patience, opts={}):
        ca.Callback.__init__(self)

        self.J_vals_past = 1000; self.signn = 0
        self.nx = nx; self.ng = ng; self.np = np; self.patience = patience
        # Initialize internal objects
        self.construct(name, opts)

    def get_n_in(self): return ca.nlpsol_n_out()
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        n = ca.nlpsol_out(i)
        if n=='f':
            return ca.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return ca.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return ca.Sparsity.dense(self.ng)
        else:
            return ca.Sparsity.dense(self.np)

    def eval(self, arg):	
        self.J_vals_curr=float(arg[1])
        if self.J_vals_past - self.J_vals_curr <= 0.1:
            self.signn = self.signn+1

        self.J_vals_past = self.J_vals_curr
        if self.signn == self.patience:
            self.J_vals_past = 1000; self.signn = 0
            print Js
        return [0]	


class MPCSurgeController(Controller):
    """MPCSurgeController implements a energy-optimal MPC controller for horizontal thruster control and SMC for depth control.
           The output of this class is the force necessary for the 6 DOFs to move the vehicle so that the required
           characteristics are achieved.
    """

    def __init__(self, dt, ctrl_config, model_config, **kwargs):
        super(MPCSurgeController, self).__init__(dt, ctrl_config, **kwargs)

        # init params for SMC #############################################
        # position gains
        self.epsilon = np.zeros(6)
        self.k = np.zeros(6)
        self.sigma = np.zeros(6)
        self.limit = np.zeros(6)

        # controller limits
        self.pos_lim = np.zeros(6)
        self.tau_ctrl = np.zeros(6)
        self.tau_prev = np.zeros(6)

        # errors
        self.err_pos = np.zeros(6)
        self.err_pos_prev = np.zeros(6)
        self.err_pos_der = np.zeros(6)
        self.err_pos_int = np.zeros(6)

        # init jacobians matrices
        self.J = np.zeros((6, 6))  # jacobian matrix (translate velocity from body referenced to Earth referenced)
        self.J_inv = np.zeros((6, 6))  # inverse jacobian matrix

        # init params for MPC #############################################
        # Some constants.
        self.Delta = .1
        self.Nx = 2
        self.Nu = 1

        #### MPC controller
        self.current_velocity = np.zeros(6)
        self.x0_pose = np.zeros(6)
        self.x0_vel = np.array([0.00001, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.save_flag = 0

        dt = 0.1   # sampling time
        N = 5   # prediction horizon 
        Tmax = 7.86 
        Tmin = -Tmax
        trmin = 0.1 #1e-4
        trmax = 30
        pi = np.pi
        u_dim = 2;
        x_dim = 6;

        # Model parameters
        rho_water = 1025; self.D_back = 0.1694;
        xg = 0; yg = 0; xb = 0; yb = 0; zb = -0.009; zg = 0.00178; 
        L = 1; rad = 0.1; m = 20.42; W = 200.116; B = 201.586; 
        Ixx = 0.12052; Iyy = 0.943099; Izz = 1.006087; 

        Xu_dot = -1*0.1*m; 
        Yv_dot = -1*math.pi * rho_water * rad**2 * L;
        Zw_dot = -1*math.pi * rho_water * rad**2 * L;
        Kp_dot = -1*math.pi * rho_water * rad**4 * 0.25;
        Mq_dot = -1*math.pi * rho_water * rad**2 * L**3 /12;
        Nr_dot = -1*math.pi * rho_water * rad**2 * L**3 /12;
        Xu = 48.17;Yv = 4.11;Zw = 4.11;Kp = 48.17;Mq = 4.11;Nr = 4.11;

        Mu_inv = 1/(m-Xu_dot);
        Mv_inv = 1/(m-Yv_dot);
        Mr_inv = 1/(Izz-Nr_dot);

        # Power consumption calculation
        def cal_p(T):
            T_abs = ca.fabs(T)
            return ca.sqrt(T_abs)*T_abs

        # Power for positive buoyancy
        ex_f = 2*cal_p((B-W)/2)

        # Model equations
        def ode(chi,T,dt):
            T_surge = T[0]+T[1]
            M_yaw = (T[0]-T[1])*self.D_back
            u_dot = - (-chi[1]*chi[2])*m - (Xu*ca.fabs(chi[0])*chi[0]) + T_surge
            v_dot = - (chi[0]*chi[2])*m - (Yv*ca.fabs(chi[1])*chi[1])
            r_dot = - (Nr*ca.fabs(chi[2])*chi[2]) + M_yaw
            x_dot = ca.cos(chi[5])*chi[0] - ca.sin(chi[5])*chi[1]
            y_dot = ca.sin(chi[5])*chi[0] + ca.cos(chi[5])*chi[1]
            psi_dot = chi[2]
            chi_dot = ca.vcat([u_dot*Mu_inv,v_dot*Mv_inv,r_dot*Mr_inv,x_dot,y_dot,psi_dot])
            chi_1 = chi + dt*chi_dot
            return chi_1

        # Stage cost
        def JL(T):
            J_L = (cal_p(T[0])+cal_p(T[1]))*dt
            return J_L

        # Normalize the angle to -pi to pi
        def wraptopi(x):
            angle_rad = x - 2*pi*ca.floor((x+pi)/(2*pi))
            return angle_rad

        # Calculate the yaw and surge energy
        def cal_yaw_moment(Dpsi,t_r,rN,T_s_est):
            r_max = 2*Dpsi/t_r - rN;
            a_psi = (r_max-rN)/t_r;
            M_yaw_max = (Nr*ca.fabs(r_max)*r_max) + (Izz-Nr_dot)*a_psi;
            M_yaw_min = (Nr*ca.fabs(rN)*rN) + (Izz-Nr_dot)*a_psi;
            M_yaw_mean = (Nr*ca.fabs(Dpsi/t_r)*Dpsi/t_r);
            T1_max = (T_s_est+M_yaw_max/self.D_back)/2;
            T1_min = (T_s_est+M_yaw_min/self.D_back)/2;
            T2_max = (T_s_est-M_yaw_max/self.D_back)/2;
            T2_min = (T_s_est-M_yaw_min/self.D_back)/2;
            T1_mean = (T_s_est+M_yaw_mean/self.D_back)/2;
            T2_mean = (T_s_est-M_yaw_mean/self.D_back)/2;
            J_horizontal = (cal_p(T1_max)+cal_p(T2_max)+cal_p(T1_min)+cal_p(T2_min))*t_r/2 + (cal_p(T1_mean)+cal_p(T2_mean))*t_r; 
            return J_horizontal

        # Terminal cost
        def JK(chi,chi_des,t_r,v_ini):
            T_s_est = Xu*chi[0]**2
            ud = ca.sqrt(chi[0]**2+v_ini**2)
            d = ca.sqrt((chi_des[0]-chi[3])**2 + (chi_des[1]-chi[4])**2)
            Dpsi = wraptopi(ca.atan2(chi_des[1]-chi[4],chi_des[0]-chi[3]) - ca.atan2(chi[1],chi[0]) -chi[5])
            t_rem = 2*t_r + ca.fabs(d/ud - 2*t_r*ca.sin(Dpsi)/(Dpsi+1e-16))
            J_horizontal = cal_yaw_moment(Dpsi,t_r,chi[2],T_s_est)
            J_K = ex_f*t_rem + 2*cal_p(T_s_est/2)*(t_rem-2*t_r) + J_horizontal;
            return J_K

        # Case scenario
        self.chi_f = [self.des_pos[0],self.des_pos[1]]    # terminal condition
        self.chi_0 = [self.x0_vel[0],self.x0_vel[1],self.x0_vel[5],self.x0_pose[0],self.x0_pose[1],self.x0_pose[5]]   # initial condition
        print ('Stop here =====================================================================')

        ############# Functions for MPC controller ########################################
        # Default setting in the NLP solver
        w0 = 1*np.ones(N*u_dim)
        self.w0 = np.append(w0,20)
        lbw = Tmin*np.ones(N*u_dim)
        ubw = Tmax*np.ones(N*u_dim)
        self.lbw = np.append(lbw,trmin)
        self.ubw = np.append(ubw,trmax)
        self.lbg = 1e-4*np.ones(N)

        mycallback = MyCallback('mycallback',N*u_dim+1,N,x_dim+2,3)
        #opts = {"ipopt.tol":1e-10, "expand":True, "iteration_callback":mycallback}
        opts = {"expand":True, "iteration_callback":mycallback, "ipopt.print_level":0}

        # A more stable way to define the problem
        wx = ca.MX.sym('wx',N*u_dim+1)	
        g = []; J = 0
        par = ca.MX.sym('param',x_dim+2)
        v_ini = par[1]
        chi_1 = par[0:x_dim]
        chi_des = par[x_dim:]
        for k in range(N):
            # New optimized varibales (control inputs)
            chi_1 = ode(chi_1,[wx[2*k],wx[2*k+1]],dt)
            g.append(chi_1[0])
            J = J + JL([wx[2*k],wx[2*k+1]])
        t_r = wx[-1]
        J = J + JK(chi_1,chi_des,t_r,v_ini)
        g = ca.vertcat(*g)

        # Create an NLP solver
        prob = {'f': J, 'x': wx, 'g': g, 'p':par}
        self.solver = ca.nlpsol('solver', 'ipopt', prob, opts);

        for ini_iter in range(2):
            sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, p=np.append(self.chi_0,self.chi_f))
            self.w0 = sol['x'].full()

        # Collect the simulation results
        self.u_plot = []; self.v_plot = []; self.T_lf_plot = []; self.T_rt_plot = [];
        self.cpu_plot = []; self.x_plot =[]; self.y_plot = []; self.psi_plot = [];


    def update_config(self, ctrl_config, model_config):
        # sliding mode parameters
        self.epsilon = np.array([
            ctrl_config['pos_x']['epsilon'],
            ctrl_config['pos_y']['epsilon'],
            ctrl_config['pos_z']['epsilon'],
            ctrl_config['pos_k']['epsilon'],
            ctrl_config['pos_m']['epsilon'],
            ctrl_config['pos_n']['epsilon'],
        ])

        self.k = np.array([
            ctrl_config['pos_x']['k'],
            ctrl_config['pos_y']['k'],
            ctrl_config['pos_z']['k'],
            ctrl_config['pos_k']['k'],
            ctrl_config['pos_m']['k'],
            ctrl_config['pos_n']['k'],
        ])

        self.limit = np.array([
            ctrl_config['pos_x']['lim'],
            ctrl_config['pos_y']['lim'],
            ctrl_config['pos_z']['lim'],
            ctrl_config['pos_k']['lim'],
            ctrl_config['pos_m']['lim'],
            ctrl_config['pos_n']['lim'],
         ])

    def sigmoid(self, x):
        "Numerically-stable sigmoid function."
        if np.all(x) >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
        return z / (1 + z)

    def update(self, position, velocity):
        # store nav updates
        self.pos = position
        self.err_pos = self.des_pos - self.pos
        self.err_pos[3:6] = cnv.wrap_pi(self.err_pos[3:6])
        # update errors
        self.err_pos_der = (self.err_pos - self.err_pos_prev) / self.dt
        self.err_pos_int = np.clip(self.err_pos_int + self.err_pos, -self.limit, self.limit)
        self.sigma = self.err_pos_der + self.k*self.err_pos
        self.tau = - self.epsilon*self.sigmoid(self.sigma)
	    #self.tau[5] = -self.k[5]*self.err_pos[5] - self.epsilon[5]*self.err_pos_int[5]
        self.tau[0], self.tau[5] = self.Simple_NMPC_Controller()
        #print "Force in yaw ", self.tau[5], " error in yaw ", self.err_pos[5]
        return self.tau

        # Save the simulation results
        # Plot the result
        #print self.save_flag, ca.sqrt((0.15-self.pos[0])**2 + (0.2-self.pos[1])**2), len(self.x_plot)
        #if self.save_flag == 0 and ca.sqrt((self.pos[0]+1.8)**2 + (self.pos[1]+2.4)**2) <= 3 and len(self.x_plot) > 1000: #triangular path
        if self.save_flag == 0 and ca.sqrt((self.pos[0]-3)**2 + (self.pos[1])**2) <= 3 and len(self.x_plot) > 1000:
             # Save the results (T_lf/T_rt/x/y/u/v)
             self.x_plot = [str(x) for x in self.x_plot];self.x_plot = ','.join(self.x_plot)
             self.y_plot = [str(x) for x in self.y_plot];self.y_plot = ','.join(self.y_plot)
             self.u_plot = [str(x) for x in self.u_plot];self.u_plot = ','.join(self.u_plot)
             self.v_plot = [str(x) for x in self.v_plot];self.v_plot = ','.join(self.v_plot)
             self.cpu_plot = [str(x) for x in self.cpu_plot];self.cpu_plot = ','.join(self.cpu_plot)
             self.T_lf_plot = [str(x[0]) for x in self.T_lf_plot];self.T_lf_plot = ','.join(self.T_lf_plot)
             self.T_rt_plot = [str(x[0]) for x in self.T_rt_plot];self.T_rt_plot = ','.join(self.T_rt_plot)
             self.psi_plot = [str(x) for x in self.psi_plot];self.psi_plot = ','.join(self.psi_plot)
             with open("/home/odroid/sphere_ws/src/controller/src/EO_EMPC.csv", "w") as f:
                 f.write(self.T_lf_plot+'\n')
                 f.write(self.T_rt_plot+'\n')
                 f.write(self.x_plot+'\n')
                 f.write(self.y_plot+'\n')
                 f.write(self.u_plot+'\n')
                 f.write(self.v_plot+'\n')
                 f.write(self.cpu_plot+'\n')
                 f.write(self.psi_plot+'\n')
             print('Results Saved!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
             self.save_flag = 1;

        print "MPC controller: ", self.tau[0]
        print "Velocity: ", self.vel
        print "Input: ", self.tau

        self.tau_prev = self.tau

        return self.tau


    def Simple_NMPC_Controller(self):
        self.chi_f = [self.des_pos[0],self.des_pos[1]]    # terminal condition
        self.chi_0 = [self.vel[0],self.vel[1],self.vel[5],self.pos[0],self.pos[1],self.pos[5]]   # initial condition

        if self.des_pos[0] == 0 and self.des_vel[0] == 0:
            T_forward = 0
            M_steer = 0
        else:
            t_sta = time.clock()
            sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, p=np.append(self.chi_0,self.chi_f))
            self.w0 = sol['x'].full()
            T_forward = self.w0[0]+self.w0[1]
            M_steer = (self.w0[0]-self.w0[1])*self.D_back
            t_end = (time.clock() - t_sta)
            if self.vel[0] != 0:
                # Collect the simulation results
                self.u_plot.append(self.vel[0]); self.v_plot.append(self.vel[1]); 
                self.T_lf_plot.append(list(self.w0[0])); self.T_rt_plot.append(list(self.w0[1])); 
                self.x_plot.append(self.pos[0]); self.y_plot.append(self.pos[1]); 
                self.psi_plot.append(self.pos[5]); self.cpu_plot.append(t_end); 
                #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                #print self.u_plot, self.T_lf_plot, self.cpu_plot
                #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return T_forward, M_steer


    def __str__(self):

        return """%s
                epsilon: %s
                k: %s
                limit: %s
                tau_c: %s
                """ % (
            super(SMPCSurgeController, self).__str__(),
            self.epsilon, self.k, self.limit, self.tau_ctrl
        )













