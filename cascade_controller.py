#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import casadi as ca
import math
import time
import rospy
np.set_printoptions(precision=3, suppress=True)

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

############# MPC controller for energy-optimal control############################
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

# MPC controller
class MPCSurgeController(Controller):
    """MPCSurgeController implements a energy-optimal MPC controller for horizontal thruster control and SMC for depth control.
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
        # varioables defined for the default PID control 
        self.vel_def = 0.15
        self.err_vel_def = np.zeros(6)
        self.err_vel_int_def = np.zeros(6)
        self.err_vel_prev_def = np.zeros(6)
        self.err_vel_der_def = np.zeros(6)


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
        self.los_angle = 0.0
        self.d_los = 0.5

        dt = 0.1   # sampling time
        N = 10   # prediction horizon 
        Tmax = 7.86 
        Tmin = -Tmax
        trmin = 0.1 #1e-4
        trmax = 30
        pi = np.pi
        u_dim = 2;
        x_dim = 6;

        # Model parameters
        self.D_back = 0.1694;
        xg = 0; yg = 0; xb = 0; yb = 0; zb = -0.009; zg = 0.00178; 
        m = 20.42; W = 200.116; B = 201.586; 
        Ixx = 0.12052; Iyy = 0.943099; Izz = 1.006087; 

        Xu_dot = -10.0 
        Yv_dot = -20.0
        Nr_dot = -2.0
        Xu = 15.0
        Yv = 10.0
        Nr = 10.0
        self.MPC_flag = 0

        Mu_inv = 1/(m-Xu_dot);
        Mv_inv = 1/(m-Yv_dot);
        Mr_inv = 1/(Izz-Nr_dot);

        # Power consumption calculation
        def cal_p(T):
            return 0.4052*T*T

        # Power for positive buoyancy
        ex_f = 2*cal_p((B-W)/2)

        # Model equations
        def ode(chi,T,dt):
            T_surge = T[0]+T[1]
            M_yaw = (T[0]-T[1])*self.D_back
            u_dot = - (-chi[1]*chi[2])*m - (Xu*chi[0]) + T_surge
            v_dot = - (chi[0]*chi[2])*m - (Yv*chi[1])
            r_dot = - (Nr*chi[2]) + M_yaw
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
            M_yaw_max = (Nr*r_max) + (Izz-Nr_dot)*a_psi;
            M_yaw_min = (Nr*rN) + (Izz-Nr_dot)*a_psi;
            M_yaw_mean = (Nr*Dpsi/t_r);
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
            T_s_est = Xu*chi[0]
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
        
        if self.MPC_flag == 1:     
        ######################### EO-MPC ########################################
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
        ##############################################################################
        else:
        ################################# T-MPC #####################################
            # Default setting in the NLP solver
            self.w0 = 1*np.ones(N*u_dim)
            self.lbw = Tmin*np.ones(N*u_dim)
            self.ubw = Tmax*np.ones(N*u_dim)
            self.lbg = 1e-4*np.ones(N)
            #self.psi_set = 0
            self.u_set = 0.15

            mycallback = MyCallback('mycallback',N*u_dim,0,x_dim+1,3)
            opts = {"expand":True, "iteration_callback":mycallback, "ipopt.print_level":0}
            #opts = {"expand":True, "ipopt.print_level":0}

            # A more stable way to define the problem
            wx = ca.MX.sym('wx',N*u_dim)	
            g = []; J = 0
            par = ca.MX.sym('param',x_dim+1)
            chi_1 = par[0:x_dim]
            psi_set = par[-1]
            # Reference smoothing
            u_diff = (self.u_set-chi_1[0])/N
            psi_diff = (psi_set-chi_1[5])/N
            st_ini = chi_1
            for k in range(N):
                # New optimized varibales (control inputs)
                chi_1 = ode(chi_1,[wx[2*k],wx[2*k+1]],dt)
                # Standard surge & yaw setpoint tracking
                #J = J + 50*(chi_1[0]-self.u_set)**2 # + (wraptopi(chi_1[5]-psi_set))**2
                # Setpoint tracking with input penalty
                J = J + 30*(chi_1[0]-0.15)**2 + 5*(wraptopi(chi_1[5]-psi_set))**2+0.03*(wx[2*k]**2+wx[2*k+1]**2)  
                # Setpoint tracking with lowest oscillations
                #if k < N-1: 
                #    J = J + 30*(chi_1[0]-0.15)**2 + (wraptopi(chi_1[5]-psi_set))**2+5.0*((wx[2*(k+1)]-wx[2*k])**2+(wx[2*(k+1)+1]-wx[2*k+1])**2)
                #else: 
                #    J = J + 30*(chi_1[0]-0.15)**2 + (wraptopi(chi_1[5]-psi_set))**2
                # Setpoint tracking with reference smoothing
                #J = J + 100*(chi_1[0]-(st_ini[0] + (k+1)*u_diff))**2 + (wraptopi(chi_1[5]-(st_ini[5] + (k+1)*psi_diff)))**2 
            g = ca.vertcat(*g)
            #J = J + 50*(chi_1[0]-self.u_set)**2 + (wraptopi(chi_1[5]-psi_set))**2

            # Create an NLP solver
            prob = {'f': J, 'x': wx, 'g': g, 'p':par}
            self.solver = ca.nlpsol('solver', 'ipopt', prob, opts);

            for ini_iter in range(2):
                sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, p=np.append(self.chi_0,self.los_angle))
                self.w0 = sol['x'].full()
       ##############################################################################

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
        return np.where(x>=0,1/(1+np.exp(-x))-0.5, np.exp(x)/(1+np.exp(x))-0.5)

    def mpcupdate(self, position, velocity):
        # SMC Controller
        self.pos = position
        self.err_pos = self.des_pos - self.pos
        self.err_pos[3:6] = cnv.wrap_pi(self.err_pos[3:6])
        self.err_pos_der = (self.err_pos - self.err_pos_prev) / self.dt
        self.err_pos_int = np.clip(self.err_pos_int + self.err_pos, -self.limit, self.limit)
        self.sigma = self.err_pos_der + self.k*self.err_pos
        self.tau = - self.epsilon*self.sigmoid(self.sigma)
	# MPC Controller
        self.tau[0], self.tau[5], opt_sol = self.Simple_NMPC_Controller()
        #print "MPC controller: ", self.tau
        #print "Velocity: ", self.vel
        #print "Input: ", self.tau
        self.tau_prev = self.tau
        return self.tau, opt_sol

    def update(self, position, velocity):
        self.pos = position
        # PID for depth control
        self.tau = np.zeros(6)
        self.err_pos = self.des_pos - self.pos
        self.err_pos_der = (self.err_pos - self.err_pos_prev) / self.dt
        self.err_pos_int = self.err_pos_int + self.err_pos*self.dt
        self.tau[2] = - (self.k[2]*self.err_pos[2] + self.epsilon[2]*self.err_pos_int[2]) #+0.5*self.err_pos_der[5]
        """
        # SMC Controller (for depth control use)
        self.err_pos = self.des_pos - self.pos
        self.err_pos[3:6] = cnv.wrap_pi(self.err_pos[3:6])
        self.err_pos_der = (self.err_pos - self.err_pos_prev) / self.dt
        self.err_pos_int = np.clip(self.err_pos_int + self.err_pos, -self.limit, self.limit)
        self.sigma = self.err_pos_der + self.k*self.err_pos
        self.tau = - self.epsilon*self.sigmoid(self.sigma) 
        self.tau[0] = 0.0

        # PID control for surge speed and yaw angle
        self.tau = np.zeros(6)
        self.des_pos[5] = math.atan2(self.des_pos[1]-self.pos[1],self.des_pos[0]-self.pos[0]) - math.atan2(self.vel[1],self.vel[0])
        self.err_pos = self.des_pos - self.pos
        self.err_pos[3:6] = cnv.wrap_pi(self.err_pos[3:6])
        self.err_pos_der = (self.err_pos - self.err_pos_prev) / self.dt
        #self.err_pos_int = np.clip(self.err_pos_int + self.err_pos*self.dt, -self.limit, self.limit)
        self.err_pos_int = self.err_pos_int + self.err_pos*self.dt
        self.tau[5] = - (self.k[5]*self.err_pos[5] + self.epsilon[5]*self.err_pos_int[5]) #+0.5*self.err_pos_der[5]
        self.err_vel_def[0] = self.vel_def-self.vel[0];
        self.err_vel_der_def[0] = (self.err_vel_def[0] - self.err_vel_prev_def[0]) / self.dt
        #self.err_vel_int_def[0] = np.clip(self.err_vel_int_def[0] + self.err_vel_def[0]*self.dt, -self.limit[0], self.limit[0])
        self.err_vel_int_def[0] = self.err_vel_int_def[0] + self.err_vel_def[0]*self.dt
        self.tau[0] =  self.k[0]*self.err_vel_def[0] + self.epsilon[0]*self.err_vel_int_def[0] #+0.5*self.err_vel_der_def[0]
        #self.tau[0] = np.clip(self.tau[0], -self.limit[1], self.limit[1])
        #self.tau[5] = np.clip(self.tau[5], -self.limit[2], self.limit[2])
        """
        #print "SMC controller: ", self.tau
        #print "Velocity: ", self.vel
        #print "Input: ", self.tau
        self.tau_prev = self.tau

        return self.tau


    def Simple_NMPC_Controller(self):
        self.chi_f = [self.des_pos[0],self.des_pos[1]]    # terminal condition
        if self.vel[0] == 0.0:
            self.chi_0 = [self.vel[0]+0.00001,self.vel[1],self.vel[5],self.pos[0],self.pos[1],self.pos[5]]
        else:
            self.chi_0 = [self.vel[0],self.vel[1],self.vel[5],self.pos[0],self.pos[1],self.pos[5]]   # initial condition
        self.psi_set = 0

        if self.des_pos[0] == 0 and self.des_vel[0] == 0:
            T_forward = 0
            M_steer = 0
        elif np.sqrt((self.des_pos[0]-self.pos[0])**2+(self.des_pos[1]-self.pos[1])**2) < self.r_coa and self.wp_ind == self.wp_num-1:
            T_forward = 0
            M_steer = 0
        else:
            if self.MPC_flag == 1:
                sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, p=np.append(self.chi_0,self.chi_f))
            else:
                # Pure pursuit guidance law
                self.los_angle = math.atan2(self.des_pos[1]-self.pos[1],self.des_pos[0]-self.pos[0]) - math.atan2(self.vel[1],self.vel[0]) 
                # LOS guidance law
                #self.track_error = np.dot(self.path_rot,np.array([[self.pos[0]-self.des_pos_pre[0]],[self.pos[1]-self.des_pos_pre[1]]]))
                #print(self.track_error)
                #self.los_angle = cnv.wrap_pi(self.path_slope-math.atan2(self.track_error[1],self.d_los)-math.atan2(self.vel[1],self.vel[0]))               
                sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, p=np.append(self.chi_0,self.los_angle))
            self.w0 = sol['x'].full()
            T_forward = self.w0[0]+self.w0[1]
            M_steer = (self.w0[0]-self.w0[1])*self.D_back
        #return T_forward, M_steer, np.append(np.append(np.append(self.w0,self.des_pos[0]),self.des_pos[1]),self.los_angle)
        #print self.w0.size
        return T_forward, M_steer, np.concatenate((self.w0.reshape(-1),np.array([self.des_pos[0],self.des_pos[1],self.los_angle])),axis=1)
        #return T_forward, M_steer, np.append(self.w0,self.los_angle)


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








