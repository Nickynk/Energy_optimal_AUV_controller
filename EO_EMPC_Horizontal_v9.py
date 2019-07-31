# This file develpes the MPC using Casadi (optimizer based on ipopt)
# The objective is to implement the EO-EMPC algorithm on the DROP-Sphere
# Several Implementation simplifications are given as follows:
# 

# Import the packages
import casadi as ca
import math
import numpy as np
import time
## SET BACKEND
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Controller Parameter Setting
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
rho_water = 1025; D_back = 0.1694;
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
	M_yaw = (T[0]-T[1])*D_back
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
    T1_max = (T_s_est+M_yaw_max/D_back)/2;
    T1_min = (T_s_est+M_yaw_min/D_back)/2;
    T2_max = (T_s_est-M_yaw_max/D_back)/2;
    T2_min = (T_s_est-M_yaw_min/D_back)/2;
    T1_mean = (T_s_est+M_yaw_mean/D_back)/2;
    T2_mean = (T_s_est-M_yaw_mean/D_back)/2;
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

############# Functions for planning ########################################
# Calculate the path deviation
def ct_error(xt,yt,x_ini,y_ini,xf,yf):
	pt = np.matrix([xt,yt,0]); v_ini = np.matrix([x_ini,y_ini,0]); v_end = np.matrix([xf,yf,0]);
	a = v_ini - v_end;
	b = pt - v_end;
	d_ct = ca.sqrt(np.sum(np.power(np.cross(a,b),2))) / ca.sqrt(np.sum(np.power(a,2)))
	return d_ct

# Mission Planner
def wp_planner(chi,chi_ini,chi_f,r_a,ini_sign,step,path_x,path_y):
	flag = 0
	if ca.mod(step,2) == ini_sign:
		if ca.sqrt((chi_f[0]-chi[3])**2 + (chi_f[1]-chi[4])**2) <= r_a:
			step = step + 1; flag = 1
	else:
		d_ct = ct_error(chi[3],chi[4],chi_ini[0],chi_ini[1],chi_f[0],chi_f[1])
		if d_ct <= 0.2:
			step = step + 1; flag = 1

	if flag == 1:
		chi_f[0] = path_x[step]
		chi_f[1] = path_y[step]
		chi_ini[0] = path_x[step-1]
		chi_ini[1] = path_y[step-1]

	return chi_ini,chi_f,step

# Plane the waypoints
def auto_wp(wp_x,wp_y,r_a):
	path_x = []; path_y = [];
	for i in range(len(wp_x)-1):
		wp_r = r_a/ca.sqrt((wp_x[i+1]-wp_x[i])**2+(wp_y[i+1]-wp_y[i])**2)
		wp_x_next = wp_x[i]+(wp_x[i+1]-wp_x[i])*wp_r;
		wp_y_next = wp_y[i]+(wp_y[i+1]-wp_y[i])*wp_r;
		path_x.append(wp_x[i]) 
		path_x.append(wp_x_next)
		path_y.append(wp_y[i]) 
		path_y.append(wp_y_next)

	path_x.append(wp_x[-1])
	path_y.append(wp_y[-1])
	path_x = path_x[1:];
	path_y = path_y[1:];
	return path_x,path_y

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

# Controller Evaluation
############################################################################################
############################################################################################
# Triangular Path
wp_x = [0,25,15,0-3*3/ca.sqrt(3**2+4**2)];
wp_y = [0,5,20,0-4*3/ca.sqrt(3**2+4**2)];

# Circular Path
#wp_x = [0];
#wp_y = [0];
#r_c = 15;seg = 20
#for angle in range(seg):
#	wp_x.append(r_c*ca.cos((angle+1)*2*pi/seg+6*pi/4))
#	wp_y.append(r_c+r_c*ca.sin((angle+1)*2*pi/seg+6*pi/4))
#wp_x.extend((0,3));
#wp_y.extend((0,0));

# Initialize the mission planner
r_a = 2; step = 0
path_x,path_y = auto_wp(wp_x,wp_y,r_a)
ini_sign = 1

#print path_x, path_y
#raw_input()

# Case scenario
chi_f = [path_x[0],path_y[0]]    # terminal condition
chi_ini = [0,0]
chi_0 = [0.0001,0,0,0,0,ca.atan2(chi_f[1]-chi_ini[1],chi_f[0]-chi_ini[0])]   # initial condition

t_ini_sta = time.clock()
############# Functions for MPC controller ########################################
# Default setting in the NLP solver
w0 = 1*np.ones(N*u_dim)
w0 = np.append(w0,20)
lbw = Tmin*np.ones(N*u_dim)
ubw = Tmax*np.ones(N*u_dim)
lbw = np.append(lbw,trmin)
ubw = np.append(ubw,trmax)
lbg = np.append(1e-4*np.ones(N),-np.inf*np.ones(N))
ubg = np.append(np.inf*np.ones(N),0.2*np.ones(N))
lbg1 = np.append(1e-4*np.ones(N),-np.inf*np.ones(N))
ubg1 = np.append(np.inf*np.ones(N),np.inf*np.ones(N))

mycallback = MyCallback('mycallback',N*u_dim+1,N*2,x_dim+4,3)
#opts = {"ipopt.tol":1e-10, "expand":True, "iteration_callback":mycallback}
opts = {"expand":True, "iteration_callback":mycallback, "ipopt.print_level":0}

# A more stable way to define the problem
wx = ca.MX.sym('wx',N*u_dim+1)	
g = []; g1 = []; J = 0
par = ca.MX.sym('param',x_dim+4)
v_ini = par[1]
chi_1 = par[0:x_dim]
chi_des = par[x_dim:x_dim+2]
chi_ini = par[x_dim+2:]
for k in range(N):
	# New optimized varibales (control inputs)
	chi_1 = ode(chi_1,[wx[2*k],wx[2*k+1]],dt)
	g.append(chi_1[0])
	g1.append(ct_error(chi_1[3],chi_1[4],chi_ini[0],chi_ini[1],chi_des[0],chi_des[1]))
	J = J + JL([wx[2*k],wx[2*k+1]])
t_r = wx[-1]
J = J + JK(chi_1,chi_des,t_r,v_ini)
g = np.append(g,g1)
g = ca.vertcat(*g)

# Create an NLP solver
prob = {'f': J, 'x': wx, 'g': g, 'p':par}
# solver = nlpsol('solver', 'worhp', prob);
solver = ca.nlpsol('solver', 'ipopt', prob, opts);

t_ini_end = (time.clock() - t_ini_sta)
cpu_ini = [t_ini_end]

chi_ini = [0,0]

for ini_iter in range(2):
	t_ini_sta = time.clock()
	sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=np.concatenate([chi_0,chi_f,chi_ini]))
	w0 = sol['x'].full()
	t_ini_end = (time.clock() - t_ini_sta)
	cpu_ini.append(t_ini_end)

############# Simulation ########################################
# For plotting x and u given w
x_plot = []; y_plot = []; u_plot = []; v_plot = []; T_lf_plot = []; T_rt_plot = []; cpu_plot = []
sim_t = 10000

# Run the simulation as well as the controller
for sim in range(sim_t):
	x_plot.append(chi_0[3]); y_plot.append(chi_0[4]); u_plot.append(chi_0[0]); v_plot.append(chi_0[1])

	# Calculate the MPC
	t_sta = time.clock()
	if ca.mod(step,2) == ini_sign:
		sol = solver(x0=w0, lbx=lbw, lbg=lbg, ubx=ubw, ubg=ubg, p=np.concatenate([np.reshape(chi_0,(-1,)),chi_f,chi_ini]))
	else:
		sol = solver(x0=w0, lbx=lbw, lbg=lbg1, ubx=ubw, ubg=ubg1, p=np.concatenate([np.reshape(chi_0,(-1,)),chi_f,chi_ini]))
	w0 = sol['x'].full()
	T_mpc = [w0[0],w0[1]] 
	t_end = (time.clock() - t_sta)
	cpu_plot.append(t_end)

	# Simulate the system
	chi_0 = ode(chi_0,T_mpc,dt)

	# Waypoint planner
	if step < len(path_x): 
		chi_ini,chi_f,step = wp_planner(chi_0,chi_ini,chi_f,r_a,ini_sign,step,path_x,path_y)

	if step >= 2:
		raw_input()

	T_lf_plot.append(T_mpc[0]); T_rt_plot.append(T_mpc[1])

	print('Step:',step, 'u:',chi_0[0],'x:',chi_0[3],'y:',chi_0[4]); 
	print('x_ini:',chi_ini[0],'y_ini:',chi_ini[1],'x_f:',chi_f[0],'y_f:',chi_f[1]); 
	print('Left:',T_mpc[0],'Right',T_mpc[1],'CPU Time',t_end)
	#raw_input()

	# Stopper
	if ca.sqrt((chi_0[3]-path_x[-1])**2+(chi_0[4]-path_y[-1])**2) <= 3 and step >= 3:
		break



# Plot the result
tgrid = np.linspace(0,sim+1,sim+1)*dt
plt.figure()
plt.clf()
plt.subplot(221)
plt.plot(tgrid, cpu_plot, '-')
plt.xlabel('t')
plt.ylabel('cpu time')
plt.grid()
plt.subplot(222)
plt.plot(tgrid, T_lf_plot, '-')
plt.plot(tgrid, T_rt_plot, '-')
plt.xlabel('t')
plt.ylabel('T')
plt.legend(['left','right'])
plt.grid()
plt.subplot(223)
plt.plot(tgrid, u_plot, '-')
plt.xlabel('t')
plt.ylabel('u')
plt.grid()
plt.subplot(224)
plt.plot(x_plot, y_plot, '-')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
plt.savefig('EO_EMPC_v9.png')

# Save the results (T_lf/T_rt/x/y/u/v)
x_plot = [str(x) for x in x_plot]
y_plot = [str(x) for x in y_plot]
u_plot = [str(x) for x in u_plot]
v_plot = [str(x) for x in v_plot]
cpu_plot = [str(x) for x in cpu_plot]
T_lf_plot = [str(x[0]) for x in T_lf_plot]
T_rt_plot = [str(x[0]) for x in T_rt_plot]
cpu_ini = [str(x) for x in cpu_ini]
T_lf_plot = ','.join(T_lf_plot)
T_rt_plot = ','.join(T_rt_plot)
x_plot = ','.join(x_plot)
y_plot = ','.join(y_plot)
u_plot = ','.join(u_plot)
v_plot = ','.join(v_plot)
cpu_plot = ','.join(cpu_plot)
cpu_ini = ','.join(cpu_ini)
with open("EO_EMPC_v9.csv", "w") as f:
	f.write(T_lf_plot+'\n')
	f.write(T_rt_plot+'\n')
	f.write(x_plot+'\n')
	f.write(y_plot+'\n')
	f.write(u_plot+'\n')
	f.write(v_plot+'\n')
	f.write(cpu_plot+'\n')
	f.write(cpu_ini+'\n')

