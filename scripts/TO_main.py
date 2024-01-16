import os
import casadi as ca
import numpy as np
import rospy
import time
import matplotlib.pyplot as plt
import utilities_TO as utils_TO
import pickle

# define the required paths
code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
path_to_repo = os.path.join(code_path, '..', '..')          # getting path to the repository

## PARAMETERS -----------------------------------------------------------------------------------------------
# are we debugging or not?
debug = True

# set the cost weights
gamma_goal = 10         # weight for distance to goal
gamma_throttle = 1      # weight for use of throttle input
gamma_steering = 1      # weight for use of steering input
gamma_acceleration = 1  # weight on the coordinates' acceleration

# goal state for the jetracer
x_goal = 5                      # x desired position
y_goal = 4                      # y desired position
eta_goal = np.deg2rad(30)       # eta (orientation) desired

# determine the time horizon and control intervals for the NLP problem
N = 30  # control intervals used (control will be constant during each interval)
T = 3.  # time horizon for the optimization

# choose collocation scheme for approximating the system dynamics
collocation_scheme = 'legendre'         # collocation points
polynomial_order = 3                    # order of polynomial basis

# constraints on throttle and steering input
tau_max = 1
tau_min = -1
sigma_max = 1
sigma_min = -1

# choose solver and set its options
solver = 'ipopt'        # available solvers depend on CasADi interfaces

opts = {'ipopt.print_level': 5, # options for the solver (check CasADi/solver docs for changing these)
        'print_time': 3,
        'expand': 1,
        'ipopt.tol': 1e-3,
        'error_on_fail':1,                                      # to guarantee transparency if solver fails
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.warm_start_bound_push': 1e-9,
        'ipopt.warm_start_bound_frac': 1e-9,
        'ipopt.warm_start_slack_bound_frac': 1e-9,
        'ipopt.warm_start_slack_bound_push': 1e-9,
        'ipopt.warm_start_mult_bound_push': 1e-9,
        'ipopt.mu_strategy': 'monotone',
        'ipopt.mu_init': 1e-4,
        'ipopt.nlp_scaling_method': 'none',
        'ipopt.bound_relax_factor': 1e-9}


# define the topics over which communication happens
shared_ros_topics = {}

# topic to receive the current state from
shared_ros_topics['state_estimation'] = 'state_estimation'

# topic to publish the control values
shared_ros_topics['control_values'] = 'control_values'

# rate at which we want to run the MPC 
rate_MPC = 20

# choose which car we are working with (can be 1, 2, or 3)
car_number = 1

# set the car length
car_length = 0.2

# -----------------------------------------------------------------------------
# Declare model variables
# The model represents a jetracer mobile robot. More specifications about the robot can be obtained
# at https://github.com/Lorenzo-Lyons/Jetracer_WS_github/tree/hackathon_18_Jan_2024

x = ca.MX.sym('x', 4)   # state vector: [x, y, eta, v], respectively in m, rad and rad/s

u = ca.MX.sym('u', 2)   # control vector: [tau, sigma] (throttle and sterring input)    

# define the goal position for the robot (zero final velocity)
x_goal = np.array([x_goal,
                   y_goal,
                   eta_goal,
                   0])      

# initial state - used to build the NLP structure, can be changed at run-time later
x_0 = 0     # x initial position
y_0 = 0     # y initial position
eta_0 = 0   # initial orientation
v_0 = 0     # initial velocity

x_0 = np.array([x_0,
                y_0,
                eta_0,
                v_0]) 

# instantiate NLP problem
nlp_jetracer = utils_TO.nlp_jetracer(car_number=car_number, car_length = car_length)

# instantiate trajectory optimization module, given the NLP problem, 
# the shared ros topics (defined in experiment_parameters.py), and setting debugging options
to_module = utils_TO.TO_module(nlp = nlp_jetracer, 
                               shared_ros_topics=shared_ros_topics, 
                               rate = rate_MPC,
                               debug_mode = debug)

# initialize the NLP problem with its parameters
to_module.nlp_module.setTimeHorizonAndDiscretization(N=N, T=T)
to_module.nlp_module.populateCollocationMatrices(order_polynomials= polynomial_order, collocation_type= collocation_scheme)

# provide states and controls to the NLP instance
to_module.nlp_module.initializeStateVariables(x = x, names = ['x', 'y', 'eta', 'v'])
to_module.nlp_module.initializeControlVariables(u = u, names= ['tau', 'sigma'])

# set the goal to be reached, and initial condition we are starting from
to_module.nlp_module.setGoal(goal = x_goal)
to_module.nlp_module.setInitialState(x_0 = x_0)

# other terms of the cost function are easier to treat in the NLP structure directly
to_module.nlp_module.setCostWeights(goal = gamma_goal,
                                     throttle = gamma_throttle,
                                     steering = gamma_steering,
                                     acceleration = gamma_acceleration)

# set up the NLP
to_module.nlp_module.formulateNLP(constraint_list = {'tau_max' : tau_max,
                                                      'tau_min' : tau_min,
                                                      'sigma_max' : sigma_max,
                                                      'sigma_min' : sigma_min})

to_module.nlp_module.setSolverOptions(solver = solver, opts = opts)

# create a function that solves the NLP given numerical values for its parameters
to_module.createMPCfunctionInitialGuesses()
to_module.createMPCfunctionWithoutInitialGuesses()

x_opt, u_opt, solution = to_module.nlp_module.solveNLPOnce()
