import os
import casadi as ca
import numpy as np
import rospy
import time
import matplotlib.pyplot as plt
import utilities_TO as utils_TO
import pickle
import pygame

# define the required paths
code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
path_to_repo = os.path.join(code_path, '..', '..')          # getting path to the repository

## PARAMETERS -----------------------------------------------------------------------------------------------
# are we debugging or not?
debug = True

# set the cost weights
gamma_goal = 1                  # weight for distance to goal
gamma_throttle = 1e-1           # weight for use of throttle input
gamma_steering = 1e-2           # weight for use of steering input
gamma_acceleration = 0          # weight on the coordinates' acceleration

# goal state for the jetracer
x_goal = 5                      # x desired position
y_goal = 4                      # y desired position
eta_goal = np.deg2rad(30)       # eta (orientation) desired

# determine the time horizon and control intervals for the NLP problem
N = 10  # control intervals used (control will be constant during each interval)
T = 1.  # time horizon for the optimization

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

# initialize pygame to set up a state machine in the code execution logic
pygame.init()

# create the window that the user will have to keep selected to give their inputs
window = pygame.display.set_mode((400, 300))

print("Use the following keys to control the robot:") 
print("- 's' to start the MPC")
print("- 'q' to stop the execution")
print("- 't' to trick the MPC, and pretend that we have a reference (debug)")

ros_rate = rospy.Rate(rate_MPC)
run_MPC = False

if not debug:
        while not rospy.is_shutdown():

                for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_s:     # user wants to start
                                        
                                        if to_module.current_state_values is None:
                                                print("Not receiving current state estimates \nTry again later...")
                                        else:   
                                                print("MPC is starting")
                                                run_MPC = True
                                                to_module.togglePublishing(run_MPC)        # start the MPC

                                if event.key == pygame.K_q:     # user wants to quit
                                        print("Quitting and freezing to current position")
                                        run_MPC = False
                                        to_module.togglePublishing(run_MPC)               # stop the MPC

                                if event.key == pygame.K_t:     # user is naughty
                                        print("User is trying a trick")
                                        run_MPC = True
                                        to_module.current_state_values = x_0            # set fictitious starting point
                                        to_module.togglePublishing(run_MPC)                # start the MPC

                if run_MPC:     # if we need to, run the optimization
                        to_module.optimize_trajectory()

                ros_rate.sleep()
else:   
        to_module.current_state_values = x_0                    # set fictitious starting point for testing
        x_opt, u_opt, j_opt = to_module.optimize_trajectory()   # optimize the trajectory from there


        # utilities for plotting while debugging
        # plot the resulting optimal trajectory
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(x_opt[0,:], x_opt[1,:])
        ax.scatter(x_goal[0], x_goal[1], label='goal position')
        ax.scatter(x_0[0], x_0[1], label='initial position')
        ax.legend()

        # plot the behavior of states and controls
        time_stamps = np.arange(0, N+1, 1)*T/N
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(time_stamps, x_opt[0], label='x position')
        ax.plot(time_stamps, x_opt[1], label='y position')
        ax.plot(time_stamps, x_opt[2], label='eta orientation')
        ax.plot(time_stamps, x_opt[3], label='forward velocity')
        ax.set_title('States')
        ax.set_xlabel('time [s]')
        ax.legend()

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(time_stamps[0:-1], u_opt[0], label='throttle')
        ax.plot(time_stamps[0:-1], u_opt[1], label='steering input')
        ax.set_title('Controls')
        ax.set_xlabel('time [s]')
        ax.legend()

        plt.show()

        aux = 0