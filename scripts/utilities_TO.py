import casadi as ca
import numpy as np
import rospy
import matplotlib.pyplot as plt
import os
import time
import threading

# Messages
from std_msgs.msg import Float64MultiArray, Bool

class TO_module:
    def __init__(self, nlp, shared_ros_topics, rate=200, debug_mode:Bool=False):
        """"
        The TO_module requires a NLP object when instantiated
        """
        # set debug level
        self.debugging = debug_mode

        # Parameters
        self.state_space_names = None               # The names of the coordinates in the model that describe the state-space in which we move
                                                    # For the case of our shoulder rehabilitation, it will be ["plane_elev", "shoulder_elev", "axial_rot"]

        self.current_state_values = None            # The current value of the variables defining the state. The dimension of this and of self.state_space_names
                                                    # must be consistent (TODO: implement a check for this)

        self.x_opt = None                           # Optimal trajectory

        self.nlp_module = nlp

        # MPC optimal map generated with CasADi opti.to_function() for iteratively solving the same
        # NLP with different parameter values
        self.MPC_iter = None                        # CasADi function that maps initial state to the optimal control


            # # initialize ROS node and set required frequency
            # rospy.init_node('TO_module')
            # self.ros_rate = rospy.Rate(rate)

            # # Create publisher for the optimal trajectory for the robot
            # self.topic_opt_traj = shared_ros_topics['optimal_cartesian_ref_ee']
            # self.pub_trajectory = rospy.Publisher(self.topic_opt_traj, Float64MultiArray, queue_size=1)

            # # Create publisher for the initial cartesian pose of the KUKA end-effector, and subscriber to the topic which stops the
            # # publishing stream.
            # self.topic_init_pose = shared_ros_topics['cartesian_init_pose']
            # self.pub_init_pose = rospy.Publisher(self.topic_init_pose, Float64MultiArray, queue_size=1)
            # self.keep_publishing_init_pose = None       # flag indicating if pub_init_pose should keep publishing or not

            # self.sub_stop_pub_init_pose = rospy.Subscriber(self.topic_init_pose+'/stop', Bool, self._stop_pub_init_pose_cb, queue_size=10)

            # # Create a subscriber to listen to the current value of the shoulder pose
            # self.topic_shoulder_pose = shared_ros_topics['estimated_shoulder_pose']
            # self.sub_crr_shoulder_pose = rospy.Subscriber(self.topic_shoulder_pose, Float64MultiArray, self._shoulder_pose_cb, queue_size=1)
            # self.flag_receiving_shoulder_pose = False       # flag indicating whether the shoulder pose is being received

            # # Create the publishers dedicated to stream the optimal trajectories and controls (they will be executed in another thread)
            # self.flag_pub_trajectory = False    # flag to check if trajectory is being published (default: False = no publishing)
            # self.topic_optimal_trajectory_shoulder = 'optimal_trajectory_shoulder'
            # self.topic_optimal_controls_shoulder_torques = 'optimal_controls_shoulder_torques'
            # self.pub_optimal_trajectory = rospy.Publisher(self.topic_optimal_trajectory_shoulder, Float64MultiArray, queue_size=1)
            # self.pub_optimal_torques = rospy.Publisher(self.topic_optimal_controls_shoulder_torques, Float64MultiArray, queue_size=1)

            # # Set up the structure to deal with the new thread, to allow continuous publication of the optimal trajectory and torques
            # # The thread itself is created later, with parameters known at run time
            # self.x_opt_lock = threading.Lock()          # Lock for synchronizing access to self.x_opt
            # self.publish_thread = None                  # creating the variable that will host the thread

            # # create a subscriber to catch when the trajectory optimization should be running
            # self.flag_run_optimization = False
            # self.sub_run_optimization = rospy.Subscriber(shared_ros_topics['request_reference'], Bool, self._flag_run_optimization_cb, queue_size=1)


    def createMPCfunctionWithoutInitialGuesses(self):
        """
        Instantiates the CasADi function that maps between parameters of the NLP to its optimal solutions.
        Parameters can represent initial state and bounds on the variables (that can differ at each iteration),
        while the optimal solution should be optimal state trajectory and control inputs.
        """
        self.MPC_iter = self.nlp_module.createOptimalMapWithoutInitialGuesses()

    def createMPCfunctionInitialGuesses(self):
        """
        Instantiates the CasADi function that maps between parameters of the NLP to its optimal solutions.
        Parameters can represent initial state and bounds on the variables (that can differ at each iteration),
        while the optimal solution should be optimal state trajectory and control inputs. The initial guesses for
        the primal and dual variables should also be input to the function.
        """
        self.MPC_iter_initGuess = self.nlp_module.createOptimalMapInitialGuesses()


    def upsampleSolution(self, solution, N, T, target_freq):
        """
        The function is dedicated to upsampling a given solution for the trajectory of the shoulder pose
        (consisting of N points over a time horizon of T seconds) to a target frequency required.
        Note that only the generalized coordinates are upsampled, as we do not care about the velocities.
        The interpolation is linear over the initial datapoints! Using SciPy, we could interpolate also in
        a different way.
        """
        required_points = int(np.ceil(target_freq*T))
        upsampled_indices = np.linspace(0, solution.shape[1] - 1, required_points)
        solution = solution[0::2, :]
        upsampled_solution = np.array([np.interp(upsampled_indices, np.arange(solution.shape[1]), row) for row in solution])
        return upsampled_solution
    

class nlp_jetracer():
    """
    Class defining the nonlinear programming (NLP) problem to be solved at each iteration of the 
    trajectory optimization (TO) algorithm. It leverages CasADi to find the optimal 
    trajectory that the jetracer should follow in order to navigate its surroundings. 

    We call this a nlp_jetracer.
    """
    def __init__(self, car_number, car_length):
        """"
        Initialization in which we set only empty variables that will be modified by the user
        through dedicated functions.
        """
        self.T = None                   # time horizon for the optimization
        self.N = None                   # number of control intervals
        self.h = None                   # duration of each control interval

        # Initial condition and final desired goal (both expressed in state-space)
        self.x_0 = None
        self.goal = None

        # CasADi symbolic variables
        self.x = None                   # CasADi variable indicating the state of the system
        
        self.u = None                   # CasADi variable representing the control vector to be applied to the system

        # Naming and ordering of states and controls
        self.state_names = None
        self.xdim = 0
        self.control_names = None
        self.dim_u = 0

        # Type of collocation used and corresponding matrices
        self.pol_order = None
        self.collocation_type = None
        self.B = None                   # quadrature matrix
        self.C = None                   # collocation matrix
        self.D = None                   # end of the interval

        # cost function
        self.cost_function = None       # CasADi function expressing the cost to be minimized
        self.gamma_goal = 0             # weight of the distance to the goal [1/(rad^2)]
        self.gamma_throttle = 0           # weight for the control torques used
        self.gamma_acceleration = 0     # weight for the coordinates' accelerations

        # CasADi optimization problem
        self.opti = ca.Opti()
        self.nlp_is_formulated = False              # has the nlp being built with formulateNLP()?
        self.solver_options_set = False             # have the solver options being set by the user?
        self.default_solver = 'ipopt'
        self.default_solver_opts = {'ipopt.print_level': 0,
                                    'print_time': 0, 
                                    'error_on_fail': 1,
                                    'ipopt.tol': 1e-3,
                                    'expand': 0,
                                    'ipopt.hessian_approximation': 'limited-memory'}


        # parameters of the NLP
        self.params_list = []           # it collects all the parameters that are used in a single instance
                                        # of the NLP
        
        # solution of the NLP
        self.Xs = None                  # the symbolic optimal trajectory for the state x
        self.Us = None                  # the symbolic optimal sequence of the controls u
        self.Js = None                  # the symbolic expression for the cost function
        self.solution = None            # the numerical solution (where state and controls are mixed together)
        self.x_opt = None               # the numerical optimal trajectory for the state x
        self.u_opt = None               # the numerical optimal sequence of the controls u
        self.lam_g0 = None              # initial guess for the dual variables (populated running solveNLPonce())
        self.solution_is_updated = False

        # which car is being used
        self.car_number = car_number
        self.car_length = car_length

        # depending on the car, define steering input - steering angle relationship
        if self.car_number==1:
            self.b = -0.49230292
            self.c = -0.01898136
        elif self.car_number==2:
            self.b = -0.4306743
            self.c = -0.0011013203
        elif self.car_number==3:
            self.b = -0.4677383
            self.c = 0.013598954

        # defining acceleration function
        #fitted from same data as GP for ICRA 2024 (Lyons, Franzese & Ferranti)
        self.v_friction = 1.0683593
        self.v_friction_static = 1.1530068
        self.v_friction_static_tanh_mult = 23.637709
        self.v_friction_quad = 0.09363517
        
        self.tau_offset = 0.16150239
        self.tau_offset_reverse = 0.16150239
        self.tau_steepness = 10.7796755
        self.tau_steepness_reverse = 90
        self.tau_sat_high = 2.496312
        self.tau_sat_high_reverse = 5.0


    def setTimeHorizonAndDiscretization(self, N, T):
        self.T = T      # time horizon for the optimal control
        self.N = N      # numeber of control intervals
        self.h = T/N


    def initializeStateVariables(self, x, names):
        self.x = x
        self.state_names = names
        self.dim_x = x.shape[0]


    def initializeControlVariables(self, u, names):
        self.u = u
        self.control_names = names
        self.dim_u = u.shape[0]


    def populateCollocationMatrices(self, order_polynomials, collocation_type):
        """
        Here, given an order of the collocation polynomials, the corresponding matrices are generated
        The collocation polynomials used are Lagrange polynomials.
        The collocation type determines the collocation points that are used ('legendre', 'radau').
        """
        # Degree of interpolating polynomial
        self.pol_order = order_polynomials

        # Get collocation points
        tau = ca.collocation_points(self.pol_order, collocation_type)

        # Get linear maps
        self.C, self.D, self.B = ca.collocation_coeff(tau)


    def setInitialState(self, x_0):
        self.x_0 = x_0


    def setGoal(self, goal):
        self.goal = goal
     
    
    def formulateNLP(self, constraint_list, initial_guess_prim_vars = None, initial_guess_dual_vars = None):
        """"
        This function creates the symbolic structure of the NLP problem.
        For now it is still not very general, needs to be rewritten if different problems
        are to be solved.
        """

        if self.goal is None:
            RuntimeError("Unable to continue. The goal of the NLP has not been set yet. \
                         Do so with setGoal()!")
        if self.x_0 is None:
            RuntimeError("Unable to continue. The initial state of the NLP has not been set yet. \
                         Do so with setInitialState()!")
        
        # create the function capturing the dynamics of the robot
        x_dot = self.x[3]*ca.cos(self.x[2])
        y_dot = self.x[3]*ca.sin(self.x[2])
        eta_dot = self.x[3]/self.car_length * ca.tan((self.c + self.u[1])/self.b)

        # friction model
        static_friction = ca.tanh(self.v_friction_static_tanh_mult  * self.x[3]) * self.v_friction_static
        v_contribution = - static_friction - self.x[3] * self.v_friction - ca.sign(self.x[3]) * self.x[3] ** 2 * self.v_friction_quad 
        
        #for positive throttle
        th_activation1 = (ca.tanh((self.u[0] - self.tau_offset) * self.tau_steepness) + 1) * self.tau_sat_high
        
        #for negative throttle
        th_activation2 = (ca.tanh((self.u[0] + self.tau_offset_reverse) * self.tau_steepness_reverse)-1) * self.tau_sat_high_reverse
        throttle_contribution = (th_activation1 + th_activation2) 
        Fx = throttle_contribution + v_contribution
        Fx_r = Fx * 0.5
        Fx_f = Fx * 0.5

        v_dot =  Fx_r + Fx_f

        state_dot = ca.vertcat(x_dot, y_dot, eta_dot, v_dot)

        # define also the cost function (without acceleration term yet, will be added in the NLP formulation)
        L = self.gamma_goal *ca.norm_2(self.x-self.goal) + \
            self.gamma_throttle * self.u[0]**2 + \
            self.gamma_steering * self.u[1]**2

        # CasADi function that evaluates cost and state derivative given the current state and control vectors
        self.sys_dynamics = ca.Function('cost_function', [self.x, self.u], [L, state_dot])

        # set cost
        J = 0

        # decode control limits
        u_max = constraint_list['tau_max']
        u_min = constraint_list['tau_min']

        assert constraint_list['tau_max'] == constraint_list['sigma_max'], "The two values must be equal!"
        assert constraint_list['tau_min'] == constraint_list['sigma_min'], "The two values must be equal!"


        # initialize empty list for the parameters used in the problem
        self.params_list = []

        #  "Lift" initial conditions
        Xk = self.opti.variable(self.dim_x)
        p = self.opti.parameter(self.dim_x)     # parametrize initial condition
        self.params_list.append(p)              # add the parameter to the list of parameters for the NLP
        self.opti.subject_to(Xk==p)

        # Collect all states/controls
        Xs = [Xk]
        Us = []

        # formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = self.opti.variable(self.dim_u)
            Us.append(Uk)
            self.opti.subject_to(u_min <= Uk)
            self.opti.subject_to(Uk <= u_max)

            # optimization variable (state) at collocation points
            Xc = self.opti.variable(self.dim_x, self.pol_order)

            # evaluate ODE right-hand-side and cost at collocation points
            ode, quad = self.sys_dynamics(Xc, Uk)

            # add contribution to quadrature function
            J = J + self.h*ca.mtimes(quad, self.B)

            # get interpolating points of collocation polynomial
            Z = ca.horzcat(Xk, Xc)

            # get slope of interpolating polynomial (normalized)
            Pidot = ca.mtimes(Z, self.C)

            # match with ODE right-hand-side (enforce dynamics at collocation points)
            self.opti.subject_to(Pidot==self.h*ode)

            # state at the end of collocation interval
            Xk_end = ca.mtimes(Z, self.D)

            # new decision variable for state at the end of interval
            Xk = self.opti.variable(self.dim_x)
            Xs.append(Xk)

            # continuity constraint
            self.opti.subject_to(Xk_end==Xk)

        # adding constraint to reach the final desired state
        self.opti.subject_to(Xk[3]==self.goal[3])   # constraint on final (zero) velocity

        self.Us = ca.vertcat(*Us)
        self.Xs = ca.vertcat(*Xs)

        # define the cost function to be minimized, and store its symbolic expression
        self.opti.minimize(J)
        self.Js = J

        # explicitly provide an initial guess for the primal variables
        if initial_guess_prim_vars is not None:
            self.opti.set_initial(initial_guess_prim_vars)

        # explicitly provide an initial guess for the dual variables
        if initial_guess_dual_vars is None:
            initial_guess_dual_vars = np.zeros((self.opti.lam_g.shape))

        self.opti.set_initial(self.opti.lam_g, initial_guess_dual_vars)

        # set the values of the parameters
        self.opti.set_value(p, self.x_0)

        # set flag indicating process was successfull
        self.nlp_is_formulated = True


    def setSolverOptions(self, solver, opts):
        """
        This function allows to set the solver and the solver options that will be used
        when solving the NLP. It sets to true the corresponding flag, so that other methods
        can operate safely.
        """
        self.opti.solver(solver, opts)
        self.solver_options_set = True


    def solveNLPOnce(self):
        """
        This function solves the NLP problem that has been formulated, assuming that the constraints,
        the initial position, the goal, the cost function and the solver have been specified already.
        It retrieves the optimal trajectory for the state and control variables using the symbolic 
        mappings computes in formulateNLP(), storing those variable and returning them to the caller 
        as well.
        """
        # change the flag so that others know that the current solution is not up-to-date
        self.solution_is_updated = False

        if self.nlp_is_formulated == False:
            RuntimeError("The NLP problem has not been formulated yet! \
                         Do so with the formulateNLP() function")
            
        if self.solver_options_set == False:
            print("No user-provided solver options. \
                  Default solver options will be used. You can provide yours with setSolverOptions()")
            self.setSolverOptions(self.default_solver, self.default_solver_opts)

        self.solution = self.opti.solve()

        self.x_opt = self.solution.value(self.Xs)
        self.u_opt = self.solution.value(self.Us)
        self.J_opt = self.solution.value(self.Js)
        self.lam_g0 = self.solution.value(self.opti.lam_g)

        # change the flag so that others know that the current solution is up to date
        self.solution_is_updated = True

        return self.x_opt, self.u_opt, self.solution
    

    def createOptimalMapWithoutInitialGuesses(self):
        """
        Provides a utility to retrieve a CasADi function out of an opti object, once the NLP stucture 
        has been formulated. It does formally not require solving the NLP problem beforehand.
        The function that will be generated can be used as follows (adapting it to your case):

        numerical_outputs_list = MPC_iter(numerical_values_for_parameters)

        The generated function does not allow warm-starting it.
        """

        if self.nlp_is_formulated == False:
            RuntimeError("Unable to continue. The NLP problem has not been formulated yet \
                         Do so with formulateNLP()!")
        
        symbolic_output_list = [self.Us, self.Xs, self.Js] 

        # inputs to the function
        input_list = self.params_list.copy()       # the parameters that are needed when building the NLP

        MPC_iter = self.opti.to_function('MPC_iter', input_list, symbolic_output_list)
        return MPC_iter


    def createOptimalMapInitialGuesses(self):
        """
        Provides a utility to retrieve a CasADi function out of an opti object, once the NLP stucture 
        has been formulated. It does formally not require solving the NLP problem beforehand.
        However, you should first run an instance of solveNLPonce() so that a good initial guess for
        primal and dual variables for the problem are used - this could speed up the some solvers.
        The function that will be generated can be used as follows (adapting it to your case):

        numerical_outputs_list = MPC_iter([numerical_values_for_parameters, init_guess_prim, init_guess_dual])

        The generated function needs as inputs the initial guesses for both primal and dual variables.
        """

        if self.nlp_is_formulated == False:
            RuntimeError("Unable to continue. The NLP problem has not been formulated yet \
                         Do so with formulateNLP()!")
            
        # inputs to the function
        input_list = self.params_list.copy()       # the parameters that are needed when building the NLP
        if self.solution is None:
            RuntimeError('No inital guess can be used for primal variables!\
                         Run solveNLPonce() first. \n')
        else:
            input_list.append(self.Us)       # the initial guess for the controls
            input_list.append(self.Xs)       # the initial guess for the state trajectory

        if self.lam_g0 is None:
            RuntimeError('No inital guess can be used dual variables! \
                         Run solveNLPonce() first \n')
        else:
            input_list.append(self.opti.lam_g)  # the inital guess for the dual variable

        symbolic_output_list = [self.Us, self.Xs, self.opti.lam_g, self.Js]

        MPC_iter = self.opti.to_function('MPC_iter', input_list, symbolic_output_list)
        return MPC_iter
    

    def getSizePrimalVars(self):
        """
        This function allows to retrieve the dimension of the primal variables of the problem, after it 
        has been solved at least once
        """
        if self.Xs is None or self.Us is None:
            RuntimeError('No stored values for primal variables!\
                         Run solveNLPonce() first. \n')
        else:
            return (self.Xs.shape, self.Us.shape)
    

    def getSizeDualVars(self):
        """
        This function allows to retrieve the dimension of the dual variables of the problem, after it 
        has been solved at least once
        """

        if self.lam_g0 is None:
            RuntimeError('No stored values for dual variables! \
                         Run solveNLPonce() first \n')
        else:
            return np.shape(self.lam_g0)


    def setCostWeights(self, goal = 0, throttle = 0, steering = 0, acceleration = 0):
        """
        Utility to set the weights for the various terms in the cost function of the NLP.
        """
        self.gamma_goal = goal                  # weight of the distance to the goal [1/(rad^2)]
        self.gamma_throttle = throttle            # weight for the control throttle used
        self.gamma_steering = steering          # weight for the control throttle used
        self.gamma_acceleration = acceleration  # weight on the accelerations produced
