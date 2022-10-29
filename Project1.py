# overhead
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim

# # I am using the given environment parameters
FRAME_TIME = 0.1  # time interval
GRAVITY_ACCEL = 0.12  # gravity constant
BOOST_ACCEL = 0.18  # thrust constant


# define system dynamics
class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    def forward(self, state, action):
        """
        action[0] = thrust controller
        action[1] = omega controller
        state[0] = y
        state[1] = ydot
        state[2] = x
        state[3] = xdot
        state[4] = theta
        """
        # Apply gravity
        # Change in ydot due to gravity per time interval
        delta_state_gravity = torch.tensor([0., -GRAVITY_ACCEL * FRAME_TIME, 0., 0., 0.])

        # Thrust
        # 5 states, need tensor 5x1
        L = len(state)
        state_tensor = torch.zeros((L, 5))

        state_tensor[:, 1] = torch.cos(state[:, 4]) # ydot affected by cos theta of thrust
        state_tensor[:, 3] = -torch.sin(state[:, 4]) # xdot affected by -sin theta of thrust

        # Change in state due to thrust per time interval
        delta_state = BOOST_ACCEL * FRAME_TIME * torch.mul(state_tensor, action[:, 0].reshape(-1, 1))

        # Change in theta per time interval
        delta_state_theta = FRAME_TIME * torch.mul(torch.tensor([0., 0., 0., 0, -1.]), action[:, 1].reshape(-1, 1))

        # Current state with gravity, thrust, and theta
        state = state + delta_state + delta_state_gravity + delta_state_theta

        # Update state
        step_mat = torch.tensor([[1., FRAME_TIME, 0., 0., 0.],
                                 [0., 1., 0., 0., 0.],
                                 [0., 0., 1., FRAME_TIME, 0.],
                                 [0., 0., 0., 1., 0.],
                                 [0., 0., 0., 0., 1.]])

        state = torch.matmul(step_mat, state.T)

        return state.T # need to transpose state

# Deterministic controller

class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to me
        """
        super(Controller, self).__init__()  # Neural network
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            nn.Sigmoid(),
            nn.Linear(dim_output, dim_input),
            nn.Tanh(),
            nn.Linear(dim_input,dim_output),
            nn.Tanh())


    def forward(self, state):
        action = self.network(state)   # thrust action between 0 and 1
        return action

class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T, L):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.L = L
        self.theta_trajectory = torch.empty((1, 0))
        self.u_trajectory = torch.empty((1, 0))

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller(state)
            state = self.dynamics(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state():

        state = torch.rand((L, 5)) # Initial conditions for the rocket in interval 0 to 1
        state[:, 1] = 0  # vx = 0
        state[:, 3] = 0  # vy = 0

        return torch.tensor(state, requires_grad=False).float()

    def error(self, state):
        return torch.sum(state ** 2) # sum of loss

class Optimize:

    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)
        self.loss_list = []

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss     # keeps track of loss

        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            self.loss_list.append(loss)
            print('[%d] loss: %.3f' % (epoch + 1, loss))
        self.visualize()


    def visualize(self):
        data = np.array([[self.simulation.state_trajectory[i][L].detach().numpy()
        for i in range(self.simulation.T)]
        for L in range(self.simulation.L)])
        for i in range(self.simulation.L):
            y = data[i, :, 0]
            x = data[i, :, 2]
            vy = data[i, :, 1]
            vx = data[i, :, 3]
            theta = data[i, :, 4]

            fig1, axs = plt.subplots(2, 2)

            axs[0, 0].plot(x, y)
            axs[0, 0].set_title('Rocket Landing Changing Position')
            axs[0, 0].set_xlabel('X Position(m)')
            axs[0, 0].set_ylabel('Y Position(m)')

            axs[0, 1].plot(list(range(self.simulation.T)), vx)
            axs[0, 1].set_title('X Velocity vs. Time')
            axs[0, 1].set_xlabel('Time Step')
            axs[0, 1].set_ylabel('Vx (m/s)')

            axs[1, 0].plot(list(range(self.simulation.T)), vy)
            axs[1, 0].set_title('Y Velocity vs. Time')
            axs[1, 0].set_xlabel('Time Step')
            axs[1, 0].set_ylabel('Vy (m/s)')

            axs[1, 1].plot(list(range(self.simulation.T)), theta)
            axs[1, 1].set_title('Theta changing with time')
            axs[1, 1].set_xlabel('Time Step')
            axs[1, 1].set_ylabel('Theta(rad)')
        plt.show()


L = 10
T = 100 # number of time steps for 10 seconds
dim_input = 5       # state space dimensions
dim_hidden = 6  # latent dimensions for convergence purposes
dim_output = 2  # action space dimensions being controlled
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T, L)  # define simulation
o = Optimize(s)  # define optimizer
o.train(50)  # solve the optimization problem using 50 iterations
