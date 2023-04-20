import numpy as np
import pandas as pd
import torch.nn as nn
import torch

# Gymnasium imports
#import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box
#from gymnasium.envs.registration import register


# Import Torch.diffeq
from torchdiffeq import odeint, odeint_event

def process_data(datapath, vel_only=False):
    csv_data = pd.read_csv(datapath, header=None)
    xtraining = csv_data.to_numpy()
    tseries = xtraining[:, 0]
    if vel_only:
        # x and y are sinusoids, hard to model
        state = xtraining[:, 4:7]
    else:
        state = xtraining[:,1:8]

    u = xtraining[:, 8:]
    dt = tseries[1] - tseries[0]
    _, num_features = state.shape
    return xtraining, tseries, state, u, dt, num_features


class DynEnv(Env):
    def __init__(self,  
                 coeffs_o, #Initial coefficients
                 state_lims, # Lower and higher limits for the state vars
                 cmd_lims,  # Lower and higher limits for the state vars
                 n, # Number of state variables of the nonlinear system
                 Xu_data,
                 ncmds=2, # Number of inputs (control cmds)
                 xhist=45, # Past experiences
                 degrees=2, # Degrees of the polinomial nonlinear dyn system
                 dt=0.01, # Step Time for the dynamical system
                 done_steps=200, # number of iterations per episode
                 iter_per_steps=10, # number of dynamical system  iterations
                 state_o=None, # Initial state
                 max_coeff=4
                 ):
      
        """
        This is the description of the environment to train the dynamic system
        Corrector. It assumes the state to be nonlinear and described by polino-
        mial functions
                 coeffs_o,          nitial coefficients
                 state_lims,        Lower and higher limits for the state vars
                 n,                 Number of state variables of the nonlinear system
                 Xu_data,           Data collected from dynamical system
                 ncmds=2,           Number of inputs (control cmds)
                 xhist=45,          Past experiences
                 degrees=2,         Degrees of the polinomial nonlinear dyn system
                 dt=0.01,           Step Time for the dynamical system
                 done_steps=200,    Number of iterations per episode
                 iter_per_steps=10, Number of dynamical system  iterations
                 state_o=None, # Initial state
        The nonlinear format is x_dot = f(x) + u*g(x)
        """
        self.coeffs_o = coeffs_o[:, :].copy()
        self.n = n
        self.ncmds = ncmds
        self.done_steps = done_steps
        self.iter_per_steps = iter_per_steps
        self.xhist = xhist
        self.Xu = Xu_data
        self.current_step = 0
        self.t = Xu_data[0, 0:xhist]
        self.done = False
        self.info = {}
        # Normalization info of the states
        self.X = Xu_data[1:n+1]
        self.u = Xu_data[n:]
        stats = self.normalization_parameters(self.X,self.u)
        self.mean_states, self.std_states, self.mean_cmd, self.std_cmd =  stats
        self.lenX = len(self.X[0])

        n_of_coeffs = 1

        for p in range(degrees):
            denominator = np.math.factorial(n - 1)*np.math.factorial(p+1)
            numerator = np.math.factorial(n + (p + 1) - 1)
            coefs = numerator / denominator
            n_of_coeffs += coefs

        self.n_of_coeffs = int(n_of_coeffs*(ncmds + 1))

        # Actions we can take, down, stay, up
        self.coef_indexes = coeffs_o != 0
        action_shape = coeffs_o[self.coef_indexes].shape
        self.action_space = Box(low=0.9*np.ones(action_shape), high=1.1*np.ones(action_shape), dtype=np.float32)


        #Reward Threshold
        self.reward_threshold = 179.99

        # State and cmd ranges
        low_state_lim_traj = state_lims[0]
        high_state_lim_traj = state_lims[1]
        low_cmd_lim_traj = cmd_lims[0]
        high_cmd_lim_traj = cmd_lims[1]
        
        low_state_lim_coeff = self.coeffs_o[self.coef_indexes]

        self.max_coeff = max_coeff
        low_state_lim_coeff[low_state_lim_coeff < 0] = low_state_lim_coeff[low_state_lim_coeff < 0]*max_coeff
        low_state_lim_coeff[low_state_lim_coeff > 0] = low_state_lim_coeff[low_state_lim_coeff > 0]*0

        high_state_lim_coeff = self.coeffs_o[self.coef_indexes]
        high_state_lim_coeff[high_state_lim_coeff > 0] = high_state_lim_coeff[high_state_lim_coeff > 0]*max_coeff
        high_state_lim_coeff[high_state_lim_coeff < 0] = high_state_lim_coeff[high_state_lim_coeff < 0]*0

        self.high_state_lim_coeff = high_state_lim_coeff
        self.low_state_lim_coeff = low_state_lim_coeff

        low_state_lim_traj = np.ndarray.flatten(low_state_lim_traj)
        high_state_lim_traj = np.ndarray.flatten(high_state_lim_traj)

        low_cmd_lim_traj = np.ndarray.flatten(low_cmd_lim_traj)
        high_cmd_lim_traj = np.ndarray.flatten(high_cmd_lim_traj)

        low_state_lim = np.concatenate((low_state_lim_coeff, low_state_lim_traj, low_cmd_lim_traj))
        high_state_lim = np.concatenate((high_state_lim_coeff, high_state_lim_traj, high_cmd_lim_traj))

        self.observation_space = Box(low=low_state_lim, high=high_state_lim)
        self.dt = dt
        self.reset()

    def update_state(self):
        coef_flat = np.ndarray.flatten(self._state["coefficients"][self.coef_indexes])
        traj_flat = np.ndarray.flatten(self._state["trajectory"])
        cmd_flat = np.ndarray.flatten(self._state["cmd"])
        self.state = np.concatenate((coef_flat, traj_flat, cmd_flat))

    def normalization_parameters(self, state, u):
        n_states, _ = state.shape
        n_cmd, _ = u.shape
        std_states = np.zeros((n_states,1))
        mean_states = np.zeros((n_states,1))
        std_cmd = np.zeros(n_cmd)
        mean_cmd = np.zeros(n_cmd)

        for i in range(n_states):
          std_states[i, :] = np.std(state[i, :])
          mean_states[i, :] = np.mean(state[i, :])

        for i in range(n_cmd):
          std_cmd[i] = np.std(u[i, :])
          mean_cmd[i] = np.mean(u[i, :])

        return mean_states, std_states, mean_cmd, std_cmd

    def get_reward(self, xhat, xtrue):
        x_norm = (xhat - self.mean_states)/self.std_states
        xtrue_norm = (xtrue - self.mean_states)/self.std_states
        x_diff = np.nan_to_num((x_norm - xtrue_norm)**2, nan=1e6)
        return np.exp(-x_diff).sum()

    def step(self, action):
        # Apply action, delta to the coefficients
        newstate = np.maximum(action*self._state["coefficients"][self.coef_indexes], self.low_state_lim_coeff)
        print(f"original state={newstate}")
        newstate = np.maximum(newstate, self.low_state_lim_coeff)
        newstate = np.minimum(newstate, self.high_state_lim_coeff)

        print(f"clipped state={newstate}")
        self._state["coefficients"][self.coef_indexes] = newstate

        # Iterate over the trajectory
        self.current_step += 1
        self.done = True if self.current_step > self.done_steps else False

        # Calculate reward
        with torch.no_grad():
            self.y0 = torch.tensor(self._state["trajectory"][:, 0]).T
            cmd = torch.tensor(self._state["cmd"]).T
            t = torch.tensor(self.t)

            func = Boat_Dynamic_System(torch.tensor(cmd), self._state["coefficients"], dt=self.dt)

            odeint_result = odeint(func, self.y0, t, method='rk4')
            self.odeint_result = odeint_result.T.numpy()
            #print(self.odeint_result)
            #print(self._state["trajectory"].shape)
        
        reward = self.get_reward(self.odeint_result, self._state["trajectory"])

        begin = self.initial_step + self.current_step*self.iter_per_steps
        end = begin + self.xhist
        self._state["trajectory"][:] = self.X[:, begin:end]
        self._state["cmd"][:] = self.u[:, begin:end]

        # Return step information
        result = self.state, reward, self.done, self.done, self.info
        self.update_state()
        
        if self.done:
            self.reset()

        return result

    def render(self):
        # Implement viz
        pass
    
    def reset(self, **kwargs):
        # Reset
        self.current_step = 0
        self.done = False


        # Simulation initial step
        self.initial_step = np.random.randint(0,
            self.lenX-self.done_steps*self.iter_per_steps-1)

        info = {'index': self.initial_step}
        # Simulation initial state
        begin = self.initial_step + self.current_step*self.iter_per_steps
        end = begin + self.xhist

        start_c = self.n + self.ncmds
        coeffs = self.coeffs_o.copy()
        self._state = {"trajectory": self.X[:,begin:end],
                      "cmd": self.u[:, begin:end],
                      "coefficients": coeffs}
        self.update_state()
        return self.state, info


class Boat_Dynamic_System(nn.Module):
    def __init__(self, cmd, coeffs, dt=1e-2, previous_steps=0):
        super().__init__()
        self.cmd = cmd
        self.dt = dt
        self.coeffs = coeffs
        self.previous_steps = previous_steps

    def Dif_Eq(self, u, v, r, Pf, rud, pro):
        Values = torch.tensor(
            [1, u, v, r, Pf, u ** 2, u * v, u * r, u * Pf, v ** 2, v * r, v * Pf, r ** 2, r * Pf, Pf ** 2,
             pro * 1, pro * u, pro * v, pro * r, pro * Pf, pro * u ** 2, pro * u * v, pro * u * r,
             pro * u * Pf, pro * v ** 2, pro * v * r, pro * v * Pf, pro * r ** 2, pro * r * Pf, pro * Pf ** 2,
             rud * 1, rud * u, rud * v, rud * r, rud * Pf, rud * u ** 2, rud * u * v, rud * u * r, rud * u * Pf,
             rud * v ** 2, rud * v * r, rud * v * Pf, rud * r ** 2, rud * r * Pf, rud * Pf ** 2])
        Indexes = self.coeffs[0]
        du = torch.sum(Values * self.coeffs[0])
        dv = torch.sum(Values * self.coeffs[1])
        dr = torch.sum(Values * self.coeffs[2])
        dPf = torch.sum(Values * self.coeffs[3])

        return torch.tensor((du, dv, dr, dPf))

    def forward(self, t, state):
        u = state[0]
        v = state[1]
        r = state[2]
        Pf = state[3]
        index = int(torch.round(t / self.dt))

        pro = self.cmd[index, 0]
        rud = self.cmd[index, 1]

        results = self.Dif_Eq(u, v, r, Pf, rud, pro)
        """
        du = 0.4000 * Pf - 2 * u + 0.0800 * r * (r + 10 * v)
        dv = 0.2211 * r - 4.9704 * v - 0.1693 * r * u + 0.0099 * rud * u
        dr = 0.8876 * v - 5.8681 * r - 0.0789 * r * u + 0.2959 * rud * u
        dPf = pro - Pf
        """

        """
        du = -2.000 * u + 0.400 * Pf + 0.800 * v * r + 0.080 * r**2
        dv = -4.625 * v + 0.426 * r + -0.159 * u * r
        dr = 0.996 * v + -5.804 * r + -0.075 * u * r + 0.293*rud*u
        dPf = -0.998 * Pf + 0.999 * pro
        """

        return results


def make_custom_env(data_path):
    coeffs = np.array([[0, -2, 0, 0, 0.4, 0, 0, 0, 0, 0, 0.8, 0, 0.08, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0],
                       [0, 0, - 4.9704, 0.2211, 0, 0, 0, - 0.1693, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0.0099, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0],
                       [0, 0, 0.8876, - 5.8681, 0, 0, 0, - 0.0789, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0.2959, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0]])

    n = 4
    xhist = 45
    low_state_lim = np.ones((n, xhist))
    hi_state_lim = np.ones((n, xhist))
    low_lims = [-5,  # u m/s
                -3,  # v m/s
                -3,  # r rad/s
                -20]  # Pf N
    hi_lims = [5,  # u m/s
               3,  # v m/s
               3,  # r rad/s
               30]  # Pf N

    for state in range(n):
        low_state_lim[state, :] = low_lims[state] * low_state_lim[state, :]
        hi_state_lim[state, :] = hi_lims[state] * hi_state_lim[state, :]

    state_lims = low_state_lim, hi_state_lim

    ncmd = 2
    low_cmd_lim = np.ones((ncmd, xhist))
    hi_cmd_lim = np.ones((ncmd, xhist))
    low_lims_cmd = [-250,  # Propeller reference for force
                    -250]  # Rudder torque per m/s
    hi_lims_cmd = [250,  # Propeller reference for force
                   250]  # Rudder torque per m/s

    for cmd in range(ncmd):
        low_cmd_lim[cmd, :] = low_lims[cmd] * low_cmd_lim[cmd, :]
        hi_cmd_lim[cmd, :] = hi_lims[cmd] * hi_cmd_lim[cmd, :]

    cmd_lims = low_cmd_lim, hi_cmd_lim

    coeffs_o = coeffs

    # xtraining, tseries, X, u, dt, num_features = process_data('/content/drive/MyDrive/Simulator/RL/Thesis_Cluster_Source/Thesis_Identification/Xu_Identification_Useed1025_Test21600s.csv', vel_only=True)
    xtraining, tseries, X, u, dt, num_features = process_data(data_path, vel_only=True)

    Xu_data = np.ndarray.transpose(np.concatenate((xtraining[:, 0:1], X, u), axis=1))
    # setup(name="gym_examples",
    #    version="0.0.1",
    #    install_requires=["gym==0.26.0"],) #, "pygame==2.1.0"],)
    #
    #
    # register(
    #    id='gym_examples/BoatEnv-v0',
    #    entry_point='gym_examples.envs:BoatEnv',
    #    max_episode_steps=300,
    # )

    return DynEnv(coeffs_o,  # Initial coefficients
                  state_lims,  # Lower and higher limits for the state vars
                  cmd_lims, # Lower and higher limits for the state vars
                  n,  # Number of state variables of the nonlinear system
                  Xu_data)