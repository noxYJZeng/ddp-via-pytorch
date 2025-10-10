import torch
import math

class PendulumEnv:
    def __init__(self, num_steps=50, dt=0.05):
        self.g = 9.81
        self.l = 1.0
        self.m = 1.0
        self.dt = dt
        self.num_steps = num_steps

        # what ddp needs
        self.state_dim = 2      # [theta, omega]
        self.control_dim = 1    # [torque]
        self.timesteps = list(range(num_steps))

        # cost weight
        self.Q = torch.diag(torch.tensor([5.0, 0.5]))  # angle + angular vel
        self.R = torch.diag(torch.tensor([0.1]))       # control
        self.Qf = torch.diag(torch.tensor([10.0, 1.0])) # terminal cost

    def step(self, t, x, u):
        """
        x: (1,2) tensor -> [theta, omega]
        u: (1,1) tensor -> [torque]
        return next_state (1,2)
        """
        theta, omega = x[0, 0], x[0, 1]
        torque = u[0, 0]

        dtheta = omega
        domega = (-self.g / self.l) * torch.sin(theta) + torque / (self.m * self.l ** 2)

        theta_next = theta + self.dt * dtheta
        omega_next = omega + self.dt * domega

        x_next = torch.stack([theta_next, omega_next], dim=0).reshape(1, -1)
        return x_next

    def running_control_cost(self, t, x, u):
        """Control effort penalty: 0.5 * u^T R u"""
        return 0.5 * (u @ self.R @ u.T)

    def running_state_cost(self, t, x, u):
        """Deviation penalty: 0.5 * (x - x_goal)^T Q (x - x_goal)"""
        x_goal = torch.tensor([[0.0, 0.0]], dtype=x.dtype, device=x.device)#upright
        dx = x - x_goal
        return 0.5 * (dx @ self.Q @ dx.T)

    def terminal_cost(self, x):
        """Final deviation penalty"""
        x_goal = torch.tensor([[0.0, 0.0]], dtype=x.dtype, device=x.device)
        dx = x - x_goal
        return 0.5 * (dx @ self.Qf @ dx.T)


