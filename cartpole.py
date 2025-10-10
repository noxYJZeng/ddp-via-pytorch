import math
import io
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


class CartPoleEnv:
    """
    DDP-compatible cartpole environment (4D state).
    State: [x, x_dot, theta, theta_dot]
    Control: [u_raw]  ->  Force = tanh(u_raw)
    Dynamics: per Razvan V. Florian, no friction.
    """

    def __init__(
        self,
        num_steps=100,
        dt=0.05,
        mp=0.1,
        mc=1.0,
        l=1.0,
        G=9.80665,
        device=None,
        dtype=torch.float32,
    ):
        self.num_steps = int(num_steps)
        self.dt = float(dt)
        self.mp, self.mc, self.l, self.G = float(mp), float(mc), float(l), float(G)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        #ddp required
        self.state_dim = 4
        self.control_dim = 1
        self.timesteps = torch.arange(self.num_steps, device=self.device, dtype=self.dtype)

        #Goal state: upright, centered, zero velocities
        self.x_goal = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device, dtype=self.dtype)

        #Cost weights (same as in your Sympy reference)
        Q = torch.eye(4, device=self.device, dtype=self.dtype)
        Q[1, 1] = 0.0  # no cost on x_dot
        Q[3, 3] = 0.0  # no cost on theta_dot
        self.Q = Q
        self.R = 0.1 * torch.eye(1, device=self.device, dtype=self.dtype)
        self.Qf = 100.0 * torch.eye(4, device=self.device, dtype=self.dtype)

    # running costs
    def running_state_cost(self, t, x, u):
        err = x - self.x_goal
        return err @ self.Q @ err.transpose(0, 1)

    def running_control_cost(self, t, x, u):
        return u @ self.R @ u.transpose(0, 1)

    # terminal cost
    def terminal_cost(self, x_T):
        err = x_T - self.x_goal
        return err @ self.Qf @ err.transpose(0, 1)

    # dynamics
    def step(self, t, x, u):
        # x = [x, x_dot, theta, theta_dot]
        x_pos = x[:, 0:1]
        x_dot = x[:, 1:2]
        theta = x[:, 2:3]
        theta_dot = x[:, 3:4]

        # control saturation
        F = torch.tanh(u.to(self.device, self.dtype))

        # Florian equations (no friction)
        temp = (F + self.mp * self.l * (theta_dot ** 2) * torch.sin(theta)) / (self.mc + self.mp)
        numer = self.G * torch.sin(theta) - torch.cos(theta) * temp
        denom = self.l * (4.0 / 3.0 - self.mp * (torch.cos(theta) ** 2) / (self.mc + self.mp))
        theta_ddot = numer / denom
        x_ddot = temp - self.mp * self.l * theta_ddot * torch.cos(theta) / (self.mc + self.mp)

        # Euler integration
        x_next = torch.cat([
            x_pos + x_dot * self.dt,
            x_dot + x_ddot * self.dt,
            theta + theta_dot * self.dt,
            theta_dot + theta_ddot * self.dt,
        ], dim=1)
        return x_next

    #visualization
    def render_state(self, x, ax=None):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy().flatten()
        pos, _, theta, _ = x

        cart_w, cart_h = 0.4, 0.2
        pole_len = self.l * 1.8
        axle_y = cart_h / 2.0

        pos_draw = max(-2.0, min(2.0, float(pos)))
        pole_x = pos_draw + pole_len * math.sin(theta)
        pole_y = axle_y + pole_len * math.cos(theta)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))
        ax.clear()
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-0.5, 1.8)
        ax.set_aspect("equal")
        ax.axhline(0, color="k", linewidth=1)
        ax.set_title(f"CartPole | x={pos:+.2f}, θ={theta:+.2f}")

        cart = patches.Rectangle((pos_draw - cart_w / 2, 0), cart_w, cart_h, fc="k")
        ax.add_patch(cart)
        ax.add_patch(patches.Circle((pos_draw, axle_y), 0.03, fc="tab:blue"))
        ax.plot([pos_draw, pole_x], [axle_y, pole_y], lw=3, color="tab:red")
        return ax

    def save_gif(self, trajectory, filename="cartpole_ddp.gif", fps=30):
        fig, ax = plt.subplots(figsize=(6, 3))
        frames = []
        for x in trajectory:
            self.render_state(x, ax=ax)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            frames.append(Image.open(buf))
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        plt.close(fig)
        print(f"[Render] Saved GIF → {filename}")

    # rollout
    def simulate(self, x0, policy_fn):
        x = x0.clone()
        traj = [x.clone()]
        for t in range(self.num_steps):
            u = policy_fn(t, x).to(device=self.device, dtype=self.dtype)
            x = self.step(self.timesteps[t], x, u)
            traj.append(x.clone())
        return traj
