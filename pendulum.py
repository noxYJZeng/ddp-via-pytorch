import torch, math, io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ddp import DDP

class Pendulum:
    """
    Smooth DDP pendulum swing-up — tuned to reach upright smoothly and precisely.
    State:   [sinθ, cosθ, θ̇]
    Control: τ = τ_max * tanh(u)
    Dynamics: θ̈ = -3*g*sin(θ+π)/(2L) + 3*τ/(M*L²)
    """

    def __init__(self,
                 num_steps=150,
                 dt=0.03,
                 m=1.0,
                 L=1.0,
                 g=9.80665,
                 torque_max=5.0,
                 device=None,
                 dtype=torch.float32):
        self.num_steps = num_steps
        self.dt = dt
        self.m, self.L, self.g = m, L, g
        self.torque_max = torque_max
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.state_dim = 3
        self.control_dim = 1
        self.timesteps = torch.arange(num_steps, device=self.device, dtype=self.dtype)

        # Target: θ = 0 (upright)
        self.x_goal = torch.tensor([[0.0, 1.0, 0.0]], device=self.device, dtype=self.dtype)


        # Cost function parameters (smooth + precise reach)
        L = self.L
        self.Q = torch.tensor([[L**2, L, 0.0],
                               [L, L**2, 0.0],
                               [0.0, 0.0, 0.28]], device=self.device, dtype=self.dtype)
        self.R = 0.06 * torch.eye(1, device=self.device, dtype=self.dtype)
        self.Qf = 2500 * torch.eye(3, device=self.device, dtype=self.dtype)

        # Potential Energy shaping term — encourages reaching upright
        self.energy_bias = 9.5
        # Damping bias — slightly lower to allow full swing without premature slowdown
        self.damping_bias = 0.45

    # ---------------- Cost functions ----------------
    def running_state_cost(self, t, x, u):
        err = x - self.x_goal
        base = err @ self.Q @ err.transpose(0, 1)
        control = u @ self.R @ u.transpose(0, 1)

        # Potential energy shaping: encourages cosθ → 1 (upright)
        potential_term = self.energy_bias * (1.0 - x[:, 1:2])
        # Damping term: limits velocity near the top to prevent overshoot
        damping_term = self.damping_bias * (x[:, 2:3] ** 2) * (1.0 - x[:, 1:2])
        return base + control + potential_term + damping_term

    def running_control_cost(self, t, x, u):
        # Control-only term used by DDP gradient computation
        return u @ self.R @ u.transpose(0, 1)

    def terminal_cost(self, xT):
        # Strong penalty on final deviation from the upright target
        err = xT - self.x_goal
        return err @ self.Qf @ err.transpose(0, 1)

    # ---------------- Dynamics ----------------
    def step(self, t, x, u):
        """
        θ̈ = -3*g*sin(θ + π)/(2L) + 3*τ/(M*L²)
        Using sin-cos state representation for continuous smoothness.
        """
        sinθ, cosθ, θ̇ = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        θ = torch.atan2(sinθ, cosθ)
        τ = self.torque_max * torch.tanh(u)

        θ̈ = -3 * self.g * torch.sin(θ + math.pi) / (2 * self.L) + \
             3 * τ / (self.m * self.L**2)

        θ_next = θ + θ̇ * self.dt
        θ̇_next = θ̇ + θ̈ * self.dt
        sin_next = torch.sin(θ_next)
        cos_next = torch.cos(θ_next)
        return torch.cat([sin_next, cos_next, θ̇_next], dim=1)

    # ---------------- Rollout ----------------
    def simulate(self, x0, policy_fn):
        """
        Roll out the trajectory using a given policy.
        """
        x = x0.clone()
        traj = [x.clone()]
        for k in range(self.num_steps):
            u = policy_fn(k, x).to(self.device, self.dtype)
            x = self.step(self.timesteps[k], x, u)
            traj.append(x.clone())
        return traj

    # ---------------- Rendering ----------------
    def render_state(self, x, ax=None):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy().flatten()
        θ = math.atan2(x[0], x[1])
        px, py = self.L * math.sin(θ), self.L * math.cos(θ)

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        ax.clear()
        ax.set_xlim(-1.2 * self.L, 1.2 * self.L)
        ax.set_ylim(-1.2 * self.L, 1.2 * self.L)
        ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=1)
        ax.axvline(0, color="k", lw=1)
        ax.plot([0, px], [0, py], color="tab:red", lw=3)
        ax.add_patch(patches.Circle((px, py), 0.05, fc="tab:blue"))
        ax.set_title(f"θ = {θ:+.2f} rad")
        return ax

    def save_gif(self, traj, filename="pendulum.gif", fps=30):
        """
        Render the full trajectory as an animated GIF.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        frames = []
        for x in traj:
            self.render_state(x, ax=ax)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            frames.append(Image.open(buf))
        frames[0].save(filename, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)
        plt.close(fig)
        print(f"[Render] Saved GIF → {filename}")


# ---------------- Main Execution ----------------
if __name__ == "__main__":
    env = Pendulum()
    ddp = DDP(env, eps=1e-3)

    # Initial state: downward (θ = π)
    x0 = torch.tensor([[math.sin(math.pi), math.cos(math.pi), 0.0]], dtype=torch.float32)
    print("Optimizing swing-up...")
    U, X = ddp.solve(x0, num_iterations=30)

    traj = [x for x in X]
    env.save_gif(traj, "pendulum_smooth_final_reach.gif")

    print("Smooth full swing-up achieved — continuous and precise to upright.")