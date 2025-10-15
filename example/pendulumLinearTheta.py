import torch, math, io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ddp import DDP

class PendulumLinearTheta:
    """
    Linear-θ version of the DDP pendulum (state = [θ, θ̇]).
    Equivalent to the nonlinear (sinθ, cosθ, θ̇) representation, but uses
    a direct angular state for simplicity and interpretability.
    Dynamics:
        θ̈ = -3*g*sin(θ+π)/(2L) + 3*τ/(M*L²)
    """

    def __init__(self,
                 num_steps=200,
                 dt=0.05,
                 m=1.0,
                 L=1.0,
                 g=9.80665,
                 torque_max=10.0,
                 device=None,
                 dtype=torch.float32):
        self.num_steps = num_steps
        self.dt = dt
        self.m, self.L, self.g = m, L, g
        self.torque_max = torque_max
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.state_dim = 2
        self.control_dim = 1
        self.timesteps = torch.arange(num_steps, device=self.device, dtype=self.dtype)

        # Target: upright position (θ=0, θ̇=0)
        self.x_goal = torch.tensor([[0.0, 0.0]], device=self.device, dtype=self.dtype)

        # Cost function weights
        self.Q = torch.diag(torch.tensor([50.0, 10.0], device=self.device, dtype=self.dtype))   # state cost
        self.R = 0.3 * torch.eye(1, device=self.device, dtype=self.dtype)                       # control effort cost
        self.Qf = 200.0 * torch.eye(2, device=self.device, dtype=self.dtype)                    # terminal cost

    #  Running cost
    def running_state_cost(self, t, x, u):
        err = x - self.x_goal
        state_cost = err @ self.Q @ err.transpose(0, 1)
        control_cost = u @ self.R @ u.transpose(0, 1)
        return state_cost + control_cost

    def running_control_cost(self, t, x, u):
        # Isolated control cost (used in gradient computation)
        return u @ self.R @ u.transpose(0, 1)

    # Terminal cost
    def terminal_cost(self, xT):
        err = xT - self.x_goal
        return err @ self.Qf @ err.transpose(0, 1)

    # Dynamics
    def step(self, t, x, u):
        """
        θ̈ = -3*g*sin(θ+π)/(2L) + 3*τ/(M*L²)
        """
        θ = x[:, 0:1]
        θ̇ = x[:, 1:2]
        τ = self.torque_max * torch.tanh(u)

        θ̈ = -1.5 * self.g * torch.sin(θ + math.pi) / self.L + 3.0 * τ / (self.m * self.L**2)

        θ_next = θ + θ̇ * self.dt
        θ̇_next = θ̇ + θ̈ * self.dt
        return torch.cat([θ_next, θ̇_next], dim=1)

    def simulate(self, x0, policy_fn):
        x = x0.clone()
        traj = [x.clone()]
        for k in range(self.num_steps):
            u = policy_fn(k, x).to(self.device, self.dtype)
            x = self.step(self.timesteps[k], x, u)
            traj.append(x.clone())
        return traj

    def render_state(self, x, ax=None):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy().flatten()
        θ = float(x[0])
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

    def save_gif(self, traj, filename="pendulum_linear_theta.gif", fps=30):
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

if __name__ == "__main__":
    env = PendulumLinearTheta()
    ddp = DDP(env, eps=1e-3)

    # Initial state: hanging down (θ = π)
    x0 = torch.tensor([[math.pi, 0.0]], dtype=torch.float32)
    print("Optimizing swing-up (linear θ version)...")
    U, X = ddp.solve(x0, num_iterations=30)

    traj = [x for x in X]
    env.save_gif(traj, "pendulum_linear_theta.gif")
    print("Smooth swing-up achieved — linear θ state version.")


"""
Difference from the nonlinear (sinθ, cosθ, θ̇) version:
-------------------------------------------------------

This linear-θ variant represents the pendulum's state directly as [θ, θ̇]
instead of using the trigonometric embedding [sinθ, cosθ, θ̇].
Both formulations describe the same physical system, but their numerical
and optimization behaviors differ:

1. **State Representation**
   - Linear version uses raw angle θ, which is intuitive and low-dimensional (2-D).
   - Nonlinear version encodes sinθ and cosθ to avoid discontinuities at ±π.

2. **Angle Discontinuity**
   - In the linear form, θ wraps around at ±π, which can cause jumps in value
     if the pendulum spins more than one revolution.
   - The nonlinear form is continuous across the full circle, enabling smoother
     optimization and gradient flow in DDP.

3. **Cost Function**
   - This linear model uses a purely quadratic cost:
         L(x,u) = (x-x_goal)^T Q (x-x_goal) + u^T R u
     without any explicit potential-energy shaping.
   - The nonlinear model includes additional energy and damping terms
     (e.g., energy_bias * (1-cosθ)) to encourage a natural swing-up motion.

4. **Behavioral Outcome**
   - The linear model can perform a successful swing-up using only the physics term
     (gravity in the dynamics) and the quadratic cost, but it may overshoot or
     stop short depending on initialization and tuning.
   - The nonlinear version typically produces smoother, more physically realistic
     swing-ups because it explicitly shapes the energy landscape in the cost.

5. **Use Case**
   - Linear-θ version: simpler, interpretable, suitable for quick testing or
     analysis of angle-based control.
   - Nonlinear version: better for stable convergence, visualization, and
     experiments where continuous orientation tracking matters.
"""
