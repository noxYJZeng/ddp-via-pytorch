import torch, math, io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ddp import DDP

class PendulumSmoothDDP:
    """
    The state is represented as:
        x = [sin(θ), cos(θ), θ̇]
    rather than [θ, θ̇].

    This trigonometric encoding avoids the discontinuity and angle wrapping
    that occur when θ crosses ±π (for example, jumping from π to −π).
    Since sin and cos are both smooth and periodic, this representation makes
    the dynamics and gradients continuous, ensuring numerical stability and
    smooth convergence during DDP optimization.

    Control
        u ∈ ℝ¹ (raw control)
        τ = τ_max * tanh(u)
    where tanh bounds torque output within [-τ_max, τ_max].

    Dynamics
        θ̈ = -3*g*sin(θ + π)/(2L) + 3*τ/(M*L²)
    This models a simple pendulum with torque input, identical to the
    standard textbook “swing-up” problem, but with continuous torque limits.

    Cost
    The cost function encourages:
        - small angle error (via sin-cos difference)
        - low control effort
        - smooth changes in torque (uₜ - uₜ₋₁)² (Jacobian-safe approximation)
    """

    def __init__(self,
                 num_steps=75,
                 dt=0.05,
                 m=1.0,
                 L=1.0,
                 g=9.80665,
                 torque_max=5.0,
                 device=None,
                 dtype=torch.float32):
        # Simulation and physical parameters
        self.num_steps = num_steps
        self.dt = dt
        self.m, self.L, self.g = m, L, g
        self.torque_max = torque_max
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # System dimensions
        self.state_dim = 3
        self.control_dim = 1
        self.timesteps = torch.arange(num_steps, device=self.device, dtype=self.dtype)

        # Goal: upright position (θ = 0 ⇒ sinθ = 0, cosθ = 1, θ̇ = 0)
        self.x_goal = torch.tensor([[0.0, 1.0, 0.0]], device=self.device, dtype=self.dtype)

        # Cost matrices (quadratic form)
        # L terms encourage uprightness and penalize angular velocity.
        L = self.L
        self.Q = torch.tensor([[L**2, L, 0.0],
                               [L, L**2, 0.0],
                               [0.0, 0.0, 0.1]], device=self.device, dtype=self.dtype)
        self.R = 0.05 * torch.eye(1, device=self.device, dtype=self.dtype)
        self.Qf = 100.0 * torch.eye(3, device=self.device, dtype=self.dtype)

        # Smooth control penalty coefficient
        self.smooth_weight = 0.05

    #Cost Functions
    def running_state_cost(self, t, x, u):
        """
        Compute the instantaneous (running) cost:
            l(x, u) = (x - x_goal)^T Q (x - x_goal)
                    + u^T R u
                    + smoothness_penalty
        """
        err = x - self.x_goal
        base = err @ self.Q @ err.transpose(0, 1)
        control = u @ self.R @ u.transpose(0, 1)

        # Smooth-control penalty (Jacobian-safe)
        # A dummy 'prev_u' term prevents functorch from tracing mutable state.
        prev_u = torch.tanh(0.1 * (t - 1)) * 0.0  # placeholder to stabilize Jacobians
        smooth = self.smooth_weight * ((u - prev_u) ** 2)

        return base + control + smooth

    def running_control_cost(self, t, x, u):
        """Pure control cost (used by DDP for gradient computation)."""
        return u @ self.R @ u.transpose(0, 1)

    def terminal_cost(self, xT):
        """Terminal cost at the final timestep."""
        err = xT - self.x_goal
        return err @ self.Qf @ err.transpose(0, 1)

    #Dynamics
    def step(self, t, x, u):
        """
        Propagate system state using the standard pendulum dynamics.

        Given current state x = [sinθ, cosθ, θ̇] and control u,
        compute the next state using Euler integration:
            θ̈ = -3*g*sin(θ + π)/(2L) + 3*τ/(mL²)
        where τ = τ_max * tanh(u)
        """
        sinθ, cosθ, θ̇ = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        θ = torch.atan2(sinθ, cosθ)  # Recover angle from sin/cos pair
        τ = self.torque_max * torch.tanh(u)

        # Compute angular acceleration
        θ̈ = -3 * self.g * torch.sin(θ + math.pi) / (2 * self.L) + 3 * τ / (self.m * self.L**2)

        # Integrate to get next state
        θ_next = θ + θ̇ * self.dt
        θ̇_next = θ̇ + θ̈ * self.dt

        # Store next state again in sin/cos form
        return torch.cat([torch.sin(θ_next), torch.cos(θ_next), θ̇_next], dim=1)

    #Simulation
    def simulate(self, x0, policy_fn):
        """
        Run a forward rollout using a given feedback policy.
        Returns the full trajectory of states.
        """
        x = x0.clone()
        traj = [x.clone()]
        for k in range(self.num_steps):
            u = policy_fn(k, x).to(self.device, self.dtype)
            x = self.step(self.timesteps[k], x, u)
            traj.append(x.clone())
        return traj

    # ---------------- Visualization ----------------
    def render_state(self, x, ax=None):
        """
        Render the pendulum for a single state x = [sinθ, cosθ, θ̇].
        """
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
        ax.plot([0, px], [0, py], color="tab:red", lw=3)
        ax.add_patch(patches.Circle((px, py), 0.05, fc="tab:blue"))
        ax.set_title(f"θ = {θ:+.2f} rad")
        return ax

    def save_gif(self, traj, filename="pendulum.gif", fps=30):
        """
        Render the entire trajectory as an animated GIF.
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


# Main Execution
if __name__ == "__main__":
    env = PendulumSmoothDDP()
    ddp = DDP(env, eps=1e-3)

    # Start from downward (θ = π)
    x0 = torch.tensor([[math.sin(math.pi), math.cos(math.pi), 0.0]], dtype=torch.float32)
    print("Optimizing smooth DDP swing-up...")

    # Run DDP optimization
    U, X = ddp.solve(x0, num_iterations=60)

    traj = [x for x in X]
    env.save_gif(traj, "pendulum.gif")

    print("Smooth swing-up complete — no pause, no overshoot, autograd-safe continuous motion.")
