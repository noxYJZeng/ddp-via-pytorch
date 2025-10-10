import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image

class CartPoleEnv:
    """
    Continuous differentiable CartPole for DDP.
    Goal: keep pole upright (θ≈0) as long as possible while staying near center.
    The optimization minimizes running cost; no terminal cost.
    """

    def __init__(
        self,
        num_steps=300,
        dt=0.02,
        m_c=1.0,
        m_p=0.1,
        l=0.5,
        g=9.81,
        u_max=2.5,
        # ---- running cost weights ----
        q_x=1.0,         # cart position (encourage center)
        q_xd=0.1,        # cart velocity
        q_theta=150.0,   # pole angle energy (1 - cos θ)
        q_thetad=3.0,    # angular velocity
        r_u=0.2,         # control effort
        # ---- soft barrier for large tilt ----
        theta_fail=0.7,  # ~40°
        w_fail=600.0,    # extra penalty beyond that
        device=None,
        dtype=torch.float32,
    ):
        self.num_steps = num_steps
        self.dt = dt
        self.m_c, self.m_p, self.l, self.g = m_c, m_p, l, g
        self.u_max = u_max

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # weights
        self.q_x, self.q_xd, self.q_theta, self.q_thetad = q_x, q_xd, q_theta, q_thetad
        self.r_u = r_u
        self.theta_fail = theta_fail
        self.w_fail = w_fail

        self.state_dim, self.control_dim = 4, 1

        # ✅ add timesteps for compatibility with DDP
        self.timesteps = torch.arange(self.num_steps, device=self.device, dtype=self.dtype)

    # ---------- helpers ----------
    @staticmethod
    def _angle_wrap(theta):
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def _angle_energy(self, theta):
        """Upright (θ=0)→0, Down (π)→2."""
        theta = self._angle_wrap(theta)
        return theta ** 2

    def _soft_fail(self, theta):
        th = self._angle_wrap(theta)
        excess = F.relu(torch.abs(th) - self.theta_fail)
        return excess**2

    # ---------- cost functions ----------
    def running_state_cost(self, t, x, u):
        x_p, x_d, th, th_d = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        cost = (
            self.q_x * x_p**2
            + self.q_xd * x_d**2
            + self.q_theta * self._angle_energy(th)
            + self.q_thetad * th_d**2
            + self.w_fail * self._soft_fail(th)
        )
        return cost

    def running_control_cost(self, t, x, u):
        return self.r_u * (u @ u.T)

    def terminal_cost(self, x_T):
        return torch.zeros((1, 1), device=x_T.device, dtype=x_T.dtype)

    # ---------- dynamics ----------
    def step(self, t, x, u):
        u = torch.clamp(u.to(self.device), -self.u_max, self.u_max)
        X, Xd, Th, Thd = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        F = u

        temp = (F + self.m_p * self.l * (Thd**2) * torch.sin(Th)) / (self.m_c + self.m_p)
        denom = self.l * (4.0 / 3.0 - (self.m_p * torch.cos(Th)**2) / (self.m_c + self.m_p))
        Thdd = (self.g * torch.sin(Th) - torch.cos(Th) * temp) / denom
        Xdd = temp - (self.m_p * self.l * Thdd * torch.cos(Th)) / (self.m_c + self.m_p)

        x_dot = torch.cat([Xd, Xdd, Thd, Thdd], dim=1)
        x_next = x + self.dt * x_dot

        # ✅ fixed: out-of-place wrap
        theta_wrapped = self._angle_wrap(x_next[:, 2:3])
        x_next = torch.cat([x_next[:, 0:2], theta_wrapped, x_next[:, 3:4]], dim=1)

        return x_next


    # ---------- rendering ----------
    def render_state(self, x, ax=None):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy().flatten()
        pos, _, theta, _ = x

        cart_w, cart_h = 0.4, 0.2
        pole_len = self.l * 2.0
        axle_y = cart_h / 2.0

        # upright = theta=0 (pole up)
        pole_x = pos + pole_len * math.sin(theta)
        pole_y = axle_y + pole_len * math.cos(theta)

        # clamp the cart position for drawing if it goes out of [-2, 2]
        pos_draw = max(-2.0, min(2.0, pos))

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))
        ax.clear()

        # ✅ fixed static range (like Gym)
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")

        ax.axhline(0, color="k")
        ax.set_title(f"Keep Upright | pos={pos:+.2f}, θ={theta:+.2f}")

        # draw the cart at clamped position
        cart = patches.Rectangle((pos_draw - cart_w/2, 0), cart_w, cart_h, fc='k')
        ax.add_patch(cart)

        # draw the pole attached to the cart (not the clamped position)
        ax.plot([pos_draw, pos_draw + pole_len * math.sin(theta)],
                [axle_y, axle_y + pole_len * math.cos(theta)],
                lw=3, color='tab:red')

        ax.add_patch(patches.Circle((pos_draw, axle_y), 0.03, fc='tab:blue'))
        return ax


    def save_gif(self, trajectory, filename="cartpole_keep_upright.gif", fps=30):
        fig, ax = plt.subplots(figsize=(6, 3))
        frames = []

        for x in trajectory:
            self.render_state(x, ax=ax)
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
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


    def simulate(self, x0, policy_fn):
        x = x0.clone()
        traj = [x.clone()]
        for t in range(self.num_steps):
            u = policy_fn(t, x)
            x = self.step(t, x, u)
            traj.append(x.clone())
        return traj
