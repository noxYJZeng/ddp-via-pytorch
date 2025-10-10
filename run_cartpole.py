import math
import torch
from cartpole import CartPoleEnv
from ddp import DDP

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment: energy-shaped running cost + soft barrier, NO terminal cost
    env = CartPoleEnv(
        num_steps=200,
        dt=0.02,
        u_max=2.0,
        device=device,
        q_theta=120.0,   # angle energy weight
        q_thetad=2.0,    # angular rate
        r_u=0.3,         # control effort
        theta_fail=0.6,  # fail cone ~34°
        w_fail=800.0,    # barrier strength
    )

    # DDP solver (your DDP class)
    ddp = DDP(
        env,
        eps=1e-3,
        success_multiplier=0.7,
        failure_multiplier=3.0,
        min_eps=1e-8,
        verbose=1,
        use_running_state_cost=True,
        seed=0,
    )

    # Start near-upright (DDP是局部法，给小扰动即可)
    init = torch.tensor([[0.0, 0.0, -0.05, 0.0]], device=device)  # ~5.7°

    actions, states = ddp.solve(init_state=init, num_iterations=25)
    print("\n[DDP] Optimization done.")
    print("[Final theta (rad)]:", states[-1][0, 2].item())

    # open-loop rollout with solved actions
    def policy_fn(t, x):
        if t >= len(actions):
            return torch.zeros((1, 1), device=device)
        return actions[t].reshape(1, 1)

    traj = env.simulate(init, policy_fn)
    env.save_gif(traj, filename="cartpole_keep_upright.gif", fps=30)
