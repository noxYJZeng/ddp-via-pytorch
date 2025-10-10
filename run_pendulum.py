import math
import torch
from cartpole import CartPoleEnv
from ddp import DDP

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment with energy-shaped angle cost + soft failure barrier
    env = CartPoleEnv(
        num_steps=200,
        dt=0.02,
        u_max=2.0,
        device=device,
        # 这些权重已经偏向“稳定保持”，而不是“猛推到终点”
        q_theta=120.0, qf_theta=150.0, r_u=0.3,
        theta_fail=0.6, w_fail=800.0,
    )

    # DDP
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

    # start NEAR upright (θ ~ 0)，不要给太大初始偏差（DDP是局部法）
    init = torch.tensor([[0.0, 0.0, 0.10, 0.0]], device=device)  # 约 5.7°
    actions, states = ddp.solve(init_state=init, num_iterations=25)

    print("\n[DDP] Done. Final state:", states[-1].detach().cpu().numpy())

    # open-loop simulate with solved actions
    def policy_fn(t, x):
        if t >= len(actions):
            return torch.zeros((1, 1), device=device)
        return actions[t].reshape(1, 1)

    traj = env.simulate(init, policy_fn)
    env.save_gif(traj, filename="cartpole_keep_upright.gif", fps=30)
