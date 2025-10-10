import math
import torch
from cartpole import CartPoleEnv
from ddp import DDP  # your existing DDP class

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = CartPoleEnv(
        num_steps=300,
        dt=0.05,
        mp=0.1,
        mc=1.0,
        l=1.0,
        G=9.80665,
        device=device,
    )

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

    # initial state: pole hanging down (theta = pi)
    init = torch.tensor([[0.0, 0.0, math.pi, 0.0]], dtype=torch.float32, device=device)

    # run optimization
    actions, states = ddp.solve(init_state=init, num_iterations=60)
    print("\n[DDP] Optimization done.")
    print("[Final theta (rad)]:", states[-1][0, 2].item())

    # rollout and render
    def policy_fn(t, x):
        if t >= len(actions):
            return torch.zeros((1, 1), device=device, dtype=torch.float32)
        return actions[t].reshape(1, 1).to(device)

    traj = env.simulate(init, policy_fn)


