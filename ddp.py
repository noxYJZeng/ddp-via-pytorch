import math
import gc
import numpy as numpy
import torch
from torch.func import jacrev, jacfwd, hessian, vjp
from tqdm import tqdm

def inv_ref(M: torch.Tensor, eps: float) -> torch.Tensor:
    #better than torch.linalg.inv because sometimes some metrics can not be inv directly
    #(M + eps * I) ^ (-1)
    #M is the metrix(d,d)

    d = M.shape[-1]
    I = torch.eye(d, device = M.device, dtype = M.dtype)

    #here we are solving (M + eps I) X = I. we return X = (M+epsI)^(-1)
    return torch.linalg.solve(M + eps * I, I)

class DDP:

    def __init__(self, env, eps: float = 1e-3, 
                sucess_multiplier: float = 1.0,
                failure_multiplier: float = 10.0,
                min_eps: float = 1e-8,
                verbose: int = 1,
                use_running_state_cost: bool = True,
                seed: int | None = None):

        self.env = env
        self.eps = float(eps)
        self.success_multiplier = float(success_multiplier)
        self.failure_multiplier = float(failure_multiplier)
        self.min_eps = float(min_eps)
        self.verbose = int(verbose)#whether print info. 1/0
        self.use_running_state_cost = bool(use_running_state_cost)
        self.seed = seed

    def solve(self, init_state, actions = None, num_iterations = 25):
        "init_state is x0, if it is pendulum, its state will be [angle, angular_velocity] -> (1,2)"
        "actions is the sequence of controls"
        "This is the most outside loop. Repeat ddp loop"
        T = self.env.num_steps
        d_u = self.env.control_dim # the dimension of control

        if actions is None:
            actions = torch.zeros(T - 1, d_u, device = init_state.device, dtype = init_state.dtype)

        #if we have T = 5, d_u = 2, actions = torch.zeros(T - 1, d_u)
        #then we have (4,2)
        #[[0, 0],   ← u0
        # [0, 0],   ← u1
        # [0, 0],   ← u2
        # [0, 0]]   ← u3

        states = None
        end_states = []

        iters = tqdm((range(num_iterations)))
        for it in iters:
            if self.seed is not None:
                torch.manual_seed(self.seed)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            states, actions = self.iterate(init_state, actions, iter_num=it, states=states)
            end_states.append(states[-1].detach().cpu())
        
        return actions.detach(), end_states

    def get_cost(self, states, actions):
        """
        states:  (T, 1, d_s) or list[(1, d_s)]
        actions: (T-1, d_u)
        return :running_cost, terminal_cost 
        -> am I good at each state in one ddp loop, how far am I from the ideal state at the end of the loop
        """   

        running_cost = torch.tensor(0.0, device = actions.device, dtype = actions.dtype)
        
        for t, x, u in zip(self.env.timesteps[:-1], states[-1], actions):#zip will match them as comb
        #here we exclude the final timestamp and state bucause there is no action at final step
            x = x.reshape(1, -1)
            u = u.reshape(1, -1)

            #here we devide running cost into control cost and state cost
            rc = self.env.running_control_cost(t, x, u).squeeze()
            running_cost = running_cost + rc

            if self.use_running_state_cost:
                rs = self.env.runing_state_cost(t, x, u).squeeze()
                running_cost = running_cost + rs

            terminal_cost = self.env.terminal_cost(states[-1].reshape(1, -1)).squeeze()
            
            return running_cost, terminal_cost

    def _fx_fu_at(self, t_idx, x_t : torch.Tensor, u_t:Tensor):
        """
        we want to get fx, fu of timestamp t:
            fx = ∂f/∂x  (d_s, d_s), how the state affect the next state
            fu = ∂f/∂u  (d_s, d_u), how the control affect the next state
        f(x,u) = env.step(t, x, u).squeeze(0)
        """

        t = self.self.env.timesteps[t_idx]

        def f_on_vec(x_vec, u_vec):
            return self.env.step(t, x_vec.unsqueeze(0), u_vec.unsqueeze(0)).squeeze(0)
        
        # Get the jacobian according to the first var
        fx = jacrev(f_on_vec, argnums = 0)(x_t.squeeze(0), u_t.squeeze(0))
        fu = jaccre(f_on_vec, argnums = 1)(x_t.squeeze(0), u_t.squeeze(0))

        return fx, fu


        def use_running_derives_at(self, t_idx, x_t : torch.Tensor, u_t:Tensor):
            """
            get derivative of running cost at each state:
                gx  = ∂g/∂x   (d_s, 1)
                gu  = ∂g/∂u   (d_u, 1)
                gxx = ∂²g/∂x² (d_s, d_s)
                guu = ∂²g/∂u² (d_u, d_u)
                gxu = ∂²g/∂x∂u (d_s, d_u)
                gux = ∂²g/∂u∂x (d_u, d_s)
            """

            t = self.env.timesteps[t_idx]

            def g_on_vecs(x_vec, u_vec):
                x1 = x_vec.unsqueeze(0)  # (1, d_s)
                u1 = u_vec.unsqueeze(0)  # (1, d_u)
                v = self.env.running_control_cost(t, x1, u1)

                if self.use_running_state_cost:
                    v = v + self.env.running_state_cost(t, x1, u1)
                
                return v.squeeze()

            lx = jacrev(g_on_vecs, argnums=0)(x_t.squeeze(0), u_t.squeeze(0)).reshape(-1, 1)
            lu = jacrev(g_on_vecs, argnums=1)(x_t.squeeze(0), u_t.squeeze(0)).reshape(-1, 1)

            lxx = jacfwd(jacrev(g_on_vecs, argnums=0), argnums=0)(x_t.squeeze(0), u_t.squeeze(0))
            luu = jacfwd(jacrev(g_on_vecs, argnums=1), argnums=1)(x_t.squeeze(0), u_t.squeeze(0))

            lxu = jacfwd(jacrev(g_on_vecs, argnums=1), argnums=0)(x_t.squeeze(0), u_t.squeeze(0))
            lux = jacfwd(jacrev(g_on_vecs, argnums=0), argnums=1)(x_t.squeeze(0), u_t.squeeze(0))

            return lx, lu, lxx, luu, lxu, lux


