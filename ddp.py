import math
import gc
import numpy as np
import torch
from torch.func import jacrev, jacfwd, hessian, vjp
from tqdm import tqdm

def inv_reg(M: torch.Tensor, eps: float) -> torch.Tensor:
    #better than torch.linalg.inv because sometimes some metrics can not be inv directly
    #(M + eps * I) ^ (-1)
    #M is the metrix(d,d)

    d = M.shape[-1]
    I = torch.eye(d, device=M.device, dtype=M.dtype)

    #here we are solving (M + eps I) X = I. we return X = (M+epsI)^(-1)
    return torch.linalg.solve(M + eps * I, I)


class DDP:

    def __init__(self, env, eps: float = 1e-3, 
                 success_multiplier: float = 1.0,
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
        self.verbose = int(verbose)  # whether print info. 1/0
        self.use_running_state_cost = bool(use_running_state_cost)
        self.seed = seed


    def solve(self, init_state, actions=None, num_iterations=25):
        "init_state is x0, if it is pendulum, its state will be [angle, angular_velocity] -> (1,2)"
        "actions is the sequence of controls"
        "This is the most outside loop. Repeat ddp loop"

        T = self.env.num_steps
        d_u = self.env.control_dim  # the dimension of control

        if actions is None:
            actions = torch.zeros(T - 1, d_u, device=init_state.device, dtype=init_state.dtype)

        #if we have T = 5, d_u = 2, actions = torch.zeros(T - 1, d_u)
        #then we have (4,2)
        #[[0, 0],   ← u0
        # [0, 0],   ← u1
        # [0, 0],   ← u2
        # [0, 0]]   ← u3

        states = None
        end_states = []

        iters = tqdm(range(num_iterations))
        for it in iters:
            if self.seed is not None:
                torch.manual_seed(self.seed)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            states, actions = self.iterate(init_state, actions, iter_num=it, states=states)
            end_states.append(states[-1].detach().cpu())

        
        return actions.detach(), states


    def get_cost(self, states, actions):
        """
        states:  (T, 1, d_s) or list[(1, d_s)]
        actions: (T-1, d_u)
        return :running_cost, terminal_cost 
        -> am I good at each state in one ddp loop, how far am I from the ideal state at the end of the loop
        """   

        running_cost = torch.tensor(0.0, device=actions.device, dtype=actions.dtype)
        
        for t, x, u in zip(self.env.timesteps[:-1], states[:-1], actions):  # fix: states[-1] → states[:-1]
            #here we exclude the final timestamp and state bucause there is no action at final step
            x = x.reshape(1, -1)
            u = u.reshape(1, -1)

            #here we devide running cost into control cost and state cost
            rc = self.env.running_control_cost(t, x, u).squeeze()
            running_cost = running_cost + rc

            if self.use_running_state_cost:
                rs = self.env.running_state_cost(t, x, u).squeeze()
                running_cost = running_cost + rs

        terminal_cost = self.env.terminal_cost(states[-1].reshape(1, -1)).squeeze()
        return running_cost, terminal_cost


    def _fx_fu_at(self, t_idx, x_t: torch.Tensor, u_t: torch.Tensor):
        """
        dynamic jacobian: forward process to update states
        we want to get fx, fu of timestamp t:
            fx = ∂f/∂x  (d_s, d_s), how the state affect the next state
            fu = ∂f/∂u  (d_s, d_u), how the control affect the next state
        f(x,u) = env.step(t, x, u).squeeze(0)
        """

        t = self.env.timesteps[t_idx]  # fix: self.self.env → self.env

        def f_on_vec(x_vec, u_vec):
            return self.env.step(t, x_vec.unsqueeze(0), u_vec.unsqueeze(0)).squeeze(0)
        
        # Get the jacobian according to the first var
        fx = jacrev(f_on_vec, argnums=0)(x_t.squeeze(0), u_t.squeeze(0))
        fu = jacrev(f_on_vec, argnums=1)(x_t.squeeze(0), u_t.squeeze(0))

        return fx, fu


    def _running_cost_derivs_at(self, t_idx, x_t: torch.Tensor, u_t: torch.Tensor):
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

        gx = jacrev(g_on_vecs, argnums=0)(x_t.squeeze(0), u_t.squeeze(0)).reshape(-1, 1)
        gu = jacrev(g_on_vecs, argnums=1)(x_t.squeeze(0), u_t.squeeze(0)).reshape(-1, 1)

        gxx = jacfwd(jacrev(g_on_vecs, argnums=0), argnums=0)(x_t.squeeze(0), u_t.squeeze(0))
        guu = jacfwd(jacrev(g_on_vecs, argnums=1), argnums=1)(x_t.squeeze(0), u_t.squeeze(0))

        gxu = jacfwd(jacrev(g_on_vecs, argnums=1), argnums=0)(x_t.squeeze(0), u_t.squeeze(0))
        gux = jacfwd(jacrev(g_on_vecs, argnums=0), argnums=1)(x_t.squeeze(0), u_t.squeeze(0))

        return gx, gu, gxx, guu, gxu, gux


    def _terminal_Vx_Vxx(self, x_T: torch.Tensor):
        """
        x_T: the final state we forward rollout. (1, ds)
        get gradient and hessian of terminal value at the final state:
        Vx  = ∂h/∂x (d_s,1)
        Vxx = ∂²h/∂x² (d_s,d_s)
        """
        def h_on_vec(x_vec):
            return self.env.terminal_cost(x_vec.unsqueeze(0)).squeeze()

        vx = jacrev(h_on_vec)(x_T.squeeze(0)).reshape(-1, 1) # (d_s,1)
        vxx = hessian(h_on_vec)(x_T.squeeze(0)) # (d_s,d_s)
        return vx, vxx


    def compute_gradients(self, states, actions):
        """
        this is used for get the cost of states from back,
        and compute corecction of u
        states:[x₀, x₁, …, x_T]
        actions:[u₀, u₁, …, u_{T-1}]

        return ks, Ks
        """

        T = actions.shape[0] + 1
        d_s = self.env.state_dim
        d_u = self.env.control_dim

        vx, vxx = self._terminal_Vx_Vxx(states[-1].reshape(1, d_s))

        #prepare to store the k_t and K_t at each timestamp
        ks = [None] * (T - 1)
        Ks = [None] * (T - 1)

        for t in reversed(range(T - 1)):
            x_t = states[t].reshape(1, d_s)
            u_t = actions[t].reshape(1, d_u)

            fx, fu = self._fx_fu_at(t, x_t, u_t)
            gx, gu, gxx, guu, gxu, gux = self._running_cost_derivs_at(t, x_t, u_t)

            fx  = fx.reshape(d_s, d_s)
            fu  = fu.reshape(d_s, d_u)
            gxx = gxx.reshape(d_s, d_s)
            guu = guu.reshape(d_u, d_u)
            gxu = gxu.reshape(d_s, d_u)# ∂²g/∂x∂u  (d_s, d_u)
            gux = gux.reshape(d_u, d_s)# ∂²g/∂u∂x  (d_u, d_s)
            vx  = vx.reshape(d_s, 1)
            vxx = vxx.reshape(d_s, d_s)


            Qxx = gxx + fx.T @ vxx @ fx
            Quu = guu + fu.T @ vxx @ fu
            Qux = gux + fu.T @ vxx @ fx
            Qxu = gxu + fx.T @ vxx @ fu

            def step_on_vec(x_vec, u_vec):
                return self.env.step(self.env.timesteps[t], x_vec.unsqueeze(0), u_vec.unsqueeze(0)).squeeze(0)
            _, vjp_fun = vjp(step_on_vec, x_t.squeeze(0), u_t.squeeze(0))
            fxT_Vx, fuT_Vx = vjp_fun(vx.squeeze(1))
            
            Qx = gx + fxT_Vx.reshape(d_s, 1)# Qx​=gx​+fxT​Vx′
            Qu = gu + fuT_Vx.reshape(d_u, 1)# Qu​=gu​+fuT​Vx′​

            Quu_inv = inv_reg(Quu, self.eps)

            k = - Quu_inv @ Qu
            K = - Quu_inv @ Qux


            #update
            vx  = Qx  - K.T @ Quu @ k
            vxx = Qxx - K.T @ Quu @ K

            ks[t] = k.detach()
            Ks[t] = K.detach()

        ks = torch.stack(ks) # (T-1, d_u, 1)
        Ks = torch.stack(Ks) # (T-1, d_u, d_s)
        return ks, Ks


    def update_actions(self, states, actions, ks, Ks, iter_num, num_tries=40):
        """
        forward update new state and new control
        """
        cur_state = states[0].reshape(1, -1)
        run_cost, ter_cost = self.get_cost(states, actions)
        orig_cost = run_cost + ter_cost

        alpha = 1.0

        for _ in range(num_tries):
            new_states = [cur_state]
            cand_actions = []

            for t, x_nom, u_nom, k, K in zip(self.env.timesteps[:-1], states, actions, ks, Ks):
                x_nom = x_nom.reshape(1, -1)
                res = new_states[-1] - x_nom  # (1, d_s) δxt​=xt​−xtnom​
                u_new = (u_nom.reshape(-1,1) + alpha*k + K @ res.T).T.reshape(1, -1)
                x_next = self.env.step(t, new_states[-1], u_new)

                new_states.append(x_next)
                cand_actions.append(u_new)

            cand_actions = torch.cat(cand_actions, dim=0)  # (T-1, d_u)
            cand_states  = torch.stack(new_states, dim=0)  # (T, 1, d_s)

            run_cost, ter_cost = self.get_cost(cand_states, cand_actions)
            cand_cost = run_cost + ter_cost

            if cand_cost < orig_cost:
                self.eps *= self.success_multiplier
                return cand_states, cand_actions

            alpha *= 0.5

        self.eps = max(self.eps * self.failure_multiplier, self.min_eps)
        if self.verbose:
            print(f"[DDP] line-search failed, eps={self.eps:.3e}")
        return states, actions


    def iterate(self, init_state, actions, iter_num, states=None):
        """
        The first time outside loop: we only have sequence of control first
        so we need to get a sequence of states
        and then backward to get k, K, and update the new controls and new states
        """
        with torch.no_grad():
            self.eps = float(np.clip(self.eps, a_min=self.min_eps, a_max=np.inf))
            
            if states is None:
                seq = [init_state.reshape(1, -1)]
                for t, u in zip(self.env.timesteps[:-1], actions):
                    nxt = self.env.step(t, seq[-1], u.reshape(1, -1))
                    seq.append(nxt)
                states = torch.stack(seq, dim=0)  # (T,1,d_s)

            ks, Ks = self.compute_gradients(states, actions)
            states, actions = self.update_actions(states, actions, ks, Ks, iter_num=iter_num)
            return states.detach(), actions.detach()
