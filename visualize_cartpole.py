import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
import io

def render_cartpole_frame(env, state, ax=None):
    """
    Draws one frame of the cart-pole system.
    `env` provides length info; `state` is a [1,4] tensor or list.
    """
    import torch, math
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy().flatten()
    pos, _, theta, _ = state

    cart_w, cart_h = 0.4, 0.2
    pole_len = env.l * 2.0
    axle_y = cart_h / 2.0
    pole_x = pos + pole_len * math.sin(theta)
    pole_y = axle_y + pole_len * math.cos(theta)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    ax.clear()
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title("CartPole Dynamics")
    ax.axhline(0, color='k')

    # Draw cart
    cart = patches.Rectangle((pos - cart_w/2, 0), cart_w, cart_h, fc='tab:gray')
    ax.add_patch(cart)

    # Draw pole
    ax.plot([pos, pole_x], [axle_y, pole_y], lw=3, color='tab:red')
    ax.add_patch(patches.Circle((pos, axle_y), 0.03, fc='tab:blue'))

    return ax


def save_cartpole_gif(env, states, filename="cartpole_ddp.gif", fps=15):
    """
    Converts a sequence of states into a smooth GIF animation.
    """
    frames = []
    fig, ax = plt.subplots(figsize=(6, 3))
    for s in states:
        render_cartpole_frame(env, s, ax=ax)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(Image.open(buf))
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000/fps),
        loop=0
    )
    plt.close(fig)
    print(f"Saved visualization to {filename}")
