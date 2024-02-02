
import diff_gaussian_rasterization as dgr
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import _C as torch_backend
from diff_gaussian_rasterization import rasterize, rasterize_jit

import jax
import jax.numpy as jnp
import torch
import math
import numpy as np
import random
import time 
import matplotlib.pyplot as plt
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic=True


N = 200

####################
# Helpers, Constants
####################

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def test(arr):
    return arr.min().item(), arr.max().item(), arr.sum().item()

def reset(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

from typing import NamedTuple
class Intrinsics(NamedTuple):
    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float
    near: float
    far: float

def compare(t, j, atol=1e-6):
    return torch.allclose(t, jax_to_torch(j), atol=atol)

def max_err(t, j):
    return torch.max(torch.abs(t - jax_to_torch(j)))

def torch_to_jax(torch_array):
    return jnp.array(torch_array.detach().cpu().numpy())

def jax_to_torch(jnp_array, grad=True):
    return torch.tensor(np.array(jnp_array), requires_grad=grad, device=device)

default_seed = 1223
gt_seed = 1354



def render_jax_with_param_transform(means3D, colors_precomp, opacity, scales, rotations):
    image_width, image_height, fx,fy = intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy
    cx, cy, near, far = intrinsics.cx, intrinsics.cy, intrinsics.near, intrinsics.far
    color = rasterize_jit(
        means3D, 
        colors_precomp, 
        opacity, 
        scales, 
        rotations,
        image_width, image_height, fx,fy, cx,cy,near,far
    ) 
    return color



#############################
# Arguments
#############################
reset(default_seed)

intrinsics = Intrinsics(
    height=200,
    width=200,
    fx=303.0, fy=303.0,
    cx=100.0, cy=100.0,
    near=0.01, far=2.5
)

fovX = jnp.arctan(intrinsics.width / 2 / intrinsics.fx) * 2.0
fovY = jnp.arctan(intrinsics.height / 2 / intrinsics.fy) * 2.0
tan_fovx = math.tan(fovX)
tan_fovy = math.tan(fovY)

means3D = jax.random.uniform(jax.random.PRNGKey(default_seed), shape=(N, 3), minval=-0.5, maxval=0.5) + jnp.array([0.0, 0.0, 1.0])
opacity = jnp.exp(jax.nn.log_sigmoid(jnp.ones(shape=(N,1)) * 1.0))
scales = jnp.exp(jnp.ones((N,3)) * -4.5)  # PREPROC 
rotations = jnp.ones((N,4)) * -1.0  
colors_precomp = jax.random.uniform(jax.random.PRNGKey(default_seed), shape=(N,3), minval=0.0, maxval=1.0)
# sh = jax.random.uniform(jax.random.PRNGKey(default_seed), shape=(N,0), minval=0.0, maxval=1.0)
sh_jax = jnp.array([])

camera_pose_jax = jnp.eye(4)
_proj_matrix = getProjectionMatrix(0.01, 100.0, fovX, fovY).transpose(0,1).cuda()
view_matrix_torch = torch.transpose(torch.tensor(np.array(jnp.linalg.inv(camera_pose_jax))),0,1).cuda()
projmatrix_torch = view_matrix_torch @ _proj_matrix

view_matrix = jnp.array(view_matrix_torch.cpu().numpy())
projmatrix = jnp.array(projmatrix_torch.cpu().numpy())



####################################
# Set up GT
####################################
means3D_gt = jax.random.uniform(jax.random.PRNGKey(gt_seed), shape=(N, 3), minval=-0.5, maxval=0.5) + jnp.array([0.0, 0.0, 1.0])
opacity_gt = jnp.ones(shape=(N,1)); opacity_gt = jnp.exp(jax.nn.log_sigmoid(opacity_gt))
scales_gt = jnp.ones((N,3)) * -4.5; scales_gt = jnp.exp(scales_gt)
rotations_gt = jnp.ones((N,4)) * -1.0 
colors_precomp_gt = jax.random.uniform(jax.random.PRNGKey(gt_seed), shape=(N,3), minval=0.0, maxval=1.0)

color_gt_jax = render_jax_with_param_transform(
    means3D_gt, colors_precomp_gt, opacity_gt, scales_gt, rotations_gt,
)

# _, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(jnp.transpose(color_gt_jax[:3], (1,2,0)))
# ax1.imshow(jnp.transpose(color_gt_jax[:3], (1,2,0)))
plt.imsave("gt303.png", jnp.transpose(color_gt_jax[:3], (1,2,0)))


##########################################################################################
# FWD TEST
##########################################################################################
## Args

## Run JAX fwd
# vjp_rasterize_fwd_jit = jax.jit(jax.tree_util.Partial(jax.vjp, rasterize)) 
for _ in range(2):
    start = time.time()
    color_jax = render_jax_with_param_transform(
        means3D, colors_precomp, opacity, scales, rotations
    )
    end = time.time()
print(f"JAX FWD TIME={end-start} s")
plt.imsave("jax_fwd_0.png", jnp.transpose(color_jax[:3], (1,2,0)))
print("JAX min/max/sum: ", test(color_jax))

# ##########################################################################################
# # BWD TEST
# ##########################################################################################

def loss(means3D, colors_precomp, opacity, scales, rotations, color_gt):
    color = render_jax_with_param_transform(
        means3D, colors_precomp, opacity, scales, rotations,
    )
    return jnp.sum(0.5 * (color[:3] - color_gt[:3])**2), color
# loss = jax.tree_util.Partial(_loss, image_width=Intrinsics.width, image_height=Intrinsics.height, 
#                              fx=Intrinsics.fx, fy=Intrinsics.fy, 
#                              cx=Intrinsics.cx, cy=Intrinsics.cy, 
#                              near=Intrinsics.near, far=Intrinsics.far)
loss_grad = jax.value_and_grad(loss, argnums=(0,1,2,3,4,), has_aux=True)
loss_grad = jax.jit(loss_grad)

######## Jax optim ##########
print("\n------------\n Jax Optim \n----------\n")

import optax
it = 250

init_params = (means3D, colors_precomp, opacity, scales, rotations)
params = init_params
param_labels = ("means3D", "colors_precomp", "opacity", "scales", "rotations")
lr1, lr2 = 1e-3, 1e-4
tx = optax.multi_transform(
    {
        'means3D': optax.adam(lr1),
        'colors_precomp': optax.adam(lr1),
        'opacity': optax.adam(lr2),
        'scales': optax.adam(lr2),
        'rotations': optax.adam(lr2),
    },
    param_labels
)
# state = tx.init(params)

pbar = tqdm(range(it))
def inference_optax(params, tx, jit=False):
    def step(params, state):
        (loss_val_jax, color_out), gradients_jax = loss_grad(
            *params, color_gt_jax
        )
        # jax.debug.print("COLOR OUT={color_out_sum}", color_out_sum=color_out.sum())
        (dL_dmeans3D, dL_dcolors, dL_dopacity, dL_dscales, dL_drotations) = gradients_jax
        # jax.debug.print("🤯 gradient1={gradients_jax}, gradient2={gradients_jax1} gradient3={gradients_jax2} 🤯 loss={loss_val_jax}", gradients_jax=gradients_jax[0].sum(), gradients_jax1=gradients_jax[1].sum(), gradients_jax2=gradients_jax[2].sum(), loss_val_jax=loss_val_jax)

        updates, state = tx.update(gradients_jax, state, params)
        params = optax.apply_updates(params, updates)

        return params, state, loss_val_jax
    if jit:
        step = jax.jit(step)

    pbar = tqdm(range(it))
    state = tx.init(params)
    for _ in pbar:
        params, state, loss_val_jax = step(params, state)
        pbar.set_description(f"loss: {loss_val_jax}")
        losses.append(loss_val_jax.item())

    return params , losses
losses = []

print("\nOptax jitted")
params, losses = inference_optax(init_params, tx, jit=True)
print(losses)
losses = []
# print("\nOptax nonjitted")
# params, losses = inference_optax(init_params, tx, jit=False)
# print(losses)




# plot final jax
_color_final_jax = render_jax_with_param_transform(
        *params
    )
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(jnp.transpose(_color_final_jax[:3], (1,2,0)))
ax2.imshow(jnp.transpose(color_gt_jax[:3], (1,2,0)))
fig.savefig(f'jax_final_optim_{it}.png')


from IPython import embed; embed()

 