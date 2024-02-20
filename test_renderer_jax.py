
import diff_gaussian_rasterization as dgr
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import _C as torch_backend
from diff_gaussian_rasterization import rasterize, rasterize_jit, rasterize_with_depth

import jax
import jax.numpy as jnp
import torch
import math
import numpy as np
import psutil
import random
import time 
import matplotlib.pyplot as plt
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic=True

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = "platform"

N = int(2e6)

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

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0][0] = 2.0 * znear / (right - left)
    P[1][1] = 2.0 * znear / (top - bottom)
    P[0][2] = (right + left) / (right - left)
    P[1][2] = (top + bottom) / (top - bottom)
    P[3][2] = z_sign
    P[2][2] = z_sign * zfar / (zfar - znear)
    P[2][3] = -(zfar * znear) / (zfar - znear)
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

default_seed = 1223
gt_seed = 1354



def render_torch_with_param_transform(means3D, means2D, colors_precomp, opacity, scales, rotations,
           t_rasterizer):
    
    def expand_color(means3D, color):
        # return color
        return torch.hstack([
            color,
            means3D[:,2:3],
            torch.ones((means3D.shape[0], 1), device=color.device),
        ])
    
    color_and_depth ,_ = t_rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = expand_color(means3D, colors_precomp),
        opacities = opacity,
        scales = scales,
        rotations = rotations
    )
    return color_and_depth[:3, :, :], color_and_depth[3:4, :, :]


#############################
# Arguments
#############################
reset(default_seed)

intrinsics = Intrinsics(
    height=200,
    width=200,
    fx=300.0, fy=300.0,
    cx=100.0, cy=100.0,
    near=0.01, far=2.5
)

fovX = jnp.arctan(intrinsics.width / 2 / intrinsics.fx) * 2.0
fovY = jnp.arctan(intrinsics.height / 2 / intrinsics.fy) * 2.0
tan_fovx = math.tan(fovX)
tan_fovy = math.tan(fovY)

means3D = jax.random.uniform(jax.random.PRNGKey(default_seed), shape=(N, 3), minval=-0.5, maxval=0.5) + jnp.array([0.0, 0.0, 1.0])
opacity = jnp.exp(jax.nn.log_sigmoid(jnp.ones(shape=(N,1)) * 1.0))
scales = jnp.exp(jnp.ones((N,3)) * -10)  # PREPROC 
rotations = jnp.ones((N,4)) * -1.0  
colors_precomp = jax.random.uniform(jax.random.PRNGKey(default_seed), shape=(N,3), minval=0.0, maxval=1.0)
# sh = jax.random.uniform(jax.random.PRNGKey(default_seed), shape=(N,0), minval=0.0, maxval=1.0)
sh_jax = jnp.array([])

camera_pose_jax = jnp.eye(4)
_proj_matrix = jnp.transpose(getProjectionMatrix(0.01, 100.0, fovX, fovY))

view_matrix = jnp.transpose(jnp.linalg.inv(camera_pose_jax))
projmatrix = view_matrix @ _proj_matrix



####################################
# Set up GT
####################################
means3D_gt = jax.random.uniform(jax.random.PRNGKey(gt_seed), shape=(N, 3), minval=-0.5, maxval=0.5) + jnp.array([0.0, 0.0, 1.0])
opacity_gt = jnp.ones(shape=(N,1)); opacity_gt = jnp.exp(jax.nn.log_sigmoid(opacity_gt))
scales_gt = jnp.ones((N,3)) * -10; scales_gt = jnp.exp(scales_gt)
rotations_gt = jnp.ones((N,4)) * -1.0 
colors_precomp_gt = jax.random.uniform(jax.random.PRNGKey(gt_seed), shape=(N,3), minval=0.0, maxval=1.0)

color_gt_jax, depth_gt_jax = rasterize_with_depth(
    means3D_gt, colors_precomp_gt, opacity_gt, scales_gt, rotations_gt,
    intrinsics.width, intrinsics.height,
    intrinsics.fx, intrinsics.fy,
    intrinsics.cx, intrinsics.cy,
    intrinsics.near, intrinsics.far
)

_, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(jnp.transpose(color_gt_jax[:3], (1,2,0)))
ax1.imshow(jnp.transpose(color_gt_jax[:3], (1,2,0)))
plt.imsave("gt.png", jnp.transpose(color_gt_jax[:3], (1,2,0)))


##########################################################################################
# FWD TEST
##########################################################################################
## Args

## Run JAX fwd
# vjp_rasterize_fwd_jit = jax.jit(jax.tree_util.Partial(jax.vjp, rasterize)) 
for _ in range(2):
    start = time.time()
    color_jax, depth_jax = rasterize_with_depth(
        means3D, colors_precomp, opacity, scales, rotations,
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, 
        intrinsics.cx, intrinsics.cy,
        intrinsics.near, intrinsics.far
    )
    end = time.time()
print(f"JAX FWD TIME={end-start} s")
plt.imsave("jax_fwd_0.png", jnp.transpose(color_jax, (1,2,0)))
print("JAX min/max/sum: ", test(color_jax))

# ##########################################################################################
# # BWD TEST
# ##########################################################################################

def loss(means3D, colors_precomp, opacity, scales, rotations, color_gt, depth_gt):
    color, depth = rasterize_with_depth(
        means3D, colors_precomp, opacity, scales, rotations,
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, 
        intrinsics.cx, intrinsics.cy,
        intrinsics.near, intrinsics.far
    )
    return jnp.sum(0.5 * (color - color_gt)**2) + jnp.sum(0.5 * (depth - depth_gt)**2)
loss_grad = jax.value_and_grad(loss, argnums=(0,1,2,3,4,))
loss_grad = jax.jit(loss_grad)

######## Jax optim ##########
print("\n------------\n Jax Optim \n----------\n")

import optax
it = 250

init_params = (means3D, colors_precomp, opacity, scales, rotations)
params = init_params
param_labels = ("means3D", "colors_precomp", "opacity", "scales", "rotations")
lr1, lr2 = 1e-3, 1e-3
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
        loss_val_jax, gradients_jax = loss_grad(
            *params, color_gt_jax, depth_gt_jax
        )
        (dL_dmeans3D, dL_dcolors, dL_dopacity, dL_dscales, dL_drotations) = gradients_jax

        updates, state = tx.update(gradients_jax, state, params)
        params = optax.apply_updates(params, updates)

        return params, state, loss_val_jax
    if jit:
        step = jax.jit(step)

    pbar = tqdm(range(it))
    state = tx.init(params)
    for iter in pbar:
        # if iter % 10 == 0:
        #     print(f"iter {iter} | {psutil.Process().memory_info().rss / 1024 ** 2:.1f} MB")
        params, state, loss_val_jax = step(params, state)
        pbar.set_description(f"loss: {loss_val_jax}")
        losses.append(loss_val_jax.item())


    return params , losses
losses = []

import gc
gc.collect()

print("\nOptax jitted")
params, losses = inference_optax(init_params, tx, jit=True)
print(losses)
# losses = []
# print("\nOptax nonjitted")
# params, losses = inference_optax(init_params, tx, jit=False)
# print(losses)




# plot final jax
means3D, colors_precomp, opacity, scales, rotations = params
_color_final_jax, _depth_final_jax = rasterize_with_depth(
    means3D, colors_precomp, opacity, scales, rotations,
    intrinsics.width, intrinsics.height,
    intrinsics.fx, intrinsics.fy,
    intrinsics.cx, intrinsics.cy,
    intrinsics.near, intrinsics.far
)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.imshow(jnp.transpose(_color_final_jax[:3], (1,2,0)))
ax2.imshow(jnp.transpose(color_gt_jax[:3], (1,2,0)))
ax3.imshow(_depth_final_jax[0])
ax4.imshow(depth_gt_jax[0])
fig.savefig(f'jax_final_optim_{it}_{N}.png')


from IPython import embed; embed()

 