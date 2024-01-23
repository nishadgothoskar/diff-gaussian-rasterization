
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

default_seed = 0
gt_seed = 1



def render_jax_with_param_transform(means3D, colors_precomp, opacity, scales, rotations,
           image_width, image_height, fx,fy, cx,cy,near,far, preproc=False):
    if not preproc:
        color = rasterize(
            means3D, 
            colors_precomp, 
            opacity, 
            scales, 
            rotations,
            image_width, image_height, fx,fy, cx,cy,near,far
        ) 
    else: 
        color = rasterize(
            means3D, 
            colors_precomp, 
            jnp.exp(jax.nn.log_sigmoid(opacity)), 
            jnp.exp(scales), 
            rotations,
            image_width, image_height, fx,fy, cx,cy,near,far
        ) 
    return color


def render_torch_with_param_transform(means3D, means2D, colors_precomp, opacity, scales, rotations,
           t_rasterizer, preproc=False):
    if not preproc: 
        color,_ = t_rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations
        )
    else: 
        color,_ = t_rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = torch.sigmoid(opacity),
            scales = torch.exp(scales),
            rotations = rotations
        )
    return color

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

N = 100
means3D = jax.random.uniform(jax.random.PRNGKey(default_seed), shape=(N, 3), minval=-0.5, maxval=0.5) + jnp.array([0.0, 0.0, 1.0])
opacity = jnp.ones(shape=(N,1)) * 1.0; opacity = jnp.exp(jax.nn.log_sigmoid(opacity))
scales = jnp.ones((N,3)) * -4.5; scales = jnp.exp(scales)  # PREPROC 
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


torch_raster_settings = GaussianRasterizationSettings(
    image_height=int(intrinsics.height),
    image_width=int(intrinsics.width),
    tanfovx=tan_fovx,
    tanfovy=tan_fovy,
    bg=torch.tensor([0.0, 0.0, 0.0]).cuda(),
    scale_modifier=1.0,
    viewmatrix=view_matrix_torch,
    projmatrix=projmatrix_torch,
    sh_degree=0,
    campos=torch.zeros(3).cuda(),
    prefiltered=False,
    debug=False
)
torch_rasterizer = GaussianRasterizer(raster_settings=torch_raster_settings)


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
    intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, 
    intrinsics.cx, intrinsics.cy,
    intrinsics.near, intrinsics.far
)
color_gt_torch = jax_to_torch(color_gt_jax)
# _, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(jnp.transpose(color_gt_jax[:3], (1,2,0)))
# ax1.imshow(jnp.transpose(color_gt_jax[:3], (1,2,0)))
plt.imsave("gt.png", jnp.transpose(color_gt_jax[:3], (1,2,0)))

##########################################################################################
# FWD TEST
##########################################################################################
## Args

## Run JAX fwd
# vjp_rasterize_fwd_jit = jax.jit(jax.tree_util.Partial(jax.vjp, rasterize)) 
for _ in range(2):
    start = time.time()
    color_jax = render_jax_with_param_transform(
        means3D, colors_precomp, opacity, scales, rotations,
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, 
        intrinsics.cx, intrinsics.cy,
        intrinsics.near, intrinsics.far
    )
    end = time.time()
print(f"JAX FWD TIME={end-start} s")
plt.imsave("jax_fwd_0.png", jnp.transpose(color_jax[:3], (1,2,0)))

torch_means3d = jax_to_torch(means3D) # jank
torch_means2d = torch.zeros_like(torch_means3d, dtype=torch_means3d.dtype, requires_grad=True, device="cuda") + 0
torch_colors_precomp = jax_to_torch(colors_precomp)
torch_opacity = jax_to_torch(opacity)
torch_scales = jax_to_torch(scales)
torch_rotations = jax_to_torch(rotations)
for _ in range(2):
    start = time.time()
    color_torch = render_torch_with_param_transform(torch_means3d, torch_means2d, torch_colors_precomp, torch_opacity, torch_scales, torch_rotations, torch_rasterizer)
    end = time.time()
print(f"TORCH FWD TIME={end-start} s")
plt.imsave("torch_fwd_0.png", np.array(torch.permute(color_torch[:3].detach().cpu(), (1,2,0))))

print("JAX min/max/sum: ", test(color_jax))
print("Torch min/max/sum: ",test(color_torch))
assert compare(color_torch[:3], color_jax[:3]), f"FWD color; max diff {max_err(color_torch[:3], color_jax[:3])}"

# ##########################################################################################
# # BWD TEST
# ##########################################################################################

def loss(means3D, colors_precomp, opacity, scales, rotations,
    image_width, image_height, fx,fy, cx,cy,near,far, color_gt):
    color = render_jax_with_param_transform(
        means3D, colors_precomp, opacity, scales, rotations,
        image_width, image_height, fx,fy, cx,cy,near,far
    )
    return jnp.sum(0.5 * (color[:3] - color_gt[:3])**2)
loss_grad = jax.jit(jax.value_and_grad(loss, argnums=(0,1,2,3,4,)), static_argnums=(5,6,7,8,9,10,11,12,))


######## Jax optim ##########
print("\n------------\n Jax Optim \n----------\n")

import optax
it = 100

params = (means3D, colors_precomp, opacity, scales, rotations)
param_labels = ("means3D", "colors_precomp", "opacity", "scales", "rotations")
tx = optax.multi_transform(
    {
        'means3D': optax.adam(0.001),
        'colors_precomp': optax.adam(0.001),
        'opacity': optax.adam(0.0001),
        'scales': optax.adam(0.0001),
        'rotations': optax.adam(0.0001),
    },
    param_labels
)
state = tx.init(params)

pbar = tqdm(range(it))
all_jax_losses = []
for _ in pbar:
    # gradients: dL_dmeans3D,
        # dL_dcolors,
        # dL_dopacity,
        # dL_dscales,
        # dL_drotations
    loss_val_jax, gradients_jax = loss_grad(
        *params,
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, 
        intrinsics.cx, intrinsics.cy,
        intrinsics.near, intrinsics.far, color_gt_jax
    )
    (dL_dmeans3D, dL_dcolors, dL_dopacity, dL_dscales, dL_drotations) = gradients_jax
    pbar.set_description(f"loss: {loss_val_jax:.4f}")

    updates, state = tx.update(gradients_jax, state, params)
    params = optax.apply_updates(params, updates)

    all_jax_losses.append(loss_val_jax.item())
print(all_jax_losses)
print([test(u) for u in gradients_jax])

# plot final jax
_color_final_jax = render_jax_with_param_transform(
        *params,
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, 
        intrinsics.cx, intrinsics.cy,
        intrinsics.near, intrinsics.far
    )
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(jnp.transpose(_color_final_jax[:3], (1,2,0)))
ax2.imshow(jnp.transpose(color_gt_jax[:3], (1,2,0)))
fig.savefig(f'jax_final_optim_{it}.png')


######## Torch optim ###########
print("\n------------\n Torch optim \n----------\n")
pbar = tqdm(range(it))

torch_optimizer = torch.optim.Adam([
    {'params': [torch_means3d], 'lr': 0.001, "name": "means3D"},
    {'params': [torch_colors_precomp], 'lr': 0.001, "name": "colors_precomp"},
    {'params': [torch_opacity], 'lr': 0.0001, "name": "opacity"},
    {'params': [torch_scales], 'lr': 0.0001, "name": "scales"},
    {'params': [torch_rotations], 'lr': 0.0001, "name": "rotations"},
])

all_torch_losses = []
for _ in pbar:
    torch_optimizer.zero_grad()

    color_torch = render_torch_with_param_transform(torch_means3d, 
                                                    torch_means2d, 
                                                    torch_colors_precomp, 
                                                    torch_opacity, torch_scales, 
                                                    torch_rotations, torch_rasterizer)

    loss_val_torch = torch.sum(0.5 * (color_torch[:3] - color_gt_torch[:3])**2) 
    loss_val_torch.backward()

    torch_optimizer.step()
    pbar.set_description(f"{loss_val_torch.item()}")
    all_torch_losses.append(loss_val_torch.item())

print(all_torch_losses)
print([test(u) for u in (torch_means3d.grad, torch_colors_precomp.grad, torch_opacity.grad, torch_scales.grad, torch_rotations.grad)])

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(jnp.transpose(color_torch.detach().cpu().numpy()[:3], (1,2,0)))
ax2.imshow(jnp.transpose(color_gt_jax[:3], (1,2,0)))
fig.savefig(f'torch_final_optim_{it}.png')

# try:
#     assert compare(torch_means3d.grad, gradients_jax[0], atol=0.1), f"Means3d: Max error {max_err(torch_means3d.grad, gradients_jax[0])}"
#     assert compare(torch_colors_precomp.grad, gradients_jax[1]), f"Colors: Max error {max_err(torch_colors_precomp.grad, gradients_jax[1])}"
#     assert compare(torch_opacity.grad, gradients_jax[2]), f"Opacity: Max error {max_err(torch_opacity.grad, gradients_jax[2])}"
#     assert compare(torch_scales.grad, gradients_jax[3]), f"Scales: Max error {max_err(torch_scales.grad, gradients_jax[3])}"
#     assert compare(torch_rotations.grad, gradients_jax[4]), f"Rotations: Max error {max_err(torch_rotations.grad, gradients_jax[4])}"
# except AssertionError:
#     print("NOT GOOD")
#     print(f"Means3d: Max error {max_err(torch_means3d.grad, gradients_jax[0])}, torch min/max/sum {test(torch_means3d.grad)}, jax min/max/sum{test(gradients_jax[0])}")
#     print(f"Colors: Max error {max_err(torch_colors_precomp.grad, gradients_jax[1])}, torch min/max/sum {test(torch_colors_precomp.grad)}, jax min/max/sum{test(gradients_jax[1])}")
    
#     print(f"Opacity: Max error {max_err(torch_opacity.grad, gradients_jax[2])}, torch min/max/sum {test(torch_opacity.grad)}, jax min/max/sum{test(gradients_jax[2])}")
#     print(f"Scales: Max error {max_err(torch_scales.grad, gradients_jax[3])}, torch min/max/sum {test(torch_scales.grad)}, jax min/max/sum{test(gradients_jax[3])}")
#     print(f"Rotations: Max error {max_err(torch_rotations.grad, gradients_jax[4])}, torch min/max/sum {test(torch_rotations.grad)}, jax min/max/sum{test(gradients_jax[4])}")
#     from IPython import embed; embed()

from IPython import embed; embed()