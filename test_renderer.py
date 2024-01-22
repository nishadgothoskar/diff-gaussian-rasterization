
import diff_gaussian_rasterization as dgr
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import _C as torch_backend
from jax_gaussian_renderer import rasterize

import jax
import jax.numpy as jnp
import torch
import math
import numpy as np
import random
import time 

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
    return arr.min(), arr.max(), arr.sum()

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

def torch_to_jax(torch_array):
    return jnp.array(torch_array.detach().cpu().numpy())

default_seed = 0
gt_seed = 1

#############################
# Arguments
#############################
reset(default_seed)

intrinsics = Intrinsics(
    height=200,
    width=300,
    fx=300.0, fy=300.0,
    cx=100.0, cy=100.0,
    near=0.01, far=2.5
)

fovX = jnp.arctan(intrinsics.width / 2 / intrinsics.fx) * 2.0
fovY = jnp.arctan(intrinsics.height / 2 / intrinsics.fy) * 2.0
tan_fovx = math.tan(fovX)
tan_fovy = math.tan(fovY)

means3D = jax.random.uniform(jax.random.PRNGKey(default_seed), shape=(100, 3), minval=-0.5, maxval=0.5) + jnp.array([0.0, 0.0, 1.0])
N = means3D.shape[0]
opacity = jnp.ones(shape=(N,1))
scales =jnp.ones((N,3)) * 4.5400e-03
rotations = jax.random.uniform(jax.random.PRNGKey(default_seed), shape=(N,4), minval=-1.0, maxval=1.0)
colors_precomp = jax.random.uniform(jax.random.PRNGKey(default_seed), shape=(N,3), minval=0.0, maxval=1.0)
cov3D_precomp = jax.random.uniform(jax.random.PRNGKey(default_seed), shape=(N,3), minval=0.0, maxval=0.1)
# sh = jax.random.uniform(jax.random.PRNGKey(default_seed), shape=(N,0), minval=0.0, maxval=1.0)
sh_jax = jnp.array([])

camera_pose_jax = jnp.eye(4)
proj_matrix = getProjectionMatrix(0.01, 100.0, fovX, fovY).transpose(0,1).cuda()
view_matrix = torch.transpose(torch.tensor(np.array(jnp.linalg.inv(camera_pose_jax))),0,1).cuda()
projmatrix = view_matrix @ proj_matrix

view_matrix = jnp.array(view_matrix.cpu().numpy())
projmatrix = jnp.array(projmatrix.cpu().numpy())



raster_settings_jax = jax.tree_util.Partial(lambda: GaussianRasterizationSettings(
    image_height=int(intrinsics.height),
    image_width=int(intrinsics.width),
    tanfovx=tan_fovx,
    tanfovy=tan_fovy,
    bg=jnp.array([0.0, 0.0, 0.0]),
    scale_modifier=1.0,
    viewmatrix=view_matrix,
    projmatrix=projmatrix,
    sh_degree=0,
    campos=jnp.zeros(3),
    prefiltered=False,
    debug=None
)) # jtu.Partial trick to create a valid nondiff_argnums and avoid Tracer errors

##########################################################################################
# FWD TEST
##########################################################################################
## Args

jax_fwd_args = (
            means3D,
            colors_precomp,
            opacity,
            scales,
            rotations,
            cov3D_precomp,
            sh_jax,
            raster_settings_jax
            )


## Run JAX fwd
vjp_rasterize_fwd_jit = jax.jit(jax.tree_util.Partial(jax.vjp, rasterize)) 
for _ in range(2):
    start = time.time()
    (color_jax, radii_jax), rasterize_vjp = vjp_rasterize_fwd_jit(*jax_fwd_args)
    end = time.time()
print(f"JAX FWD TIME={end-start} s")
print(test(color_jax), test(radii_jax))

##########################################################################################
# BWD TEST
##########################################################################################
## Args
grad_out_color_jax = jnp.array(color_jax)
grad_out_radii_jax = jnp.zeros_like(radii_jax)  # dummy variable

## Run JAX bwd
for _ in range(2):
    start = time.time()
    _grad_out_jax = rasterize_vjp((grad_out_color_jax, grad_out_radii_jax))
    end = time.time() 
print(f"JAX BWD TIME={end-start} s")  # jitting rasterize_vjp itself fails, but in a standard use case, that function would reside within a larger jitted workflow
grad_means3D_jax, grad_colors_precomp_jax, grad_opacities_jax, grad_scales_jax, grad_rotations_jax, grad_cov3Ds_precomp_jax, grad_sh_jax, _ = _grad_out_jax

print(f"grad_means3d = {test(grad_means3D_jax)}")
print(f"grad_colors_precomp_jax = {test(grad_colors_precomp_jax)}")
print(f"grad_opacities_jax = {test(grad_opacities_jax)}")
print(f"grad_scales_jax = {test(grad_scales_jax)}")
print(f"grad_rotations_jax = {test(grad_rotations_jax)}")
print(f"grad_cov3Ds_precomp_jax = {test(grad_cov3Ds_precomp_jax)}")


######################
# Function things
######################
from jax import grad, jit, vmap


def _loss(raster_settings_jax, color_gt,means3D,
            colors_precomp,
            opacity,
            scales,
            rotations,
            cov3D_precomp,
            sh_jax,
            ):
    error = (color_gt - rasterize(means3D,
            colors_precomp,
            opacity,
            scales,
            rotations,
            cov3D_precomp,
            sh_jax,
            raster_settings_jax)[0][0])
    return jnp.mean(error**2)
    
loss_partial = jax.tree_util.Partial(_loss, raster_settings_jax)
loss = jax.jit(loss_partial)
grad_func = jax.grad(loss, argnums=(0,1,2,3,4,5,6,))
out = grad_func(grad_out_color_jax,             
          means3D,
            colors_precomp,
            opacity,
            scales,
            rotations,
            cov3D_precomp,
            sh_jax,
        )
grad_means3D_jax, grad_colors_precomp_jax, grad_opacities_jax, grad_scales_jax, grad_rotations_jax, grad_cov3Ds_precomp_jax, grad_sh_jax = out

for o in out:
    print(test(o))

from IPython import embed; embed()