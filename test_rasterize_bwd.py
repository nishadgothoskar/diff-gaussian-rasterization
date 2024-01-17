
import diff_gaussian_rasterization as dgr
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import _C as torch_backend
from jax_renderer import _build_rasterize_gaussians_fwd_primitive, _build_rasterize_gaussians_bwd_primitive

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as R
import torch
import functools
import matplotlib.pyplot as plt
import math
import numpy as np
import random
from random import randint
from tqdm import tqdm
from time import time
 
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

def jax_to_torch(jnp_array):
    return torch.tensor(np.array(jnp_array), requires_grad=True, device=device)

rasterizer_fwd_jax = _build_rasterize_gaussians_fwd_primitive()
rasterizer_bwd_jax = _build_rasterize_gaussians_bwd_primitive()

default_seed = 1222
gt_seed = 1201223

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


raster_settings = GaussianRasterizationSettings(
    image_height=int(intrinsics.height),
    image_width=int(intrinsics.width),
    tanfovx=tan_fovx,
    tanfovy=tan_fovy,
    bg=torch.tensor([0.0, 0.0, 0.0]).cuda(),
    scale_modifier=1.0,
    viewmatrix=jax_to_torch(view_matrix),
    projmatrix=jax_to_torch(projmatrix),
    sh_degree=0,
    campos=torch.zeros(3).cuda(),
    prefiltered=False,
    debug=None
)
# rasterizer_fwd_torch = GaussianRasterizer(raster_settings=raster_settings)

##########################################################################################
# FWD TEST
##########################################################################################
## Args
jax_fwd_args = (means3D,
            colors_precomp,
            opacity,
            scales,
            rotations,
            cov3D_precomp,
            view_matrix,
            projmatrix,
            sh_jax)

torch_fwd_args = (
    raster_settings.bg, 
    jax_to_torch(means3D),
    jax_to_torch(colors_precomp),
    jax_to_torch(opacity),
    jax_to_torch(scales),
    jax_to_torch(rotations),
    raster_settings.scale_modifier,
    jax_to_torch(cov3D_precomp), # (None -> torch.Tensor([])),
    raster_settings.viewmatrix,
    raster_settings.projmatrix,
    raster_settings.tanfovx,
    raster_settings.tanfovy,
    raster_settings.image_height,
    raster_settings.image_width,
    jax_to_torch(sh_jax), # (None -> torch.Tensor([])),
    raster_settings.sh_degree,
    raster_settings.campos,
    raster_settings.prefiltered,
    raster_settings.debug
)

## Run 
num_rendered_jax, color_jax, radii_jax, geomBuffer_jax, binningBuffer_jax, imgBuffer_jax = rasterizer_fwd_jax.bind(
            jnp.zeros(3), # bg
            *jax_fwd_args,
            jnp.zeros(3), # campos
            tanfovx=tan_fovx, 
            tanfovy=tan_fovy, 
            image_height=int(intrinsics.height), 
            image_width=int(intrinsics.width),  
            sh_degree=0
)

num_rendered_torch, color_torch, radii_torch, geomBuffer_torch, binningBuffer_torch, imgBuffer_torch = torch_backend.rasterize_gaussians(*torch_fwd_args)
color_torch = color_torch.detach()
color_torch_jax = torch_to_jax(color_torch)


## Compare
assert num_rendered_torch == int(num_rendered_jax[0]), "num_rendered mismatch"
assert jnp.allclose(color_torch_jax, color_jax), "color mismatch"
assert jnp.allclose(torch_to_jax(radii_torch), radii_jax), "radii mismatch"
print("FORWARD PASSED: NUM_RENDER, COLOR, RADII")

##########################################################################################
# BWD TEST
##########################################################################################

## Args
grad_out_color_jax = jnp.array(color_jax)
jax_bwd_args = (
    jnp.zeros(3),
    means3D, #1
    radii_jax, #2 
    colors_precomp, #3 
    scales, #4
    rotations, #5 
    cov3D_precomp, #6 
    view_matrix, #7 
    projmatrix, #8
    grad_out_color_jax, #9
    sh_jax, #10
    raster_settings.sh_degree, #11
    geomBuffer_jax, #12
    num_rendered_jax, #13 
    binningBuffer_jax, #14
    imgBuffer_jax #15
)

grad_out_color_torch= torch.Tensor(np.array(color_torch_jax)).cuda()
torch_bwd_args = (
    raster_settings.bg,
    jax_to_torch(means3D), 
    radii_torch, 
    jax_to_torch(colors_precomp), 
    jax_to_torch(scales), 
    jax_to_torch(rotations), 
    raster_settings.scale_modifier, 
    jax_to_torch(cov3D_precomp), 
    raster_settings.viewmatrix, 
    raster_settings.projmatrix, 
    raster_settings.tanfovx, 
    raster_settings.tanfovy, 
    grad_out_color_torch, 
    jax_to_torch(sh_jax), 
    raster_settings.sh_degree, 
    raster_settings.campos,
    geomBuffer_torch,
    num_rendered_torch,
    binningBuffer_torch,
    imgBuffer_torch,
    raster_settings.debug
)

## Run 
torch.cuda.empty_cache()
reset(gt_seed)
(grad_means2D_jax,
 grad_colors_precomp_jax,
 grad_opacities_jax,
 grad_means3D_jax,
 grad_cov3Ds_precomp_jax,
 grad_sh_jax,
 grad_scales_jax, grad_rotations_jax, _) = rasterizer_bwd_jax.bind(
            *jax_bwd_args,
            tanfovx=tan_fovx, 
            tanfovy=tan_fovy, 
            sh_degree=0
)  
torch.cuda.empty_cache()
reset(gt_seed)
(grad_means2D_torch_1,
 grad_colors_precomp_torch_1,
 grad_opacities_torch_1,
 grad_means3D_torch_1,
 grad_cov3Ds_precomp_torch_1,
 grad_sh_torch_1,
 grad_scales_torch_1, grad_rotations_torch_1) = torch_backend.rasterize_gaussians_backward(*torch_bwd_args)

torch.cuda.empty_cache()
reset(gt_seed)
(grad_means2D_torch_2,
 grad_colors_precomp_torch_2,
 grad_opacities_torch_2,
 grad_means3D_torch_2,
 grad_cov3Ds_precomp_torch_2,
 grad_sh_torch_2,
 grad_scales_torch_2, grad_rotations_torch_2) = torch_backend.rasterize_gaussians_backward(*torch_bwd_args)


## Compare
assert (jnp.allclose(torch_to_jax(grad_means2D_torch_1), grad_means2D_jax) or not torch.allclose(grad_means2D_torch_1, grad_means2D_torch_2, atol=1e-6)), f"grad_means2D mismatch, max {abs(torch_to_jax(grad_means2D_torch_1) - grad_means2D_jax).max()}"
assert jnp.allclose(torch_to_jax(grad_colors_precomp_torch_1), grad_colors_precomp_jax) or not torch.allclose(grad_colors_precomp_torch_1, grad_colors_precomp_torch_2, atol=1e-6), "grad_colors_precomp mismatch"
assert jnp.allclose(torch_to_jax(grad_opacities_torch_1), grad_opacities_jax) or not torch.allclose(grad_opacities_torch_1, grad_opacities_torch_2, atol=1e-6), "grad_opacities mismatch"
assert jnp.allclose(torch_to_jax(grad_means3D_torch_1), grad_means3D_jax) or not torch.allclose(grad_means3D_torch_1, grad_means3D_torch_2, atol=1e-6), f"grad_means3D mismatch, max {abs(torch_to_jax(grad_means3D_torch_1) - grad_means3D_jax).max()}"
assert jnp.allclose(torch_to_jax(grad_cov3Ds_precomp_torch_1), grad_cov3Ds_precomp_jax) or not torch.allclose(grad_cov3Ds_precomp_torch_1, grad_cov3Ds_precomp_torch_2, atol=1e-6), "grad_cov3Ds_precomp mismatch"
assert jnp.allclose(torch_to_jax(grad_sh_torch_1), grad_sh_jax) or not torch.allclose(grad_sh_torch_1, grad_sh_torch_2, atol=1e-6), "grad_sh mismatch"
assert jnp.allclose(torch_to_jax(grad_scales_torch_1), grad_scales_jax) or not torch.allclose(grad_scales_torch_1, grad_scales_torch_2, atol=1e-6), f"grad_scales mismatch, max {abs(torch_to_jax(grad_scales_torch_1) - grad_scales_jax).max()}"
assert jnp.allclose(torch_to_jax(grad_rotations_torch_1), grad_rotations_jax) or not torch.allclose(grad_rotations_torch_1, grad_rotations_torch_2, atol=1e-6), "grad_scales mismatch"
print("BACKWARD PASSED: all grads")