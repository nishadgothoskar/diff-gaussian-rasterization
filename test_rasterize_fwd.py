
import diff_gaussian_rasterization as dgr
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import _C as torch_backend
from jax_renderer import _build_rasterize_gaussians_fwd_primitive

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as R
import torch
import functools
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint
from tqdm import tqdm
from time import time
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

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

default_seed = 1222
gt_seed = 1201223

#############################
# Arguments
#############################
torch.manual_seed(gt_seed)
import random
random.seed(gt_seed)
np.random.seed(gt_seed)

intrinsics = Intrinsics(
    height=300,
    width=200,
    fx=300.0, fy=300.0,
    cx=100.0, cy=100.0,
    near=0.01, far=2.5
)

means3D = torch.tensor(torch.rand((100,3))-0.5 + torch.tensor([0.0, 0.0, 1.0]), requires_grad=True, device=device)
N = means3D.shape[0]; print(f"number of initial gaussians N={N}")
means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
opacity = torch.sigmoid(torch.tensor(torch.ones((N,1)), requires_grad=True, device=device))
colors_precomp = torch.tensor(torch.rand((N,3)), requires_grad=True, device=device).detach()
shs = torch.tensor(torch.rand((N,3)), requires_grad=True, device=device).detach()
scales = torch.exp(torch.tensor(-10.0 * torch.ones((N,3)), requires_grad=True, device=device))
rotations = torch.tensor(-10.0 * torch.ones((N,4)), requires_grad=True, device=device)
cov3D_precomp = torch.tensor(0.01 * torch.ones((N,3)), requires_grad=True, device=device)  # TODO not used in test currently

fovX = jnp.arctan(intrinsics.width / 2 / intrinsics.fx) * 2.0
fovY = jnp.arctan(intrinsics.height / 2 / intrinsics.fy) * 2.0
tan_fovx = math.tan(fovX)
tan_fovy = math.tan(fovY)
print(tan_fovx, tan_fovy)

camera_pose_jax = jnp.eye(4)
proj_matrix = getProjectionMatrix(0.01, 100.0, fovX, fovY).transpose(0,1).cuda()
view_matrix = torch.transpose(torch.tensor(np.array(jnp.linalg.inv(camera_pose_jax))),0,1).cuda()
projmatrix = view_matrix @ proj_matrix

##############################
# Torch
##############################
print("==========TORCH==========")
raster_settings = GaussianRasterizationSettings(
    image_height=int(intrinsics.height),
    image_width=int(intrinsics.width),
    tanfovx=tan_fovx,
    tanfovy=tan_fovy,
    bg=torch.tensor([0.0, 0.0, 0.0]).cuda(),
    scale_modifier=1.0,
    viewmatrix=view_matrix,
    projmatrix=projmatrix,
    sh_degree=0,
    campos=torch.zeros(3).cuda(),
    prefiltered=False,
    debug=None
)
rasterizer_fwd_torch = GaussianRasterizer(raster_settings=raster_settings)

torch_args = (
    raster_settings.bg, 
    means3D,
    colors_precomp,
    opacity,
    scales,
    rotations,
    raster_settings.scale_modifier,
    torch.Tensor([]), #cov3Ds_precomp (None -> torch.Tensor([])),
    raster_settings.viewmatrix,
    raster_settings.projmatrix,
    raster_settings.tanfovx,
    raster_settings.tanfovy,
    raster_settings.image_height,
    raster_settings.image_width,
    torch.Tensor([]), # sh (None -> torch.Tensor([])),
    raster_settings.sh_degree,
    raster_settings.campos,
    raster_settings.prefiltered,
    raster_settings.debug
)
start = time()
torch_outs = torch_backend.rasterize_gaussians(*torch_args)
end = time() 

print(f"Elapsed time = {end - start} s")

##############################
# Jax
##############################
print("==========JAX==========")

rasterizer_fwd_jax = _build_rasterize_gaussians_fwd_primitive()
jax_args = (torch_to_jax(means3D),
            torch_to_jax(colors_precomp),
            torch_to_jax(opacity),
            torch_to_jax(scales),
            torch_to_jax(rotations),
            torch_to_jax(cov3D_precomp),
            torch_to_jax(view_matrix),
            torch_to_jax(projmatrix),
            torch_to_jax(shs))
for iter in range(1,3):
    start = time()
    jax_outs = rasterizer_fwd_jax.bind(
                jnp.zeros(3), # bg
                *jax_args,
                jnp.zeros(3), # campos
                tanfovx=tan_fovx, 
                tanfovy=tan_fovy, 
                image_height=int(intrinsics.height), 
                image_width=int(intrinsics.width),  
                sh_degree=0
    )  
    end = time() 
    print(f"Elapsed time (iter {iter})= {end - start} s")


########################
# Compare values
########################
num_rendered_torch, color_torch, radii_torch, geomBuffer_torch, binningBuffer_torch, imgBuffer_torch = torch_outs

num_rendered_jax, color_jax, radii_jax, geomBuffer_jax, binningBuffer_jax, imgBuffer_jax = jax_outs

# (1) num_rendered
print("\n")
print(f"Torch num_rendered = {num_rendered_torch}; Jax num_rendered = {num_rendered_jax}")
print(f"Num_rendered PASS: {num_rendered_torch == num_rendered_jax[0]}") 


# (2) color, radii
print("\n")
print(f"color PASS: {jnp.isclose(torch_to_jax(color_torch), color_jax).all()}")
print(f"radii PASS: {jnp.isclose(torch_to_jax(radii_torch), radii_jax).all()}")


# (3) buffers (TODO these are clearly wrong currently)
print("\n")
print(f"geomBuffer torch: sum {geomBuffer_torch.sum().item()}, min {geomBuffer_torch.min().item()}, max {geomBuffer_torch.max().item()}")
print(f"geomBuffer jax: sum {geomBuffer_jax.sum().item()}, min {geomBuffer_jax.min().item()}, max {geomBuffer_jax.max().item()}")
print(f"geomBuffer PASS: {jnp.isclose(torch_to_jax(geomBuffer_torch), geomBuffer_jax[:geomBuffer_torch.shape[0]]).all()}")

print()
print(f"binningBuffer torch: sum {binningBuffer_torch.sum().item()}, min {binningBuffer_torch.min().item()}, max {binningBuffer_torch.max().item()}")
print(f"binningBuffer jax: sum {binningBuffer_jax.sum().item()}, min {binningBuffer_jax.min().item()}, max {binningBuffer_jax.max().item()}")
print(f"binningBuffer PASS: {jnp.isclose(torch_to_jax(binningBuffer_torch), binningBuffer_jax[:binningBuffer_torch.shape[0]]).all()}")

print()
print(f"imgBuffer torch: sum {imgBuffer_torch.sum().item()}, min {imgBuffer_torch.min().item()}, max {imgBuffer_torch.max().item()}")
print(f"imgBuffer jax: sum {imgBuffer_jax.sum().item()}, min {imgBuffer_jax.min().item()}, max {imgBuffer_jax.max().item()}")
print(f"imgBuffer PASS: {jnp.isclose(torch_to_jax(imgBuffer_torch), imgBuffer_jax[:imgBuffer_torch.shape[0]]).all()}")




from IPython import embed; embed()