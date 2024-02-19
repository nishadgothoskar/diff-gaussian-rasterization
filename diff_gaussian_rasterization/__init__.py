#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,  #n,3
            grad_means2D, # n 3
            grad_sh, # n 0 3
            grad_colors_precomp, # n 3
            grad_opacities,  # n 1
            grad_scales,  # n 3
            grad_rotations, # n 4 
            grad_cov3Ds_precomp,  # n 6
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )


## JAX ######
    
from diff_gaussian_rasterization import _C
import functools
import jax
from jax import core, dtypes
from jax.core import ShapedArray
from jaxlib.hlo_helpers import custom_call
from jax.interpreters import batching, mlir, xla
from jax.lib import xla_client
import numpy as np
import jax.numpy as jnp
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CHANNELS = 5

################################
# Helpers and boilerplates
################################

TEN_E_5 = int(1e5)
TEN_E_6 = int(1e6)
TEN_E_7 = int(1e7)
TEN_E_8 = int(1e8)
TEN_E_9 = int(1e9)

for _name, _value in _C.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

# XLA array layout in memory
def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

################################
# Rasterize fwd primitive
################################

def _build_rasterize_gaussians_fwd_primitive():
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_fwd_abstract(
            bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            viewmatrix,
            projmatrix,
            campos,
            tanfovx, tanfovy, 
            image_height, image_width,  
        ):
        float_dtype = dtypes.canonicalize_dtype(np.float32)
        int_dtype = dtypes.canonicalize_dtype(np.int32)
        byte_dtype = dtypes.canonicalize_dtype(np.uint8)

        num_gaussians, _ = means3D.shape

        GEOM_BUFFER_SIZE = 2 * TEN_E_9
        BINNING_BUFFER_SIZE = 2 * TEN_E_9
        IMG_BUFFER_SIZE = 2 * TEN_E_9

        return [ShapedArray((1,), int_dtype),
                ShapedArray((NUM_CHANNELS, image_height, image_width),  float_dtype),
                ShapedArray((num_gaussians,), int_dtype),
                ShapedArray((GEOM_BUFFER_SIZE,),  byte_dtype),
                ShapedArray((BINNING_BUFFER_SIZE,),  byte_dtype),
                ShapedArray((IMG_BUFFER_SIZE,),  byte_dtype),
        ]
    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_fwd_lowering(ctx,
            bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            viewmatrix,
            projmatrix,
            campos,
            tanfovx, tanfovy, 
            image_height, image_width,  
        ):
        float_to_ir = mlir.dtype_to_ir_type(np.dtype(np.float32))
        int_to_ir = mlir.dtype_to_ir_type(np.dtype(np.int32))
        byte_to_ir = mlir.dtype_to_ir_type(np.dtype(np.uint8))

        GEOM_BUFFER_SIZE = 2 * TEN_E_9
        BINNING_BUFFER_SIZE = 2 * TEN_E_9
        IMG_BUFFER_SIZE = 2 * TEN_E_9
        
        num_gaussians = ctx.avals_in[1].shape[0]    
        opaque = _C.build_gaussian_rasterize_descriptor(
            image_height, image_width, 0, num_gaussians, tanfovx, tanfovy, 
            GEOM_BUFFER_SIZE, BINNING_BUFFER_SIZE, IMG_BUFFER_SIZE   
        )

        op_name = "rasterize_gaussians_fwd"

        operands = [bg, means3D, colors_precomp, opacities, scales, rotations,
                      viewmatrix, projmatrix, campos]

        operands_ctx = ctx.avals_in[:len(operands)]

        output_shapes = [
            (1,), (NUM_CHANNELS, image_height, image_width), (num_gaussians,), (GEOM_BUFFER_SIZE,), (BINNING_BUFFER_SIZE,), (IMG_BUFFER_SIZE,)
        ]

        result_types = [
            mlir.ir.RankedTensorType.get(
                [1],
                int_to_ir),
            mlir.ir.RankedTensorType.get(
                [NUM_CHANNELS, image_height, image_width],
                float_to_ir),
            mlir.ir.RankedTensorType.get(
                [num_gaussians],
                int_to_ir),
            mlir.ir.RankedTensorType.get(
                [GEOM_BUFFER_SIZE],
                byte_to_ir),
            mlir.ir.RankedTensorType.get(
                [BINNING_BUFFER_SIZE],
                byte_to_ir),
            mlir.ir.RankedTensorType.get(
                [IMG_BUFFER_SIZE],
                byte_to_ir),
        ]

        return custom_call(
            op_name,
            # Output types
            result_types=result_types,
            # The inputs:
            operands=operands,
            backend_config=opaque,
            operand_layouts=default_layouts(*[i.shape for i in operands_ctx]),
            result_layouts=default_layouts(*output_shapes),
        ).results
    # *********************************************
    # *  REGISTER THE OP WITH JAX  *
    # *********************************************
    _rasterize_prim = core.Primitive(f"jax_render_primitive_fwd")
    _rasterize_prim.multiple_results = True
    _rasterize_prim.def_impl(functools.partial(xla.apply_primitive, _rasterize_prim))

    # # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_rasterize_prim, _rasterize_fwd_lowering, platform="gpu")
    _rasterize_prim.def_abstract_eval(_rasterize_fwd_abstract)

    return _rasterize_prim


################################
# Rasterize bwd primitive
################################

def _build_rasterize_gaussians_bwd_primitive():
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_bwd_abstract(
            bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            viewmatrix,
            projmatrix,
            grad_out_color,
            campos,
            geomBuffer,
            num_rendered_array,
            binningBuffer,
            imgBuffer,
            tanfovx, 
            tanfovy, 
        ):
        float_dtype = dtypes.canonicalize_dtype(np.float32)
        int_dtype = dtypes.canonicalize_dtype(np.int32)
        byte_dtype = dtypes.canonicalize_dtype(np.uint8)

        num_gaussians, _ = means3D.shape


        return [
                ShapedArray((num_gaussians, 3), float_dtype),  # dL_dmeans3D,
                ShapedArray((num_gaussians, 3), float_dtype),  # dL_dmeans2D,
                ShapedArray((num_gaussians, NUM_CHANNELS),  float_dtype), # dL_dcolors
                ShapedArray((num_gaussians, 2,2),  float_dtype), # dL_dconic
                ShapedArray((num_gaussians, 1), float_dtype),  # dL_dopacity
                ShapedArray((num_gaussians, 6), float_dtype),  # dL_dcov3D
                ShapedArray((num_gaussians, 1, 3), float_dtype),  # dL_dsh
                ShapedArray((num_gaussians, 3), float_dtype),  # dL_dscales
                ShapedArray((num_gaussians, 4), float_dtype),  # dL_drotations
        ]

    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_bwd_lowering(ctx,
            bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            viewmatrix,
            projmatrix,
            grad_out_color,
            campos,
            geomBuffer,
            num_rendered_array,
            binningBuffer,
            imgBuffer,
            tanfovx, 
            tanfovy, 
    ):

        float_to_ir = mlir.dtype_to_ir_type(np.dtype(np.float32))

        num_gaussians = ctx.avals_in[1].shape[0]  
        image_height, image_width = ctx.avals_in[8].shape[1:3]
        opaque = _C.build_gaussian_rasterize_descriptor(
            image_height, image_width, 0, num_gaussians, tanfovx, tanfovy,   
            1, 1, 1   # buffer sizes are irrelevant for bwd  
        )

        op_name = "rasterize_gaussians_bwd"

        operands = [bg, means3D, radii, colors_precomp, scales, rotations,
                    viewmatrix, projmatrix, 
                    grad_out_color, 
                    campos, 
                    geomBuffer, num_rendered_array, binningBuffer, imgBuffer]

        operands_ctx = ctx.avals_in[:len(operands)]

        M = 1

        output_shapes = [   
                (num_gaussians, 3),  # dL_dmeans3D,
                (num_gaussians, 3),  # dL_dmeans2D,
                (num_gaussians, NUM_CHANNELS), # dL_dcolors
                (num_gaussians, 2,2), # dL_dconic
                (num_gaussians, 1),  # dL_dopacity
                (num_gaussians, 6),  # dL_dcov3D
                (num_gaussians, 1, 3),  # dL_dsh
                (num_gaussians, 3),  # dL_dscales
                (num_gaussians, 4),  # dL_drotations
        ]
        result_types = [mlir.ir.RankedTensorType.get(list(shp), float_to_ir) for shp in output_shapes]

        result = custom_call(
            op_name,
            # Output types
            result_types=result_types,
            # The inputs:
            operands=operands,
            backend_config=opaque,
            operand_layouts=default_layouts(*[i.shape for i in operands_ctx]),
            result_layouts=default_layouts(*output_shapes),
        ).results

        del imgBuffer
        del binningBuffer
        del geomBuffer

        return result


    # *********************************************
    # *  REGISTER THE OP WITH JAX  *
    # *********************************************
    _rasterize_prim = core.Primitive(f"jax_render_primitive_bwd")
    _rasterize_prim.multiple_results = True
    _rasterize_prim.def_impl(functools.partial(xla.apply_primitive, _rasterize_prim))

    # # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_rasterize_prim, _rasterize_bwd_lowering, platform="gpu")
    _rasterize_prim.def_abstract_eval(_rasterize_bwd_abstract)

    return _rasterize_prim

rasterizer_fwd_primitive = _build_rasterize_gaussians_fwd_primitive()
rasterizer_bwd_primitive = _build_rasterize_gaussians_bwd_primitive()

def getProjectionMatrixJax(width, height, fx, fy, cx, cy, znear, zfar):
    fovX = jnp.arctan(width / 2 / fx) * 2.0
    fovY = jnp.arctan(height / 2 / fy) * 2.0

    tanHalfFovY = jnp.tan((fovY / 2))
    tanHalfFovX = jnp.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0
    P = jnp.transpose(jnp.array([
        [2.0 * znear / (right - left), 0.0, (right + left) / (right - left), 0.0],
        [0.0, 2.0 * znear / (top - bottom), (top + bottom) / (top - bottom), 0.0],
        [0.0, 0.0, z_sign * zfar / (zfar - znear), -(zfar * znear) / (zfar - znear)],
        [0.0, 0.0, z_sign, 0.0]
    ]))
    return P

rasterizer_fwd_primitive = _build_rasterize_gaussians_fwd_primitive()
rasterizer_bwd_primitive = _build_rasterize_gaussians_bwd_primitive()


def expand_color(means3D, color):
    # return color
    return jnp.hstack([
        color,
        means3D[:,2:3],
        jnp.ones((means3D.shape[0], 1)),
    ])

def rasterize_with_depth(
    means3D, colors_precomp, opacity, scales, rotations,
    image_width, image_height, fx,fy, cx,cy, near, far
):
    full_color = rasterize(
        means3D, expand_color(means3D, colors_precomp), opacity, scales, rotations,
        image_width, image_height, fx,fy, cx,cy, near, far
    )
    return full_color[:3, ...], full_color[3:4, ...]

@functools.partial(jax.custom_vjp, nondiff_argnums=(5,6,7,8,9,10,11,12))
def rasterize(
    means3D, colors_precomp, opacity, scales, rotations,
    image_width, image_height, fx,fy, cx,cy, near, far
):
    fovX = np.arctan(image_width / 2 / fx) * 2.0
    fovY = np.arctan(image_height / 2 / fy) * 2.0
    tan_fovx = np.tan(fovX)
    tan_fovy = np.tan(fovY)

    pmatrix = getProjectionMatrixJax(image_width, image_height, fx,fy, cx,cy,near,far)
    view_matrix = jnp.transpose(jnp.linalg.inv(jnp.eye(4)))

    projmatrix = view_matrix @ pmatrix

    bg = jnp.zeros(NUM_CHANNELS)
    campos = jnp.zeros(3)
    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = rasterizer_fwd_primitive.bind(
                bg,
                means3D,
                colors_precomp,
                opacity,
                scales,
                rotations,
                view_matrix,
                projmatrix,
                campos,
                tanfovx=tan_fovx, 
                tanfovy=tan_fovy, 
                image_height=image_height, 
                image_width=image_width,  
    )
    return color

def rasterize_fwd(
    means3D, colors_precomp, opacity, scales, rotations,
    image_width, image_height, fx,fy, cx,cy, near, far
):
    fovX = np.arctan(image_width / 2 / fx) * 2.0
    fovY = np.arctan(image_height / 2 / fy) * 2.0
    tan_fovx = np.tan(fovX)
    tan_fovy = np.tan(fovY)

    pmatrix = getProjectionMatrixJax(image_width, image_height, fx,fy, cx,cy,near,far)
    view_matrix = jnp.transpose(jnp.linalg.inv(jnp.eye(4)))

    projmatrix = view_matrix @ pmatrix

    bg = jnp.zeros(NUM_CHANNELS)
    campos = jnp.zeros(3)
    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = rasterizer_fwd_primitive.bind(
                bg,
                means3D,
                colors_precomp,
                opacity,
                scales,
                rotations,
                view_matrix,
                projmatrix,
                campos,
                tanfovx=tan_fovx, 
                tanfovy=tan_fovy, 
                image_height=image_height, 
                image_width=image_width,  
    )
    return color, (
        means3D, colors_precomp, opacity, scales, rotations,
        # image_width, image_height, fx,fy, cx,cy,near,far,
        bg,campos,
        num_rendered,
        color,
        radii,
        geomBuffer,
        binningBuffer,
        imgBuffer,
        view_matrix,
        projmatrix
    )

rasterize_fwd_jit = jax.jit(
    rasterize_fwd,
    static_argnums=(5,6,7,8,9,10,11,12)
)
def rasterize_bwd(image_width, image_height, fx,fy, cx,cy, near, far, res, gradients):
    (
        means3D, colors_precomp, opacity, scales, rotations,
        # image_width, image_height, fx,fy, cx,cy,near,far,
        bg,campos,
        num_rendered,
        color,
        radii,
        geomBuffer,
        binningBuffer,
        imgBuffer,
        view_matrix,
        projmatrix
    ) = res
    fovX = np.arctan(image_width / 2 / fx) * 2.0
    fovY = np.arctan(image_height / 2 / fy) * 2.0
    tan_fovx = np.tan(fovX)
    tan_fovy = np.tan(fovY)
    

    (dL_dmeans3D,
    dL_dmeans2D,
    dL_dcolors,
    dL_dconic,
    dL_dopacity,
    dL_dcov3D,
    dL_dsh,
    dL_dscales,
    dL_drotations) = rasterizer_bwd_primitive.bind(
                bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                view_matrix,
                projmatrix,
                gradients,
                campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                tanfovx=tan_fovx, 
                tanfovy=tan_fovy,
    )

    return (
        dL_dmeans3D,
        dL_dcolors,
        dL_dopacity,
        dL_dscales,
        dL_drotations,
    )
rasterize_bwd_jit = jax.jit(
    rasterize_bwd,
    static_argnums=(0,1,2,3,4,5,6,7,)
)

rasterize.defvjp(rasterize_fwd, rasterize_bwd)
rasterize_jit = jax.jit(rasterize, static_argnums=(5,6,7,8,9,10,11,12))