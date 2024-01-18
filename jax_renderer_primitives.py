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

################################
# Helpers and boilerplates
################################

TEN_E_5 = int(1e5)
TEN_E_6 = int(1e6)
TEN_E_7 = int(1e7)

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

        GEOM_BUFFER_SIZE = TEN_E_6
        BINNING_BUFFER_SIZE = TEN_E_6
        IMG_BUFFER_SIZE = TEN_E_7

        return [ShapedArray((1,), int_dtype),
                ShapedArray((3, image_height, image_width),  float_dtype),
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

        GEOM_BUFFER_SIZE = TEN_E_6
        BINNING_BUFFER_SIZE = TEN_E_6
        IMG_BUFFER_SIZE = TEN_E_7

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
            (1,), (3, image_height, image_width), (num_gaussians,), (GEOM_BUFFER_SIZE,), (BINNING_BUFFER_SIZE,), (IMG_BUFFER_SIZE,)
        ]

        result_types = [
            mlir.ir.RankedTensorType.get(
                [1],
                int_to_ir),
            mlir.ir.RankedTensorType.get(
                [3, image_height, image_width],
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
                ShapedArray((num_gaussians, 3),  float_dtype), # dL_dcolors
                ShapedArray((num_gaussians, 2,2),  float_dtype), # dL_dconic
                ShapedArray((num_gaussians, 1), float_dtype),  # dL_dopacity
                ShapedArray((num_gaussians, 2,2),  float_dtype), # dL_dconic
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
        image_height, image_width = ctx.avals_in[9].shape[1:3]

        opaque = _C.build_gaussian_rasterize_descriptor(
            image_height, image_width, 0, num_gaussians, tanfovx, tanfovy,   
            -1, -1, -1   # buffer sizes are irrelevant for bwd  
        )

        op_name = "rasterize_gaussians_bwd"

        operands = [bg, means3D, radii, colors_precomp, scales, rotations,
                    viewmatrix, projmatrix, 
                    grad_out_color, 
                    campos, 
                    geomBuffer, num_rendered_array, binningBuffer, imgBuffer]

        operands_ctx = ctx.avals_in[:len(operands)]

        M = ctx.avals_in[10].shape[0]  # sh.shape[0]
        if M != 0:
            M = ctx.avals_in[10].shape[1]

        output_shapes = [
                (num_gaussians, 3),  # dL_dmeans2D
                (num_gaussians, 3),  # dL_dmeans3D
                (num_gaussians, 3), # dL_dcolors
                (num_gaussians, 1),  # dL_dopacity
                (num_gaussians, 3),  # dL_dscales
                (num_gaussians, 4),# dL_drotations
                ]

        result_types = [mlir.ir.RankedTensorType.get(list(shp), float_to_ir) for shp in output_shapes]

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
    _rasterize_prim = core.Primitive(f"jax_render_primitive_bwd")
    _rasterize_prim.multiple_results = True
    _rasterize_prim.def_impl(functools.partial(xla.apply_primitive, _rasterize_prim))

    # # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_rasterize_prim, _rasterize_bwd_lowering, platform="gpu")
    _rasterize_prim.def_abstract_eval(_rasterize_bwd_abstract)

    return _rasterize_prim



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

def rasterize_fwd(
    means3D, colors_precomp, opacity, scales, rotations,
    camera_pose,
    image_width, image_height, fx,fy, cx,cy,near,far
):
    fovX = jnp.arctan(image_width / 2 / fx) * 2.0
    fovY = jnp.arctan(image_height / 2 / fy) * 2.0
    tan_fovx = math.tan(fovX)
    tan_fovy = math.tan(fovY)

    pmatrix = getProjectionMatrixJax(image_width, image_height, fx,fy, cx,cy,near,far)
    view_matrix = jnp.transpose(jnp.linalg.inv(camera_pose))

    camera_pose_jax = jnp.eye(4)
    view_matrix = jnp.transpose(jnp.linalg.inv(camera_pose_jax))

    cov3D_precomp = jnp.zeros((means3D.shape[0], 3))
    sh = jnp.zeros((means3D.shape[0], 3))

    projmatrix = view_matrix @ pmatrix
    (
        num_rendered_jax,
        color_jax,
        radii_jax,
        geomBuffer_jax,
        binningBuffer_jax,
        imgBuffer_jax
    ) = rasterizer_fwd_primitive.bind(
                jnp.zeros(3), # bg
                means3D,
                colors_precomp,
                opacity,
                scales,
                rotations,
                cov3D_precomp,
                view_matrix,
                projmatrix,
                sh,
                jnp.zeros(3), # campos
                tanfovx=tan_fovx, 
                tanfovy=tan_fovy, 
                image_height=image_height, 
                image_width=image_width,  
                sh_degree=0
    )
    return color_jax, (
        means3D, colors_precomp, opacity, scales, rotations,
        image_width, image_height, fx,fy, cx,cy,near,far,
        num_rendered_jax,
        color_jax,
        radii_jax,
        geomBuffer_jax,
        binningBuffer_jax,
        imgBuffer_jax,
        view_matrix,
        projmatrix
    )

# def rasterize_bwd(res, gradients):
#     (
#         means3D, colors_precomp, opacity, scales, rotations,
#         image_width, image_height, fx,fy, cx,cy,near,far,
#         num_rendered_jax,
#         color_jax,
#         radii_jax,
#         geomBuffer_jax,
#         binningBuffer_jax,
#         imgBuffer_jax,
#         view_matrix,
#         projmatrix
#     ) = res
#     fovX = jnp.arctan(image_width / 2 / fx) * 2.0
#     fovY = jnp.arctan(image_height / 2 / fy) * 2.0
#     tan_fovx = math.tan(fovX)
#     tan_fovy = math.tan(fovY)
#     jax_bwd_args = (
#         jnp.zeros(3),
#         means3D, #1
#         radii_jax, #2 
#         colors_precomp, #3 
#         scales, #4
#         rotations, #5 
#         # raster_settings.scale_modifier), 
#         jnp.array([]), #6 
#         view_matrix, #7 
#         projmatrix, #8
#         gradients, #9
#         jnp.array([]), #10
#         jnp.zeros(3), #11
#         geomBuffer_jax, #12
#         num_rendered_jax,
#         binningBuffer_jax, #14
#         imgBuffer_jax #15
#     )
#     # print(tan_fovx, tan_fovy)
#     # return jax_bwd_args, tan_fovx, tan_fovy

#     (grad_means2D_jax,
#     grad_colors_precomp_jax,
#     grad_opacities_jax,
#     grad_means3D_jax,
#     grad_cov3Ds_precomp_jax,
#     grad_sh_jax,
#     grad_scales_jax, grad_rotations_jax, grad_conic) = rasterizer_bwd_primitive.bind(
#                 *jax_bwd_args,
#                 tanfovx=tan_fovx, 
#                 tanfovy=tan_fovy, 
#                 sh_degree=0
#     )
#     return (
#         grad_means3D_jax,
#         grad_colors_precomp_jax,
#         grad_opacities_jax,
#         grad_scales_jax,
#         grad_rotations_jax,
#         None, None, None,
#         None, None, None,
#         None, None, None,
#     )

def rasterize_bwd(res, gradients):
    (
        means3D, colors_precomp, opacity, scales, rotations,
        image_width, image_height, fx,fy, cx,cy,near,far,
        num_rendered_jax,
        color_jax,
        radii_jax,
        geomBuffer_jax,
        binningBuffer_jax,
        imgBuffer_jax,
        view_matrix,
        projmatrix
    ) = res
    fovX = jnp.arctan(image_width / 2 / fx) * 2.0
    fovY = jnp.arctan(image_height / 2 / fy) * 2.0
    tan_fovx = math.tan(fovX)
    tan_fovy = math.tan(fovY)

    cov3D_precomp = jnp.zeros((means3D.shape[0], 3))
    sh = jnp.zeros((means3D.shape[0], 3))

    jax_bwd_args = (
        jnp.zeros(3),
        means3D, #1
        radii_jax, #2 
        colors_precomp, #3 
        scales, #4
        rotations, #5 
        # raster_settings.scale_modifier), 
        cov3D_precomp, #6 
        view_matrix, #7 
        projmatrix, #8
        gradients, #9
        sh, #10
        jnp.zeros(3), #11
        geomBuffer_jax, #12
        num_rendered_jax,
        binningBuffer_jax, #14
        imgBuffer_jax #15
    )
    # print(tan_fovx, tan_fovy)
    # return jax_bwd_args, tan_fovx, tan_fovy

    (grad_means2D_jax,
    grad_colors_precomp_jax,
    grad_opacities_jax,
    grad_means3D_jax,
    grad_cov3Ds_precomp_jax,
    grad_sh_jax,
    grad_scales_jax, grad_rotations_jax, grad_conic) = rasterizer_bwd_primitive.bind(
                *jax_bwd_args,
                tanfovx=tan_fovx, 
                tanfovy=tan_fovy, 
                sh_degree=0
    )

    return (grad_means2D_jax,
                grad_colors_precomp_jax,
                grad_opacities_jax,
                grad_means3D_jax,
                grad_cov3Ds_precomp_jax,
                grad_sh_jax,
                grad_scales_jax, grad_rotations_jax, grad_conic)