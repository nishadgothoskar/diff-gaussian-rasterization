import diff_gaussian_rasterization as dgr
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import jax.numpy as jnp
from random import randint
import jax
from jax.scipy.spatial.transform import Rotation as R
from diff_gaussian_rasterization import _C
import jax.numpy as jnp
import jax
import functools
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import batching, mlir, xla
from jax.lib import xla_client
import numpy as np
from jaxlib.hlo_helpers import custom_call
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################
# Helpers and boilerplates
################################

for _name, _value in _C.registrations().items():
    print(_name)
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

# XLA array layout in memory
def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]
    

################################
# Rasterize logic
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
            cov3Ds_precomp, 
            viewmatrix,
            projmatrix,
            sh,
            campos,
            tanfovx, tanfovy, 
            image_height, image_width,  
            sh_degree,
        ):
        float_dtype = dtypes.canonicalize_dtype(np.float32)
        int_dtype = dtypes.canonicalize_dtype(np.int32)
        num_gaussians, _ = means3D.shape

        GEOM_BUFFER_SIZE = int(1e6)
        BINNING_BUFFER_SIZE = int(1e7)
        IMG_BUFFER_SIZE = int(1e6)

        return [ShapedArray((1,), int_dtype),
                ShapedArray((3, image_height, image_width),  float_dtype),
                ShapedArray((num_gaussians,), int_dtype),
                ShapedArray((GEOM_BUFFER_SIZE,),  float_dtype),
                ShapedArray((BINNING_BUFFER_SIZE,),  float_dtype),
                ShapedArray((IMG_BUFFER_SIZE,),  float_dtype),
        ]
    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_fwd_lowering(ctx,
            bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp, 
            viewmatrix,
            projmatrix,
            sh,
            campos,
            tanfovx, tanfovy, image_height, image_width,  sh_degree
    ):
        float_to_ir = mlir.dtype_to_ir_type(np.dtype(np.float32))
        int_to_ir = mlir.dtype_to_ir_type(np.dtype(np.int32))

        num_gaussians = ctx.avals_in[1].shape[0]    
        opaque = _C.build_gaussian_rasterize_fwd_descriptor(
            image_height, image_width, sh_degree, num_gaussians, tanfovx, tanfovy,   
        )

        op_name = "rasterize_gaussians_fwd"

        operands = [bg, means3D, colors_precomp, opacities, scales, rotations,
                      cov3Ds_precomp, viewmatrix, projmatrix, sh, campos]

        operands_ctx = ctx.avals_in[:11]

        GEOM_BUFFER_SIZE = int(1e6)
        BINNING_BUFFER_SIZE = int(1e7)
        IMG_BUFFER_SIZE = int(1e6)
        output_shapes = [
            (1,), (image_height, image_width, 3), (num_gaussians,), (GEOM_BUFFER_SIZE,), (BINNING_BUFFER_SIZE,), (IMG_BUFFER_SIZE,)
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
                float_to_ir),
            mlir.ir.RankedTensorType.get(
                [BINNING_BUFFER_SIZE],
                float_to_ir),
            mlir.ir.RankedTensorType.get(
                [IMG_BUFFER_SIZE],
                float_to_ir),
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

