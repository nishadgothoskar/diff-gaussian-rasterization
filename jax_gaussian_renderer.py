import jax 
import jax.numpy as jnp 
import functools
from jax_renderer_primitives import _build_rasterize_gaussians_fwd_primitive, _build_rasterize_gaussians_bwd_primitive

################################
# MAIN RASTERIZER
################################
_rasterizer_fwd_prim = _build_rasterize_gaussians_fwd_primitive()
_rasterizer_bwd_prim = _build_rasterize_gaussians_bwd_primitive()

@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def rasterize(  
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        sh,
        _raster_settings_fcn
):  
    raster_settings = _raster_settings_fcn()

    out = _rasterizer_fwd_prim.bind(
        raster_settings.bg,#
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp, 
        raster_settings.viewmatrix,#
        raster_settings.projmatrix,#
        sh,#
        raster_settings.campos,#
        tanfovx=raster_settings.tanfovx, #
        tanfovy=raster_settings.tanfovy, #
        image_height=raster_settings.image_height, #
        image_width=raster_settings.image_width,  #
        sh_degree=raster_settings.sh_degree#
    )
    return out 


def rasterize_bwd(
        _raster_settings_fcn,
        saved_tensors,
        grads_color_radii,   
):  
    grad_out_color, _ = grads_color_radii
    r = _raster_settings_fcn()

    bg, viewmatrix, projmatrix, campos, tanfovx, tanfovy, image_height, image_width, sh_degree = r.bg, r.viewmatrix, r.projmatrix, r.campos, r.tanfovx, r.tanfovy, r.image_height, r.image_width, r.sh_degree
    num_rendered, colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = saved_tensors

    out = _rasterizer_bwd_prim.bind(
        bg,
        means3D,         
        radii, 
        colors_precomp, 
        scales,         
        rotations, 
        cov3Ds_precomp, 
        viewmatrix, 
        projmatrix,         
        grad_out_color,         
        sh, 
        sh_degree, 
        geomBuffer, 
        num_rendered,  
        binningBuffer, 
        imgBuffer, 
        tanfovx=tanfovx,
        tanfovy=tanfovy, 
        sh_degree=0 
    )
    grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, _ = out

    return grad_means3D, grad_colors_precomp, grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp, grad_sh


def rasterize_fwd(  
    means3D,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    sh,
    _raster_settings_fcn
):  
    out = rasterize( 
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        sh,
        _raster_settings_fcn
    )
    r = _raster_settings_fcn()                                                                    
    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = out

    # keep relevant tensors for bwd
    saved_tensors = (num_rendered, colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
    
    return (color, radii), saved_tensors


rasterize.defvjp(rasterize_fwd, rasterize_bwd)
