//! Per-frame uniform data and its GPU layout.

use bytemuck::{Pod, Zeroable};

use crate::params::{DenoiseParams, GtaoParams};

/// Per-frame uniform data supplied by the caller.
///
/// The renderer stores this in [`crate::GtaoRenderer::uniforms`].
#[derive(Clone, Copy, Debug)]
pub struct GtaoUniforms {
    /// The projection matrix: camera local space to clip space.
    pub camera_projection: [[f32; 4]; 4],
    /// The inverse projection matrix: clip space to camera local space.
    pub inv_camera_projection: [[f32; 4]; 4],
    /// Parameters of the GTAO pass, mutable per frame.
    pub params: GtaoParams,
}

impl Default for GtaoUniforms {
    fn default() -> Self {
        Self {
            camera_projection: IDENTITY,
            inv_camera_projection: IDENTITY,
            params: GtaoParams::default(),
        }
    }
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Exact GPU layout shared by the GTAO and denoise shaders.
#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuUniforms {
    camera_projection: [[f32; 4]; 4],
    inv_camera_projection: [[f32; 4]; 4],
    radius: f32,
    distance_exponent: f32,
    thickness: f32,
    distance_falloff: f32,
    scale: f32,
    denoise_rings: f32,
    denoise_radius_exponent: f32,
    ao_width: f32,
    ao_height: f32,
    denoise_luma_phi: f32,
    denoise_depth_phi: f32,
    denoise_normal_phi: f32,
    denoise_radius: f32,
    /// Pad the struct to a multiple of 16 bytes for uniform buffer layout.
    _padding: [f32; 3],
}

impl GpuUniforms {
    pub(crate) fn new(uniforms: GtaoUniforms, denoise: DenoiseParams, ao_size: (u32, u32)) -> Self {
        let params = uniforms.params;
        Self {
            camera_projection: uniforms.camera_projection,
            inv_camera_projection: uniforms.inv_camera_projection,
            radius: params.radius,
            distance_exponent: params.distance_exponent,
            thickness: params.thickness,
            distance_falloff: params.distance_falloff,
            scale: params.scale,
            denoise_rings: denoise.rings,
            denoise_radius_exponent: denoise.radius_exponent,
            ao_width: ao_size.0 as f32,
            ao_height: ao_size.1 as f32,
            denoise_luma_phi: denoise.luma_phi,
            denoise_depth_phi: denoise.depth_phi,
            denoise_normal_phi: denoise.normal_phi,
            denoise_radius: denoise.radius,
            _padding: [0.0; 3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_uniforms_layout_is_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<GpuUniforms>(), 192);
        assert_eq!(std::mem::align_of::<GpuUniforms>(), 16);
    }
}
