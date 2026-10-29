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
    /// The world-to-view rotation, column-major.
    ///
    /// Only used when [`crate::NormalSpace::WorldSpace`] is selected. For a
    /// rigid camera this is the upper-left 3x3 block of the view matrix. The
    /// translation is intentionally omitted because it does not affect
    /// direction vectors. If the view matrix contains non-uniform scaling,
    /// pass the inverse-transpose normal matrix instead.
    pub camera_view_rotation: [[f32; 3]; 3],
    /// Parameters of the GTAO pass, mutable per frame.
    pub params: GtaoParams,
}

impl Default for GtaoUniforms {
    fn default() -> Self {
        Self {
            camera_projection: IDENTITY,
            inv_camera_projection: IDENTITY,
            camera_view_rotation: IDENTITY_3,
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

const IDENTITY_3: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// GPU representation of a column-major `mat3x3f`.
///
/// WGSL pads each matrix column to 16 bytes in the uniform address space, so
/// the public `[[f32; 3]; 3]` cannot be uploaded directly.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuMat3 {
    cols: [[f32; 4]; 3],
}

impl From<[[f32; 3]; 3]> for GpuMat3 {
    fn from(m: [[f32; 3]; 3]) -> Self {
        Self {
            cols: [
                [m[0][0], m[0][1], m[0][2], 0.0],
                [m[1][0], m[1][1], m[1][2], 0.0],
                [m[2][0], m[2][1], m[2][2], 0.0],
            ],
        }
    }
}

/// Exact GPU layout shared by the GTAO and denoise shaders.
#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuUniforms {
    camera_projection: [[f32; 4]; 4],
    inv_camera_projection: [[f32; 4]; 4],
    camera_view_rotation: GpuMat3,
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
            camera_view_rotation: uniforms.camera_view_rotation.into(),
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
    use crate::pipeline::GTAO_SHADER;

    #[test]
    fn gpu_mat3_matches_wgsl_layout() {
        // WGSL pads every mat3x3f column to 16 bytes.
        assert_eq!(std::mem::size_of::<GpuMat3>(), 48);
        assert_eq!(std::mem::align_of::<GpuMat3>(), 16);
    }

    #[test]
    fn gpu_mat3_is_column_major_and_padded() {
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        assert_eq!(
            GpuMat3::from(m).cols,
            [
                [1.0, 2.0, 3.0, 0.0],
                [4.0, 5.0, 6.0, 0.0],
                [7.0, 8.0, 9.0, 0.0],
            ]
        );
    }

    #[test]
    fn wgsl_uniform_layout_matches_rust() {
        let module =
            wgpu::naga::front::wgsl::parse_str(GTAO_SHADER).expect("GTAO shader should parse");

        let (span, members) = module
            .types
            .iter()
            .find_map(|(_, ty)| {
                if ty.name.as_deref() == Some("Uniforms")
                    && let wgpu::naga::TypeInner::Struct { span, members, .. } = &ty.inner
                {
                    return Some((*span, members));
                }
                None
            })
            .expect("Uniforms struct should exist");

        let offset = |name: &str| {
            members
                .iter()
                .find(|member| member.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("missing member {name}"))
                .offset as usize
        };

        // Check the layout boundaries that matter: the matrix block, the
        // mat3x3f padding, the padding tail and the total size. If a field is
        // added on only one side, the size or padding offset will diverge.
        assert_eq!(
            offset("camera_projection"),
            std::mem::offset_of!(GpuUniforms, camera_projection),
        );
        assert_eq!(
            offset("inv_camera_projection"),
            std::mem::offset_of!(GpuUniforms, inv_camera_projection),
        );
        assert_eq!(
            offset("camera_view_rotation"),
            std::mem::offset_of!(GpuUniforms, camera_view_rotation),
        );
        assert_eq!(offset("radius"), std::mem::offset_of!(GpuUniforms, radius),);

        // The Rust struct groups the three padding scalars into one array.
        let padding = std::mem::offset_of!(GpuUniforms, _padding);
        assert_eq!(offset("_padding_0"), padding);
        assert_eq!(offset("_padding_1"), padding + 4);
        assert_eq!(offset("_padding_2"), padding + 8);

        assert_eq!(
            span as usize,
            std::mem::size_of::<GpuUniforms>(),
            "WGSL struct size must match the Rust upload type",
        );
    }
}
