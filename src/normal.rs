//! Normal input types.

/// The space a normal texture is stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalSpace {
    /// Normal is in camera local space (view space).
    ///
    /// This is the cheapest option, because the GTAO and denoise passes work
    /// in view space and can use the decoded normal directly.
    CameraLocalSpace,
    /// Normal is in world space.
    ///
    /// The renderer transforms the normal into view space using
    /// [`crate::GtaoUniforms::camera_view_rotation`] in the shader.
    WorldSpace,
}

/// How a normal texture encodes its normals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalEncoding {
    /// Plain encoding or no encoding, stored as a signed vec3f.
    Plain,
    /// Octahedral encoding, stored as a signed vec2f ranging [-1, 1].
    /// Example wgsl code to encode and decode it:
    /// ```wgsl
    /// fn encode(n_in: vec3f) -> vec2f {
    ///     let n = n_in / (abs(n_in.x) + abs(n_in.y) + abs(n_in.z));
    ///     var p = n.xy;
    ///
    ///     if (n.z < 0.0) {
    ///         p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
    ///     }
    ///
    ///     return p;
    /// }
    /// fn decode(p: vec2f) -> vec3f {
    ///     var n = vec3f(p, 1.0 - abs(p.x) - abs(p.y));
    ///
    ///     if (n.z < 0.0) {
    ///         n = vec3f(
    ///             (1.0 - abs(n.yx)) * select(vec2f(-1.0), vec2f(1.0), n.xy >= vec2f(0.0)),
    ///             n.z
    ///         );
    ///     }
    ///
    ///     return normalize(n);
    /// }
    /// ```
    OctahedralSigned,
}

/// The normal texture and its conventions.
pub struct NormalInput<'a> {
    /// The normal texture view.
    pub view: &'a wgpu::TextureView,
    /// How the texture encodes its normals.
    pub encoding: NormalEncoding,
    /// The space the normals are stored in.
    pub space: NormalSpace,
}

/// Pipeline-time normal settings derived from [`NormalInput`].
///
/// Both the encoding and the space are baked into the pipelines as shader
/// overrides, so they cannot change without recreating the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NormalConfig {
    pub(crate) encoding: Option<NormalEncoding>,
    pub(crate) space: NormalSpace,
}

impl NormalConfig {
    pub(crate) fn from_input(input: Option<&NormalInput<'_>>) -> Self {
        match input {
            Some(input) => Self {
                encoding: Some(input.encoding),
                space: input.space,
            },
            None => Self {
                encoding: None,
                space: NormalSpace::CameraLocalSpace,
            },
        }
    }

    /// Value of the `NORMAL_MODE` shader override.
    pub(crate) fn mode(self) -> i32 {
        match self.encoding {
            None => 0,
            Some(NormalEncoding::Plain) => 1,
            Some(NormalEncoding::OctahedralSigned) => 2,
        }
    }

    /// Value of the `NORMAL_SPACE` shader override.
    pub(crate) fn space_override(self) -> i32 {
        match self.space {
            NormalSpace::CameraLocalSpace => 0,
            NormalSpace::WorldSpace => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_config_maps_to_shader_overrides() {
        let view = NormalConfig {
            encoding: Some(NormalEncoding::Plain),
            space: NormalSpace::CameraLocalSpace,
        };
        assert_eq!(view.mode(), 1);
        assert_eq!(view.space_override(), 0);

        let world = NormalConfig {
            encoding: Some(NormalEncoding::OctahedralSigned),
            space: NormalSpace::WorldSpace,
        };
        assert_eq!(world.mode(), 2);
        assert_eq!(world.space_override(), 1);

        let depth = NormalConfig::from_input(None);
        assert_eq!(depth.mode(), 0);
        assert_eq!(depth.space_override(), 0);
    }
}
