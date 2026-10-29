//! Normal input types.

/// The space a normal texture is stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalSpace {
    /// Normal is in camera local space (view space).
    CameraLocalSpace,
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
    pub view: &'a wgpu::TextureView,
    pub encoding: NormalEncoding,
    pub space: NormalSpace,
}

pub(crate) fn normal_mode(encoding: Option<NormalEncoding>) -> i32 {
    match encoding {
        None => 0,
        Some(NormalEncoding::Plain) => 1,
        Some(NormalEncoding::OctahedralSigned) => 2,
    }
}
