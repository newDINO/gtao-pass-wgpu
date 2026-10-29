//! Tunable parameters of the GTAO and denoise passes.

/// Tunable parameters of the GTAO pass. The defaults match three.js.
#[derive(Clone, Copy, Debug)]
pub struct GtaoParams {
    /// View-space radius of the occlusion search.
    pub radius: f32,
    /// Falloff exponent applied to the sample distance.
    pub distance_exponent: f32,
    /// Depth difference below which a sample is considered to belong to the
    /// same surface.
    pub thickness: f32,
    /// Controls the distance falloff of the horizon samples.
    pub distance_falloff: f32,
    /// Final AO exponent. Values above one increase contrast.
    pub scale: f32,
}

impl Default for GtaoParams {
    fn default() -> Self {
        Self {
            radius: 0.25,
            distance_exponent: 1.0,
            thickness: 1.0,
            distance_falloff: 1.0,
            scale: 1.0,
        }
    }
}

/// Tunable parameters of the Poisson denoise pass. The defaults match
/// three.js' `GTAOPass`.
#[derive(Clone, Copy, Debug)]
pub struct DenoiseParams {
    /// Luminance similarity threshold.
    pub luma_phi: f32,
    /// Depth similarity threshold.
    pub depth_phi: f32,
    /// Normal similarity exponent.
    pub normal_phi: f32,
    /// Denoise radius in pixels (at the AO resolution).
    pub radius: f32,
    /// Number of sample rings.
    pub rings: f32,
    /// Radius exponent.
    pub radius_exponent: f32,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            luma_phi: 10.0,
            depth_phi: 2.0,
            normal_phi: 3.0,
            radius: 8.0,
            rings: 2.0,
            radius_exponent: 2.0,
        }
    }
}
