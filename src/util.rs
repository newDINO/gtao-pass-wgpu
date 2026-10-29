//! Small helpers shared by the renderer.

/// Scale a source resolution by `scale`, rounding down and never returning
/// zero in either dimension.
pub fn scale_resolution(w: u32, h: u32, scale: f32) -> (u32, u32) {
    fn scale_single(a: u32, s: f32) -> u32 {
        ((a as f32 * s) as u32).max(1)
    }
    (scale_single(w, scale), scale_single(h, scale))
}

pub(crate) fn texture_size(view: &wgpu::TextureView) -> (u32, u32) {
    let size = view.texture().size();
    (size.width, size.height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_scale_never_returns_zero() {
        assert_eq!(scale_resolution(0, 10, 0.5), (1, 5));
        assert_eq!(scale_resolution(1920, 1080, 0.5), (960, 540));
    }
}
