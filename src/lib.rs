pub enum NormalSpace {
    /// Normal is in camera local space(view space).
    CameraLocalSpace,
}

pub enum NormalEncoding {
    /// Plain encoding or no encoding, stored as a signed vec3f.
    Plain,
    /// Octahedral encoding, stored as a sigend vec2f ranging [-1, 1].
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

pub struct NormalInput<'a> {
    pub view: &'a wgpu::TextureView,
    pub encoding: NormalEncoding,
    pub space: NormalSpace,
}

pub struct GtaoDescriptor<'a> {
    pub depth_input: &'a wgpu::TextureView,
    /// If `normal_input` is [`None`], normal will be reconstructed from depth.
    pub normal_input: Option<NormalInput<'a>>,
    pub ao_format: wgpu::TextureFormat,
    /// The size of the output ao texture view determined by the size of the input depth texture,
    /// processed by [`scale_resolution`] function.
    pub resolution_scale: f32,
}

pub struct UniformData {
    /// The projection matrix: camera local space to clip space.
    pub camera_projection: [[f32; 4]; 4],
    pub inv_camera_projection: [[f32; 4]; 4],
}

pub struct GtaoRenderer {}

impl GtaoRenderer {
    pub fn new(device: &wgpu::Device, descriptor: &GtaoDescriptor) -> Self {
        todo!()
    }

    /// Update uniform data before calling [`GtaoRenderer::render`].
    pub fn update(&self, queue: &wgpu::Queue, uniform_data: UniformData) {}

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder) {}

    pub fn ao_view(&self) -> &wgpu::TextureView {
        todo!()
    }

    /// Recreate [`wgpu::BindGroup`] from resized depth and normal texture view.
    /// Recreate ao texture from new input size and resolution scale.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        normal_view: Option<&wgpu::TextureView>,
        resolution_scale: Option<f32>,
    ) {
    }
}

pub fn scale_resolution(w: u32, h: u32, scale: f32) -> (u32, u32) {
    fn scale_single(a: u32, s: f32) -> u32 {
        ((a as f32 * s) as u32).max(1)
    }
    (scale_single(w, scale), scale_single(h, scale))
}
