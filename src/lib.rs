//! Ground truth ambient occlusion (GTAO) for [`wgpu`].
//!
//! This is a fairly direct WGSL port of the GTAO implementation in three.js
//! (`GTAOShader` + `PoissonDenoiseShader`). The renderer expects a depth
//! texture and, optionally, a view-space normal texture. Both are sampled at
//! the current output resolution, so the AO buffer can be rendered at a lower
//! resolution than the scene.
//!
//! # Coordinate conventions
//!
//! * Depth is the standard WebGPU `[0, 1]` depth range (not reversed).
//! * Matrices are column-major, like three.js / `glam`.
//! * Normals passed through [`NormalInput`] are in camera local (view) space.
//! * Depth and normal textures are sampled with their top-left corner at UV
//!   `(0, 0)`.
//!
//! # Example
//!
//! ```no_run
//! # use wgpu_gtao::*;
//! # fn example(device: &wgpu::Device, queue: &wgpu::Queue, depth: &wgpu::TextureView) {
//! let projection = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
//! let projection_inverse = projection;
//! let mut gtao = GtaoRenderer::new(
//!     device,
//!     &GtaoDescriptor {
//!         depth_input: depth,
//!         normal_input: None,
//!         ao_format: wgpu::TextureFormat::R8Unorm,
//!         resolution_scale: 1.0,
//!         params: GtaoParams::default(),
//!         denoise: Some(DenoiseParams::default()),
//!     },
//! );
//!
//! gtao.uniforms.camera_projection = projection;
//! gtao.uniforms.inv_camera_projection = projection_inverse;
//!
//! let mut encoder = device.create_command_encoder(&Default::default());
//! gtao.render(queue, &mut encoder);
//! queue.submit([encoder.finish()]);
//! # }
//! ```

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

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

/// Description of a [`GtaoRenderer`].
pub struct GtaoDescriptor<'a> {
    /// A single-sample depth texture view. For combined depth-stencil formats,
    /// create the view with [`wgpu::TextureAspect::DepthOnly`].
    pub depth_input: &'a wgpu::TextureView,
    /// If `normal_input` is [`None`], normal will be reconstructed from depth.
    pub normal_input: Option<NormalInput<'a>>,
    /// The format of the AO texture. Must be a single-channel float format with
    /// render-attachment support, e.g. [`wgpu::TextureFormat::R8Unorm`] or
    /// [`wgpu::TextureFormat::R16Float`].
    pub ao_format: wgpu::TextureFormat,
    /// The size of the output ao texture view determined by the size of the input depth texture,
    /// processed by [`scale_resolution`] function.
    pub resolution_scale: f32,
    /// Parameters of the GTAO pass.
    pub params: GtaoParams,
    /// Parameters of the Poisson denoise pass. Use [`None`] to skip denoising;
    /// in that case [`GtaoRenderer::ao_view`] returns the raw AO.
    pub denoise: Option<DenoiseParams>,
}

/// Per-frame uniform data supplied by the caller.
///
/// The renderer stores this in [`GtaoRenderer::uniforms`].
#[derive(Clone, Copy, Debug)]
pub struct GtaoUniforms {
    /// The projection matrix: camera local space to clip space.
    pub camera_projection: [[f32; 4]; 4],
    /// The inverse projection matrix: clip space to camera local space.
    pub inv_camera_projection: [[f32; 4]; 4],
}

impl Default for GtaoUniforms {
    fn default() -> Self {
        Self {
            camera_projection: IDENTITY,
            inv_camera_projection: IDENTITY,
        }
    }
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuUniforms {
    camera_projection: [[f32; 4]; 4],
    inv_camera_projection: [[f32; 4]; 4],
    radius: f32,
    distance_exponent: f32,
    thickness: f32,
    distance_falloff: f32,
    scale: f32,
    normal_mode: f32,
    denoise_rings: f32,
    denoise_radius_exponent: f32,
    ao_width: f32,
    ao_height: f32,
    denoise_luma_phi: f32,
    denoise_depth_phi: f32,
    denoise_normal_phi: f32,
    denoise_radius: f32,
    /// Pad the struct to a multiple of 16 bytes for uniform buffer layout.
    _padding: [f32; 2],
}

/// A GTAO renderer.
pub struct GtaoRenderer {
    gtao_pipeline: wgpu::RenderPipeline,
    denoise_pipeline: Option<wgpu::RenderPipeline>,

    gtao_bind_group_layout: wgpu::BindGroupLayout,
    denoise_bind_group_layout: Option<wgpu::BindGroupLayout>,

    uniform_buffer: wgpu::Buffer,
    gtao_bind_group: wgpu::BindGroup,
    denoise_bind_group: Option<wgpu::BindGroup>,

    /// Raw GTAO output, used as the denoise input.
    ao_raw_view: wgpu::TextureView,
    /// Final output. [`None`] when denoising is disabled, in which case
    /// [`Self::ao_raw_view`] is returned by [`GtaoRenderer::ao_view`].
    ao_final_view: Option<wgpu::TextureView>,

    sampler: wgpu::Sampler,
    ao_format: wgpu::TextureFormat,
    ao_size: (u32, u32),
    resolution_scale: f32,
    normal_encoding: Option<NormalEncoding>,
    params: GtaoParams,
    denoise_params: Option<DenoiseParams>,

    /// Per-frame uniform data. Mutate this before [`GtaoRenderer::render`].
    pub uniforms: GtaoUniforms,
}

impl GtaoRenderer {
    pub fn new(device: &wgpu::Device, descriptor: &GtaoDescriptor) -> Self {
        let source_size = texture_size(descriptor.depth_input);
        let ao_size = scale_resolution(source_size.0, source_size.1, descriptor.resolution_scale);
        let normal_encoding = descriptor.normal_input.as_ref().map(|n| n.encoding);
        let normal_mode = normal_mode(normal_encoding);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gtao uniforms"),
            contents: bytemuck::bytes_of(&GpuUniforms::new(
                GtaoUniforms::default(),
                descriptor.params,
                descriptor.denoise.unwrap_or_default(),
                normal_mode,
                ao_size,
            )),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let gtao_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gtao bind group layout"),
                entries: &[
                    uniform_entry(0),
                    texture_entry(1, wgpu::TextureSampleType::Depth),
                    texture_entry(2, wgpu::TextureSampleType::Float { filterable: false }),
                ],
            });

        let gtao_pipeline =
            create_gtao_pipeline(device, &gtao_bind_group_layout, descriptor.ao_format);

        let denoise_bind_group_layout = descriptor.denoise.map(|_| {
            let sample_type = ao_sample_type(descriptor.ao_format);
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gtao denoise bind group layout"),
                entries: &[
                    uniform_entry(0),
                    texture_entry(1, sample_type),
                    texture_entry(2, wgpu::TextureSampleType::Depth),
                    texture_entry(3, wgpu::TextureSampleType::Float { filterable: false }),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(if is_filterable(sample_type) {
                            wgpu::SamplerBindingType::Filtering
                        } else {
                            wgpu::SamplerBindingType::NonFiltering
                        }),
                        count: None,
                    },
                ],
            })
        });

        let denoise_pipeline = denoise_bind_group_layout
            .as_ref()
            .map(|layout| create_denoise_pipeline(device, layout, descriptor.ao_format));

        let ao_raw_view = create_ao_texture(device, ao_size, descriptor.ao_format, "gtao raw ao");
        let ao_final_view = descriptor
            .denoise
            .is_some()
            .then(|| create_ao_texture(device, ao_size, descriptor.ao_format, "gtao denoised ao"));

        // A valid texture must be bound even when normals are reconstructed
        // from depth. The shader will not read it in that case.
        let dummy_normal = descriptor
            .normal_input
            .is_none()
            .then(|| create_dummy_normal(device));
        let normal_view = descriptor
            .normal_input
            .as_ref()
            .map(|n| n.view)
            .or(dummy_normal.as_ref())
            .expect("normal view is always available");

        let gtao_bind_group = create_gtao_bind_group(
            device,
            &gtao_bind_group_layout,
            &uniform_buffer,
            descriptor.depth_input,
            normal_view,
        );

        let sampler = create_sampler(device, descriptor.ao_format);
        let denoise_bind_group = denoise_bind_group_layout.as_ref().map(|layout| {
            create_denoise_bind_group(
                device,
                layout,
                &uniform_buffer,
                &ao_raw_view,
                descriptor.depth_input,
                normal_view,
                &sampler,
            )
        });

        Self {
            gtao_pipeline,
            denoise_pipeline,
            gtao_bind_group_layout,
            denoise_bind_group_layout,
            uniform_buffer,
            gtao_bind_group,
            denoise_bind_group,
            ao_raw_view,
            ao_final_view,
            sampler,
            ao_format: descriptor.ao_format,
            ao_size,
            resolution_scale: descriptor.resolution_scale,
            normal_encoding,
            params: descriptor.params,
            denoise_params: descriptor.denoise,
            uniforms: GtaoUniforms::default(),
        }
    }

    /// Record the GTAO (and optional denoise) pass into `encoder`.
    ///
    /// Uploads [`GtaoRenderer::uniforms`] and the current output resolution
    /// before recording the passes.
    pub fn render(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        let uniforms = GpuUniforms::new(
            self.uniforms,
            self.params,
            self.denoise_params.unwrap_or_default(),
            normal_mode(self.normal_encoding),
            self.ao_size,
        );
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gtao pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ao_raw_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.gtao_pipeline);
            pass.set_bind_group(0, &self.gtao_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        if let (Some(pipeline), Some(bind_group), Some(final_view)) = (
            &self.denoise_pipeline,
            &self.denoise_bind_group,
            &self.ao_final_view,
        ) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gtao denoise pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: final_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// The final AO texture view. This is the denoised output when denoising
    /// is enabled, otherwise it is the raw GTAO output.
    pub fn ao_view(&self) -> &wgpu::TextureView {
        self.ao_final_view.as_ref().unwrap_or(&self.ao_raw_view)
    }

    /// The size of the AO texture in pixels.
    pub fn ao_size(&self) -> (u32, u32) {
        self.ao_size
    }

    /// Recreate [`wgpu::BindGroup`] from resized depth and normal texture view.
    /// Recreate ao texture from new input size and resolution scale.
    ///
    /// Passing `None` for `normal_view` switches to depth-reconstructed
    /// normals. Passing `None` for `resolution_scale` keeps the current scale.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        normal_view: Option<&wgpu::TextureView>,
        resolution_scale: Option<f32>,
    ) {
        let source_size = texture_size(depth_view);
        let scale = resolution_scale.unwrap_or(self.resolution_scale);
        let ao_size = scale_resolution(source_size.0, source_size.1, scale);

        let ao_raw_view = create_ao_texture(device, ao_size, self.ao_format, "gtao raw ao");
        let ao_final_view = self
            .denoise_params
            .is_some()
            .then(|| create_ao_texture(device, ao_size, self.ao_format, "gtao denoised ao"));

        self.normal_encoding =
            normal_view.map(|_| self.normal_encoding.unwrap_or(NormalEncoding::Plain));

        let dummy_normal = normal_view.is_none().then(|| create_dummy_normal(device));
        let normal_view = normal_view
            .or(dummy_normal.as_ref())
            .expect("normal view is always available");

        let gtao_bind_group = create_gtao_bind_group(
            device,
            &self.gtao_bind_group_layout,
            &self.uniform_buffer,
            depth_view,
            normal_view,
        );

        let denoise_bind_group = self.denoise_bind_group_layout.as_ref().map(|layout| {
            create_denoise_bind_group(
                device,
                layout,
                &self.uniform_buffer,
                &ao_raw_view,
                depth_view,
                normal_view,
                &self.sampler,
            )
        });

        self.resolution_scale = scale;
        self.ao_size = ao_size;
        self.gtao_bind_group = gtao_bind_group;
        self.denoise_bind_group = denoise_bind_group;
        self.ao_raw_view = ao_raw_view;
        self.ao_final_view = ao_final_view;
    }
}

impl GpuUniforms {
    fn new(
        uniforms: GtaoUniforms,
        params: GtaoParams,
        denoise: DenoiseParams,
        normal_mode: f32,
        ao_size: (u32, u32),
    ) -> Self {
        Self {
            camera_projection: uniforms.camera_projection,
            inv_camera_projection: uniforms.inv_camera_projection,
            radius: params.radius,
            distance_exponent: params.distance_exponent,
            thickness: params.thickness,
            distance_falloff: params.distance_falloff,
            scale: params.scale,
            normal_mode,
            denoise_rings: denoise.rings,
            denoise_radius_exponent: denoise.radius_exponent,
            ao_width: ao_size.0 as f32,
            ao_height: ao_size.1 as f32,
            denoise_luma_phi: denoise.luma_phi,
            denoise_depth_phi: denoise.depth_phi,
            denoise_normal_phi: denoise.normal_phi,
            denoise_radius: denoise.radius,
            _padding: [0.0; 2],
        }
    }
}

fn normal_mode(encoding: Option<NormalEncoding>) -> f32 {
    match encoding {
        None => 0.0,
        Some(NormalEncoding::Plain) => 1.0,
        Some(NormalEncoding::OctahedralSigned) => 2.0,
    }
}

fn texture_size(view: &wgpu::TextureView) -> (u32, u32) {
    let size = view.texture().size();
    (size.width, size.height)
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn is_filterable(sample_type: wgpu::TextureSampleType) -> bool {
    matches!(
        sample_type,
        wgpu::TextureSampleType::Float { filterable: true }
    )
}

fn ao_sample_type(format: wgpu::TextureFormat) -> wgpu::TextureSampleType {
    if format.components() != 1 {
        panic!("GTAO ao_format must be a single-channel format, got {format:?}");
    }
    match format.sample_type(None, None) {
        Some(sample_type @ wgpu::TextureSampleType::Float { .. }) => sample_type,
        _ => panic!("GTAO ao_format must be a float texture format"),
    }
}

fn create_ao_texture(
    device: &wgpu::Device,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_dummy_normal(device: &wgpu::Device) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gtao dummy normal"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_sampler(device: &wgpu::Device, ao_format: wgpu::TextureFormat) -> wgpu::Sampler {
    let filterable = is_filterable(ao_sample_type(ao_format));
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("gtao denoise sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: if filterable {
            wgpu::FilterMode::Linear
        } else {
            wgpu::FilterMode::Nearest
        },
        min_filter: if filterable {
            wgpu::FilterMode::Linear
        } else {
            wgpu::FilterMode::Nearest
        },
        ..Default::default()
    })
}

fn create_gtao_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    ao_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gtao shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gtao.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gtao pipeline layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gtao pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: ao_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_denoise_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    ao_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gtao poisson denoise shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("poisson_denoise.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gtao denoise pipeline layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gtao denoise pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: ao_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_gtao_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    depth_view: &wgpu::TextureView,
    normal_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gtao bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(normal_view),
            },
        ],
    })
}

fn create_denoise_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    ao_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    normal_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gtao denoise bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(ao_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(normal_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

pub fn scale_resolution(w: u32, h: u32, scale: f32) -> (u32, u32) {
    fn scale_single(a: u32, s: f32) -> u32 {
        ((a as f32 * s) as u32).max(1)
    }
    (scale_single(w, scale), scale_single(h, scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_are_valid_wgsl() {
        for source in [
            include_str!("gtao.wgsl"),
            include_str!("poisson_denoise.wgsl"),
        ] {
            let module = wgpu::naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("invalid WGSL: {error}"));
            let mut validator = wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|error| panic!("invalid WGSL module: {error}"));
        }
    }

    #[test]
    fn resolution_scale_never_returns_zero() {
        assert_eq!(scale_resolution(0, 10, 0.5), (1, 5));
        assert_eq!(scale_resolution(1920, 1080, 0.5), (960, 540));
    }
}
