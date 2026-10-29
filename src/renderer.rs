//! The GTAO renderer.

use wgpu::util::DeviceExt;

use crate::normal::{NormalConfig, NormalInput};
use crate::params::{DenoiseParams, GtaoParams};
use crate::pipeline::{
    create_ao_texture, create_denoise_bind_group, create_denoise_bind_group_layout,
    create_denoise_pipeline, create_dummy_normal, create_gtao_bind_group,
    create_gtao_bind_group_layout, create_gtao_pipeline, create_sampler,
};
use crate::uniforms::{GpuUniforms, GtaoUniforms};
use crate::util::{scale_resolution, texture_size};

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
    /// Initial parameters of the GTAO pass. Mutate [`GtaoUniforms::params`] to
    /// change them per frame.
    pub params: GtaoParams,
    /// Parameters of the Poisson denoise pass. Use [`None`] to skip denoising;
    /// in that case [`GtaoRenderer::ao_view`] returns the raw AO.
    pub denoise: Option<DenoiseParams>,
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
    normal_config: NormalConfig,
    denoise_params: Option<DenoiseParams>,

    /// Per-frame uniform data. Mutate this before [`GtaoRenderer::render`].
    pub uniforms: GtaoUniforms,
}

impl GtaoRenderer {
    pub fn new(device: &wgpu::Device, descriptor: &GtaoDescriptor) -> Self {
        let source_size = texture_size(descriptor.depth_input);
        let ao_size = scale_resolution(source_size.0, source_size.1, descriptor.resolution_scale);
        let normal_config = NormalConfig::from_input(descriptor.normal_input.as_ref());

        let uniforms = GtaoUniforms {
            params: descriptor.params,
            ..Default::default()
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gtao uniforms"),
            contents: bytemuck::bytes_of(&GpuUniforms::new(
                uniforms,
                descriptor.denoise.unwrap_or_default(),
                ao_size,
            )),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let gtao_bind_group_layout = create_gtao_bind_group_layout(device);

        let gtao_pipeline = create_gtao_pipeline(
            device,
            &gtao_bind_group_layout,
            descriptor.ao_format,
            normal_config,
        );

        let denoise_bind_group_layout = descriptor
            .denoise
            .map(|_| create_denoise_bind_group_layout(device, descriptor.ao_format));

        let denoise_pipeline = denoise_bind_group_layout.as_ref().map(|layout| {
            create_denoise_pipeline(device, layout, descriptor.ao_format, normal_config)
        });

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
            normal_config,
            denoise_params: descriptor.denoise,
            uniforms,
        }
    }

    /// Record the GTAO (and optional denoise) pass into `encoder`.
    ///
    /// Uploads [`GtaoRenderer::uniforms`] and the current output resolution
    /// before recording the passes.
    pub fn render(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        let uniforms = GpuUniforms::new(
            self.uniforms,
            self.denoise_params.unwrap_or_default(),
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
    /// `normal_view` must match the normal input mode used at construction:
    /// pass `Some` if the renderer was created with a normal input, and `None`
    /// if it reconstructs normals from depth. Changing the mode requires
    /// creating a new [`GtaoRenderer`].
    ///
    /// Passing `None` for `resolution_scale` keeps the current scale.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        normal_view: Option<&wgpu::TextureView>,
        resolution_scale: Option<f32>,
    ) {
        assert_eq!(
            self.normal_config.encoding.is_some(),
            normal_view.is_some(),
            "resize: normal_view must match the normal input mode used at construction; \
             create a new GtaoRenderer to change it",
        );

        let source_size = texture_size(depth_view);
        let scale = resolution_scale.unwrap_or(self.resolution_scale);
        let ao_size = scale_resolution(source_size.0, source_size.1, scale);

        let ao_raw_view = create_ao_texture(device, ao_size, self.ao_format, "gtao raw ao");
        let ao_final_view = self
            .denoise_params
            .is_some()
            .then(|| create_ao_texture(device, ao_size, self.ao_format, "gtao denoised ao"));

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
