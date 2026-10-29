// Offscreen GTAO demo.
//
// Run with:
//     cargo run --example gtao_demo
//
// It writes two BMP files into the current directory:
//     gtao_example.bmp  - lit scene with AO applied
//     gtao_ao.bmp       - the raw AO buffer (white = unoccluded)

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wgpu_gtao::{
    DenoiseParams, GtaoDescriptor, GtaoParams, GtaoRenderer, NormalEncoding, NormalInput,
    NormalSpace,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

type Mat4 = [[f32; 4]; 4];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_projection: Mat4,
    view: Mat4,
}

fn main() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("no wgpu adapter available");
    println!("adapter: {:?}", adapter.get_info());

    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gtao example device"),
        ..Default::default()
    }))
    .expect("failed to create device");

    let depth_texture = create_texture(
        &device,
        "depth",
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let depth_view = depth_texture.create_view(&Default::default());

    let normal_texture = create_texture(
        &device,
        "normal",
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let normal_view = normal_texture.create_view(&Default::default());

    let color_texture = create_texture(
        &device,
        "color",
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let color_view = color_texture.create_view(&Default::default());

    let final_texture = create_texture(
        &device,
        "final",
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let final_view = final_texture.create_view(&Default::default());

    let ao_debug_texture = create_texture(
        &device,
        "ao debug",
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let ao_debug_view = ao_debug_texture.create_view(&Default::default());

    let mut gtao = GtaoRenderer::new(
        &device,
        &GtaoDescriptor {
            depth_input: &depth_view,
            normal_input: Some(NormalInput {
                view: &normal_view,
                encoding: NormalEncoding::Plain,
                space: NormalSpace::CameraLocalSpace,
            }),
            ao_format: wgpu::TextureFormat::R8Unorm,
            resolution_scale: 1.0,
            params: GtaoParams::default(),
            denoise: Some(DenoiseParams::default()),
        },
    );

    let view = look_at([7.0, 5.0, 9.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]);
    let projection = perspective(
        60.0f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        100.0,
    );
    let view_projection = mat4_mul(projection, view);

    gtao.uniforms.camera_projection = projection;
    gtao.uniforms.inv_camera_projection = mat4_inverse(projection);

    let camera_uniform = CameraUniform {
        view_projection,
        view,
    };
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera uniform"),
        contents: bytemuck::bytes_of(&camera_uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera bind group"),
        layout: &camera_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });

    let mut vertices = Vec::new();
    build_scene(&mut vertices);
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scene vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let gbuffer_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gbuffer shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gtao_demo/gbuffer.wgsl").into()),
    });
    let gbuffer_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gbuffer pipeline layout"),
        bind_group_layouts: &[Some(&camera_layout)],
        immediate_size: 0,
    });
    let gbuffer_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gbuffer pipeline"),
        layout: Some(&gbuffer_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &gbuffer_shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 24,
                        shader_location: 2,
                    },
                ],
            })],
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &gbuffer_shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        multiview_mask: None,
        cache: None,
    });

    let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("composite shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gtao_demo/composite.wgsl").into()),
    });
    let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite layout"),
        entries: &[
            texture_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
            texture_entry(1, wgpu::TextureSampleType::Float { filterable: true }),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("composite sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite bind group"),
        layout: &composite_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(gtao.ao_view()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&composite_sampler),
            },
        ],
    });
    let composite_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite pipeline layout"),
            bind_group_layouts: &[Some(&composite_layout)],
            immediate_size: 0,
        });
    let composite_pipeline = create_composite_pipeline(
        &device,
        &composite_pipeline_layout,
        &composite_shader,
        "fs_main",
    );
    let ao_pipeline = create_composite_pipeline(
        &device,
        &composite_pipeline_layout,
        &composite_shader,
        "fs_ao",
    );

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gbuffer pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &normal_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&gbuffer_pipeline);
        pass.set_bind_group(0, &camera_bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }

    gtao.render(&queue, &mut encoder);

    draw_fullscreen(
        &mut encoder,
        &composite_pipeline,
        &composite_bind_group,
        &final_view,
    );
    draw_fullscreen(
        &mut encoder,
        &ao_pipeline,
        &composite_bind_group,
        &ao_debug_view,
    );

    let padded_bytes_per_row = (WIDTH * 4).div_ceil(256) * 256;
    let readback_size = (padded_bytes_per_row * HEIGHT) as u64;
    let final_readback = create_readback_buffer(&device, readback_size, "final readback");
    let ao_readback = create_readback_buffer(&device, readback_size, "ao readback");

    copy_texture_to_buffer(
        &mut encoder,
        &final_texture,
        &final_readback,
        padded_bytes_per_row,
    );
    copy_texture_to_buffer(
        &mut encoder,
        &ao_debug_texture,
        &ao_readback,
        padded_bytes_per_row,
    );

    queue.submit([encoder.finish()]);

    let final_pixels = read_buffer(&device, &final_readback);
    let ao_pixels = read_buffer(&device, &ao_readback);

    write_bmp(
        "target/gtao_example.bmp",
        WIDTH,
        HEIGHT,
        &final_pixels,
        padded_bytes_per_row,
    )
    .expect("failed to write gtao_example.bmp");
    write_bmp(
        "target/gtao_ao.bmp",
        WIDTH,
        HEIGHT,
        &ao_pixels,
        padded_bytes_per_row,
    )
    .expect("failed to write gtao_ao.bmp");
}

fn create_composite_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry_point),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn draw_fullscreen(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("composite pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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

fn create_texture(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn create_readback_buffer(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn copy_texture_to_buffer(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    buffer: &wgpu::Buffer,
    padded_bytes_per_row: u32,
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
}

fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("failed to wait for GPU");
    rx.recv()
        .expect("map callback was dropped")
        .expect("map failed");
    let data = slice
        .get_mapped_range()
        .expect("failed to map buffer")
        .to_vec();
    buffer.unmap();
    data
}

fn write_bmp(
    path: &str,
    width: u32,
    height: u32,
    rgba: &[u8],
    row_pitch: u32,
) -> std::io::Result<()> {
    let row_size = (width * 3).div_ceil(4) * 4;
    let file_size = 54 + row_size * height;
    let mut out = Vec::with_capacity(file_size as usize);

    out.extend_from_slice(b"BM");
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(width as i32).to_le_bytes());
    out.extend_from_slice(&(height as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(row_size * height).to_le_bytes());
    out.extend_from_slice(&2835i32.to_le_bytes());
    out.extend_from_slice(&2835i32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());

    for y in (0..height).rev() {
        let start = (y * row_pitch) as usize;
        let row = &rgba[start..start + (width * 4) as usize];
        for pixel in row.as_chunks::<4>().0 {
            out.push(pixel[2]);
            out.push(pixel[1]);
            out.push(pixel[0]);
        }
        out.resize(out.len() + (row_size - width * 3) as usize, 0);
    }

    println!("Writing {}", path);

    std::fs::write(path, out)
}

fn build_scene(vertices: &mut Vec<Vertex>) {
    add_box(
        vertices,
        [-6.0, -0.3, -6.0],
        [6.0, 0.0, 6.0],
        [0.55, 0.55, 0.58],
    );
    add_box(
        vertices,
        [-6.0, 0.0, -6.3],
        [6.0, 4.0, -5.9],
        [0.68, 0.63, 0.58],
    );
    add_box(
        vertices,
        [-6.3, 0.0, -6.0],
        [-5.9, 4.0, 6.0],
        [0.58, 0.62, 0.68],
    );

    add_box(
        vertices,
        [-3.1, 0.0, -3.6],
        [-1.3, 1.8, -1.8],
        [0.82, 0.25, 0.2],
    );
    add_box(
        vertices,
        [0.8, 0.0, -4.4],
        [2.0, 1.2, -3.2],
        [0.25, 0.7, 0.3],
    );
    add_box(
        vertices,
        [2.4, 0.0, -1.1],
        [3.9, 2.5, 0.4],
        [0.2, 0.4, 0.82],
    );
    add_box(
        vertices,
        [-2.3, 0.0, 1.4],
        [-0.7, 1.0, 3.0],
        [0.85, 0.55, 0.15],
    );
    add_box(
        vertices,
        [0.2, 0.0, 2.1],
        [1.8, 0.8, 3.7],
        [0.55, 0.25, 0.7],
    );
    add_box(
        vertices,
        [-4.8, 0.0, 2.5],
        [-3.6, 2.8, 3.7],
        [0.75, 0.7, 0.2],
    );
}

fn add_box(vertices: &mut Vec<Vertex>, min: [f32; 3], max: [f32; 3], color: [f32; 3]) {
    let p = [
        [min[0], min[1], min[2]],
        [max[0], min[1], min[2]],
        [max[0], max[1], min[2]],
        [min[0], max[1], min[2]],
        [min[0], min[1], max[2]],
        [max[0], min[1], max[2]],
        [max[0], max[1], max[2]],
        [min[0], max[1], max[2]],
    ];

    let faces = [
        ([4, 5, 6, 7], [0.0, 0.0, 1.0]),
        ([1, 0, 3, 2], [0.0, 0.0, -1.0]),
        ([0, 4, 7, 3], [-1.0, 0.0, 0.0]),
        ([5, 1, 2, 6], [1.0, 0.0, 0.0]),
        ([0, 1, 5, 4], [0.0, -1.0, 0.0]),
        ([3, 7, 6, 2], [0.0, 1.0, 0.0]),
    ];

    for (indices, normal) in faces {
        push_vertex(vertices, p[indices[0]], normal, color);
        push_vertex(vertices, p[indices[1]], normal, color);
        push_vertex(vertices, p[indices[2]], normal, color);
        push_vertex(vertices, p[indices[0]], normal, color);
        push_vertex(vertices, p[indices[2]], normal, color);
        push_vertex(vertices, p[indices[3]], normal, color);
    }
}

fn push_vertex(vertices: &mut Vec<Vertex>, position: [f32; 3], normal: [f32; 3], color: [f32; 3]) {
    vertices.push(Vertex {
        position,
        normal,
        color,
    });
}

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> Mat4 {
    let f = normalize(sub(center, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = (0..4).map(|k| a[k][r] * b[c][k]).sum::<f32>();
        }
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn mat4_inverse(m: Mat4) -> Mat4 {
    // Gauss-Jordan inversion on row-major scratch storage.
    let mut a = [[0.0f32; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            a[r][c] = m[c][r];
        }
        a[r][4 + r] = 1.0;
    }

    for col in 0..4 {
        let mut pivot = col;
        for r in (col + 1)..4 {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        a.swap(col, pivot);
        let scale = a[col][col];
        for c in 0..8 {
            a[col][c] /= scale;
        }
        for r in 0..4 {
            if r != col {
                let factor = a[r][col];
                for c in 0..8 {
                    a[r][c] -= factor * a[col][c];
                }
            }
        }
    }

    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[c][r] = a[r][4 + c];
        }
    }
    out
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct ThreadWaker(std::thread::Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn demo_shaders_are_valid_wgsl() {
        for source in [
            include_str!("gtao_demo/gbuffer.wgsl"),
            include_str!("gtao_demo/composite.wgsl"),
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
}
