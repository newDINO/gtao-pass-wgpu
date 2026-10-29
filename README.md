# wgpu-gtao

Ground truth ambient occlusion (GTAO) for [`wgpu`], ported from three.js'
`GTAOShader` and `PoissonDenoiseShader`.

## Features

- GTAO with optional Poisson denoise.
- Depth-only normal reconstruction, or a supplied view-space normal texture
  (`Plain` / `OctahedralSigned`).
- Lower-resolution AO via `resolution_scale`.
- Single-channel AO output (`R8Unorm`, `R16Float`, `R32Float`, ...).

## Usage

```rust
let mut gtao = GtaoRenderer::new(
    &device,
    &GtaoDescriptor {
        depth_input: &depth_view,
        normal_input: None, // or Some(NormalInput { ... })
        ao_format: wgpu::TextureFormat::R8Unorm,
        resolution_scale: 1.0,
        params: GtaoParams::default(),
        denoise: Some(DenoiseParams::default()),
    },
);

gtao.uniforms.camera_projection = camera_projection;
gtao.uniforms.inv_camera_projection = inv_camera_projection;

let mut encoder = device.create_command_encoder(&Default::default());
// ... render your scene's depth (and optional normal) buffer first ...
gtao.render(&queue, &mut encoder);
// Sample gtao.ao_view() in your lighting/composite pass.
```

## Conventions

- Standard WebGPU depth range `[0, 1]`, not reversed.
- Column-major matrices.
- Input normals are in camera local (view) space.
- The depth/normal textures use a top-left UV origin.
- `depth_input` must be a single-sample depth texture view. For combined
  depth-stencil formats, create the view with `TextureAspect::DepthOnly`.

## Example

`examples/gtao_demo.rs` renders an offscreen scene and writes:

- `gtao_example.bmp` — lit scene with AO applied
- `gtao_ao.bmp` — raw AO

```sh
cargo run --example gtao_demo
```

The example needs a working wgpu adapter (Vulkan/Metal/DX12/GL). It does not
need a window or display server.

## Limitations

- Perspective camera only.
- `SCREEN_SPACE_RADIUS` and `SCENE_CLIP_BOX` from three.js are not implemented.
- AO sample count is fixed at 16 (3 directions, 6 steps each).
- Denoise sample count is fixed at 16.
