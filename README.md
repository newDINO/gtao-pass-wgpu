# gtao-pass-wgpu

Ground truth ambient occlusion (GTAO) for [`wgpu`], ported from three.js'
`GTAOShader` and `PoissonDenoiseShader`.

## Features

- GTAO with optional Poisson denoise.
- Depth-only normal reconstruction, or a supplied normal texture in view or
  world space (`Plain` / `OctahedralSigned`).
- Lower-resolution AO via `resolution_scale`.
- Single-channel AO output (`R8Unorm`, `R16Float`, `R32Float`, ...).

## Installation

```sh
cargo add gtao-pass-wgpu
```

## Usage

```rust
use gtao_pass_wgpu::*;

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
// Only needed when NormalInput::space is NormalSpace::WorldSpace:
gtao.uniforms.camera_view_rotation = world_to_view_rotation;

let mut encoder = device.create_command_encoder(&Default::default());
// ... render your scene's depth (and optional normal) buffer first ...
gtao.render(&queue, &mut encoder);
// Sample gtao.ao_view() in your lighting/composite pass.
```

## Conventions

- Standard WebGPU depth range `[0, 1]`, not reversed.
- Column-major matrices.
- Input normals are in camera local (view) space by default. Set
  `NormalInput::space` to `NormalSpace::WorldSpace` for world-space normals;
  the shader then transforms them with `GtaoUniforms::camera_view_rotation`
  (column-major world-to-view rotation). For a rigid camera this is the
  upper-left 3x3 block of the view matrix; if the view matrix contains
  non-uniform scaling, pass the inverse-transpose normal matrix instead.
- The depth/normal textures use a top-left UV origin.
- `depth_input` must be a single-sample depth texture view. For combined
  depth-stencil formats, create the view with `TextureAspect::DepthOnly`.

## Example

`examples/gtao_demo.rs` renders an offscreen scene once per normal mode and
writes two files per mode into `target/gtao_modes/`:

| mode | normal input |
| --- | --- |
| `depth` | reconstructed from depth |
| `plain_view` | plain, view space |
| `plain_world` | plain, world space |
| `octa_view` | octahedral, view space |
| `octa_world` | octahedral, world space |

Each mode produces `<mode>_lit.bmp` (lit scene with AO applied) and
`<mode>_ao.bmp` (final AO). The view-space and world-space variants should look
nearly identical, which makes it easy to spot a broken normal transform.

After rendering, the example prints a table comparing every AO buffer against
`plain_view` (the view-space reference): standard deviation and mean absolute
difference of the per-pixel differences, maximum absolute difference, and the
percentage of pixels that differ. A correct world-space transform produces
statistics close to zero; depth-reconstructed normals differ mainly along
silhouettes and box edges.

```sh
cargo run --example gtao_demo
```

The example needs a working wgpu adapter (Vulkan/Metal/DX12/GL). It does not
need a window or display server.

## License

The GTAO and Poisson denoise shaders are ported from
[three.js](https://github.com/mrdoob/three.js), which is MIT licensed.
See [`LICENSE-THREEJS`](LICENSE-THREEJS) for the original copyright and
license notice.

## Limitations

- Perspective camera only.
- `SCREEN_SPACE_RADIUS` and `SCENE_CLIP_BOX` from three.js are not implemented.
- AO sample count is fixed at 16 (3 directions, 6 steps each).
- Denoise sample count is fixed at 16.
