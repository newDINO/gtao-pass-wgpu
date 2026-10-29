//! Ground truth ambient occlusion (GTAO) for [`wgpu`].
//!
//! This is a fairly direct WGSL port of the GTAO implementation in three.js
//! (`GTAOShader` + `PoissonDenoiseShader`). three.js is MIT licensed; see the
//! `LICENSE-THREEJS` file for the original copyright and license notice. The
//! renderer expects a depth texture and, optionally, a normal texture. Both are
//! sampled at the current output resolution, so the AO buffer can be rendered
//! at a lower resolution than the scene. Supplied normals may be in camera
//! local (view) space or world space; see [`NormalSpace`].
//!
//! # Coordinate conventions
//!
//! * Depth is the standard WebGPU `[0, 1]` depth range (not reversed).
//! * Matrices are column-major, like three.js / `glam`.
//! * Normals passed through [`NormalInput`] are in camera local (view) space
//!   or world space, as selected by [`NormalInput::space`].
//! * World-space normals are transformed with
//!   [`GtaoUniforms::camera_view_rotation`], the column-major world-to-view
//!   rotation. For a rigid camera this is the upper-left 3x3 block of the view
//!   matrix; a camera with non-uniform scaling needs the inverse-transpose
//!   normal matrix.
//! * Depth and normal textures are sampled with their top-left corner at UV
//!   `(0, 0)`.
//!
//! # Example
//!
//! ```no_run
//! # use gtao_pass_wgpu::*;
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

mod normal;
mod params;
mod pipeline;
mod renderer;
mod uniforms;
mod util;

pub use normal::{NormalEncoding, NormalInput, NormalSpace};
pub use params::{DenoiseParams, GtaoParams};
pub use renderer::{GtaoDescriptor, GtaoRenderer};
pub use uniforms::GtaoUniforms;
pub use util::scale_resolution;
