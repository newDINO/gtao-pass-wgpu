// Poisson denoise pass. Ported from three.js' PoissonDenoiseShader.
// Original: examples/jsm/shaders/PoissonDenoiseShader.js
// three.js is MIT licensed; see LICENSE-THREEJS.
//
// Shared declarations live in wgsl/common.wgsl and are prepended by
// src/pipeline.rs.

@group(0) @binding(1) var t_ao: texture_2d<f32>;
// Depth is bound as an unfilterable float texture rather than
// texture_depth_2d so that textureLoad works on the GL backend.
@group(0) @binding(2) var t_depth: texture_2d<f32>;
@group(0) @binding(3) var t_normal: texture_2d<f32>;
@group(0) @binding(4) var s_ao: sampler;

const SAMPLES: i32 = 16;

fn hash_u32(value_in: u32) -> u32 {
    var value = value_in;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

fn rand01(pixel: vec2u) -> f32 {
    let h = hash_u32(pixel.x * 1973u + pixel.y * 9277u + 26699u);
    return f32(h) / 4294967295.0;
}

fn denoise_sample(
    center: f32,
    view_normal: vec3f,
    view_pos: vec3f,
    sample_uv: vec2f,
) -> vec2f {
    let sample_texel = textureSampleLevel(t_ao, s_ao, sample_uv, 0.0);
    let sample_depth = get_depth(sample_uv);
    let sample_normal = get_view_normal(sample_uv);
    let neighbor = sample_texel.r;
    let sample_view_pos = get_view_position(sample_uv, sample_depth);

    let normal_diff = dot(view_normal, sample_normal);
    let normal_similarity = pow(max(normal_diff, 0.0), u.denoise_normal_phi);
    let luma_diff = abs(neighbor - center);
    let luma_similarity = max(1.0 - luma_diff / u.denoise_luma_phi, 0.0);
    let depth_diff = abs(dot(view_pos - sample_view_pos, view_normal));
    let depth_similarity = max(1.0 - depth_diff / u.denoise_depth_phi, 0.0);
    let w = luma_similarity * depth_similarity * normal_similarity;

    return vec2f(neighbor * w, w);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let uv = in.uv;
    let depth = get_depth(uv);
    let view_normal = get_view_normal(uv);
    if (depth >= 1.0 || dot(view_normal, view_normal) == 0.0) {
        discard;
    }

    let texel = textureSampleLevel(t_ao, s_ao, uv, 0.0);
    let center = texel.r;
    let view_pos = get_view_position(uv, depth);

    let ao_size = vec2f(u.ao_width, u.ao_height);
    let pixel = vec2u(uv * ao_size);
    let rotation_angle = rand01(pixel) * 2.0 * PI;
    let noise_vec = vec2f(sin(rotation_angle), cos(rotation_angle));
    let rotation_matrix = mat2x2f(noise_vec.x, -noise_vec.y, noise_vec.x, noise_vec.y);

    let radius = u.denoise_radius;
    var total_weight = 1.0;
    var denoised = center;

    for (var i: i32 = 0; i < SAMPLES; i++) {
        let angle = 2.0 * PI * u.denoise_rings * f32(i) / f32(SAMPLES);
        let sample_radius = pow(f32(i) / f32(SAMPLES - 1), u.denoise_radius_exponent);
        let sample_dir = vec3f(cos(angle), sin(angle), sample_radius);
        let offset = rotation_matrix
            * (sample_dir.xy * (1.0 + sample_dir.z * (radius - 1.0)) / ao_size);
        let sample_uv = uv + offset;
        let contribution = denoise_sample(center, view_normal, view_pos, sample_uv);
        denoised += contribution.x;
        total_weight += contribution.y;
    }

    if (total_weight > 0.0) {
        denoised /= total_weight;
    }
    return denoised;
}
