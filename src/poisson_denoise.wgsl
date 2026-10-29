// Poisson denoise pass. Ported from three.js' PoissonDenoiseShader.

struct Uniforms {
    camera_projection: mat4x4f,
    inv_camera_projection: mat4x4f,
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
    source_width: f32,
    source_height: f32,
    denoise_luma_phi: f32,
    denoise_depth_phi: f32,
    denoise_normal_phi: f32,
    denoise_radius: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var t_ao: texture_2d<f32>;
@group(0) @binding(2) var t_depth: texture_depth_2d;
@group(0) @binding(3) var t_normal: texture_2d<f32>;
@group(0) @binding(4) var s_ao: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    let p = positions[vertex_index];
    var out: VertexOutput;
    out.position = vec4f(p, 0.0, 1.0);
    out.uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

const PI: f32 = 3.141592653589793;
const SAMPLES: i32 = 16;

fn clamp_coord(p: vec2i, size: vec2u) -> vec2i {
    return min(max(p, vec2i(0, 0)), vec2i(size) - vec2i(1, 1));
}

fn safe_normalize(v: vec3f) -> vec3f {
    let len = length(v);
    if (len > 0.0) {
        return v / len;
    }
    return vec3f(0.0);
}

fn fetch_depth(p: vec2i) -> f32 {
    return textureLoad(t_depth, clamp_coord(p, textureDimensions(t_depth)), 0);
}

fn get_depth(uv: vec2f) -> f32 {
    let size = textureDimensions(t_depth);
    let p = vec2i(uv * vec2f(size));
    return textureLoad(t_depth, clamp_coord(p, size), 0);
}

fn get_view_position(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let clip = vec4f(ndc, depth, 1.0);
    let view = u.inv_camera_projection * clip;
    return view.xyz / view.w;
}

fn compute_normal_from_depth(uv: vec2f) -> vec3f {
    let size = textureDimensions(t_depth);
    let size_f = vec2f(size);
    let p = clamp_coord(vec2i(uv * size_f), size);

    let c0 = fetch_depth(p);
    let l2 = fetch_depth(p - vec2i(2, 0));
    let l1 = fetch_depth(p - vec2i(1, 0));
    let r1 = fetch_depth(p + vec2i(1, 0));
    let r2 = fetch_depth(p + vec2i(2, 0));
    let b2 = fetch_depth(p + vec2i(0, 2));
    let b1 = fetch_depth(p + vec2i(0, 1));
    let t1 = fetch_depth(p - vec2i(0, 1));
    let t2 = fetch_depth(p - vec2i(0, 2));

    let dl = abs((2.0 * l1 - l2) - c0);
    let dr = abs((2.0 * r1 - r2) - c0);
    let db = abs((2.0 * b1 - b2) - c0);
    let dt = abs((2.0 * t1 - t2) - c0);

    let ce = get_view_position(uv, c0);
    let texel = vec2f(1.0 / size_f.x, 1.0 / size_f.y);

    var dpdx: vec3f;
    if (dl < dr) {
        dpdx = ce - get_view_position(uv - vec2f(texel.x, 0.0), l1);
    } else {
        dpdx = -ce + get_view_position(uv + vec2f(texel.x, 0.0), r1);
    }

    var dpdy: vec3f;
    if (db < dt) {
        dpdy = ce - get_view_position(uv + vec2f(0.0, texel.y), b1);
    } else {
        dpdy = -ce + get_view_position(uv - vec2f(0.0, texel.y), t1);
    }

    return safe_normalize(cross(dpdx, dpdy));
}

fn decode_octahedral(p_in: vec2f) -> vec3f {
    var n = vec3f(p_in, 1.0 - abs(p_in.x) - abs(p_in.y));
    if (n.z < 0.0) {
        n = vec3f(
            (1.0 - abs(n.yx)) * select(vec2f(-1.0), vec2f(1.0), n.xy >= vec2f(0.0)),
            n.z,
        );
    }
    return safe_normalize(n);
}

fn get_view_normal(uv: vec2f) -> vec3f {
    let mode = i32(u.normal_mode);
    if (mode == 1 || mode == 2) {
        let size = textureDimensions(t_normal);
        let p = clamp_coord(vec2i(uv * vec2f(size)), size);
        let texel = textureLoad(t_normal, p, 0);
        if (mode == 1) {
            return safe_normalize(texel.xyz);
        }
        return decode_octahedral(texel.xy);
    }
    return compute_normal_from_depth(uv);
}

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
