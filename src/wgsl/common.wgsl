// Shared declarations and helpers for the GTAO and Poisson denoise passes.
// Prepended to both shaders by src/pipeline.rs.
// Ported from three.js; see LICENSE-THREEJS.

struct Uniforms {
    camera_projection: mat4x4f,
    inv_camera_projection: mat4x4f,
    // Upper-left 3x3 of the world-to-view matrix, column-major.
    camera_view_rotation: mat3x3f,
    radius: f32,
    distance_exponent: f32,
    thickness: f32,
    distance_falloff: f32,
    scale: f32,
    denoise_rings: f32,
    denoise_radius_exponent: f32,
    ao_width: f32,
    ao_height: f32,
    denoise_luma_phi: f32,
    denoise_depth_phi: f32,
    denoise_normal_phi: f32,
    denoise_radius: f32,
    // Pad the struct to a multiple of 16 bytes for uniform buffer layout.
    _padding_0: f32,
    _padding_1: f32,
    _padding_2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

const PI: f32 = 3.141592653589793;

// Specialized at pipeline creation: 0 = depth, 1 = plain, 2 = octahedral.
override NORMAL_MODE: i32 = 0;
// Specialized at pipeline creation: 0 = camera local (view) space, 1 = world space.
override NORMAL_SPACE: i32 = 0;

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
    // WebGPU texture space has its origin in the top-left corner.
    out.uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

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
    return textureLoad(t_depth, clamp_coord(p, textureDimensions(t_depth)), 0).r;
}

fn get_depth(uv: vec2f) -> f32 {
    let size = textureDimensions(t_depth);
    let p = vec2i(uv * vec2f(size));
    return textureLoad(t_depth, clamp_coord(p, size), 0).r;
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

    // Our texture V axis points down (WebGPU), so +V is down on screen.
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
    if (NORMAL_MODE == 0) {
        return compute_normal_from_depth(uv);
    }

    let size = textureDimensions(t_normal);
    let p = clamp_coord(vec2i(uv * vec2f(size)), size);
    let texel = textureLoad(t_normal, p, 0);
    var n: vec3f;
    if (NORMAL_MODE == 1) {
        n = safe_normalize(texel.xyz);
    } else {
        n = decode_octahedral(texel.xy);
    }

    if (NORMAL_SPACE == 1) {
        n = u.camera_view_rotation * n;
    }
    return safe_normalize(n);
}
