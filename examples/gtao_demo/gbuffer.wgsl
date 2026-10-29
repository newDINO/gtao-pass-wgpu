struct Camera {
    view_projection: mat4x4f,
    view: mat4x4f,
};

@group(0) @binding(0) var<uniform> camera: Camera;

// Specialized per mode by the example:
// 0 = plain view, 1 = plain world, 2 = octahedral view, 3 = octahedral world.
override NORMAL_OUTPUT: i32 = 0;

struct VsIn {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec3f,
};

struct VsOut {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) color: vec3f,
    @location(2) view_normal: vec3f,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_position = camera.view_projection * vec4f(in.position, 1.0);
    out.world_normal = in.normal;
    out.view_normal = normalize((camera.view * vec4f(in.normal, 0.0)).xyz);
    out.color = in.color;
    return out;
}

fn encode_octahedral(n_in: vec3f) -> vec2f {
    let n = n_in / (abs(n_in.x) + abs(n_in.y) + abs(n_in.z));
    var p = n.xy;

    if (n.z < 0.0) {
        p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
    }

    return p;
}

struct FsOut {
    @location(0) color: vec4f,
    @location(1) normal: vec4f,
};

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let world_normal = normalize(in.world_normal);
    let light_dir = normalize(vec3f(0.45, 0.8, 0.35));
    let diffuse = max(dot(world_normal, light_dir), 0.0);
    let light = 0.22 + 0.78 * diffuse;

    var n: vec3f;
    if (NORMAL_OUTPUT == 1 || NORMAL_OUTPUT == 3) {
        n = world_normal;
    } else {
        n = normalize(in.view_normal);
    }

    var out: FsOut;
    out.color = vec4f(in.color * light, 1.0);
    if (NORMAL_OUTPUT >= 2) {
        out.normal = vec4f(encode_octahedral(n), 0.0, 1.0);
    } else {
        out.normal = vec4f(n, 1.0);
    }
    return out;
}
