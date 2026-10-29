struct Camera {
    view_projection: mat4x4f,
    view: mat4x4f,
};

@group(0) @binding(0) var<uniform> camera: Camera;

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

struct FsOut {
    @location(0) color: vec4f,
    @location(1) normal: vec4f,
};

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let normal = normalize(in.world_normal);
    let light_dir = normalize(vec3f(0.45, 0.8, 0.35));
    let diffuse = max(dot(normal, light_dir), 0.0);
    let light = 0.22 + 0.78 * diffuse;

    var out: FsOut;
    out.color = vec4f(in.color * light, 1.0);
    out.normal = vec4f(normalize(in.view_normal), 1.0);
    return out;
}
