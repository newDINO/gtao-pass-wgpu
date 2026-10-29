struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var t_ao: texture_2d<f32>;
@group(0) @binding(2) var s_color: sampler;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let color = textureSampleLevel(t_color, s_color, in.uv, 0.0).rgb;
    let ao = textureSampleLevel(t_ao, s_color, in.uv, 0.0).r;
    return vec4f(color * ao, 1.0);
}

@fragment
fn fs_ao(in: VertexOutput) -> @location(0) vec4f {
    let ao = textureSampleLevel(t_ao, s_color, in.uv, 0.0).r;
    return vec4f(ao, ao, ao, 1.0);
}
