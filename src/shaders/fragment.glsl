#version 400 core

uniform sampler2DRect points;

in vec2 coord;
out vec4 color;

void main() {
    float t = texture(points, coord).r;
    float u = log(1 + 400 * t) / log(1 + 400.0);

    color = u == 1.0 ? vec4(0, 0, 0, 1.0) : vec4(u, u * u * u, 0.5 - u * u * u * u, 1.0);
    //color = vec4(0.5, 0.25, 0.0, 1.0);
}
