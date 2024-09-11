#version 330 core

uniform sampler2D points;
uniform double limit;

in vec2 coord;
out vec4 color;

void main() {
    double k = texture(points, coord);
    double t = k / limit;       // Normalize to [0, 1]

    color = vec4(t, t * 0.5, 1.0 - t, 1.0);
}
