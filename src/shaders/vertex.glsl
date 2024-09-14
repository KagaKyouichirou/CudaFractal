#version 400 core

attribute vec2 vertex;
uniform mat4 projMatrix;

out vec2 coord;

void main() {
    gl_Position = projMatrix * vec4(vertex, 0.0, 1.0);
    coord = vertex;
}
