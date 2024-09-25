#version 400 core

in vec2 vertexCoord;
out vec2 coord;

void main()
{
    gl_Position = vec4(vertexCoord, 0.0, 1.0);
    // [0, 1] * [0, 1]
    coord = (vertexCoord + 1.0) / 2;
}
