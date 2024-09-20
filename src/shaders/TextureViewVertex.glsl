#version 400 core

in vec2 vertexCoord;
in vec2 textureCoord;
uniform mat4 matrixProj;

out vec2 coord;

void main()
{
    gl_Position = matrixProj * vec4(vertexCoord, 0.0, 1.0);
    coord = textureCoord;
}
