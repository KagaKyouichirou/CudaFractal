#version 400 core

uniform sampler2D points;

uniform mat4 matrixColorR;
uniform mat4 matrixColorG;
uniform mat4 matrixColorB;

in vec2 coord;
out vec4 color;

void main()
{
    double t = 4 * texture(points, coord).r;
    if (t == 4.0) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    int i = int(t);
    t -= i;
    double t2 = t * t;
    double t3 = t2 * t;
    double r = double(matrixColorR[i][0]) + double(matrixColorR[i][1]) * t + double(matrixColorR[i][2]) * t2 + double(matrixColorR[i][3]) * t3;
    double g = double(matrixColorG[i][0]) + double(matrixColorG[i][1]) * t + double(matrixColorG[i][2]) * t2 + double(matrixColorG[i][3]) * t3;
    double b = double(matrixColorB[i][0]) + double(matrixColorB[i][1]) * t + double(matrixColorB[i][2]) * t2 + double(matrixColorB[i][3]) * t3;
    color = vec4(r, g, b, 1.0);
}
