#version 400 core

uniform mat4 matrixColorR;
uniform mat4 matrixColorG;
uniform mat4 matrixColorB;
uniform float strokePeak;
uniform float strokeNorm;

in vec2 textureCoord;
out vec4 color;

void main()
{
    float x = textureCoord.x;
    int i = x == 4.0 ? 3 : int(x);
    x -= i;
    double x2 = x * x;
    double x3 = x2 * x;

    double ra = double(matrixColorR[i][0]);
    double rb = double(matrixColorR[i][1]);
    double rc = double(matrixColorR[i][2]);
    double rd = double(matrixColorR[i][3]);
    double ry = ra + rb * x + rc * x2 + rd * x3;
    double rk = rb + 2 * rc * x + 3 * rd * x2;
    double rh = abs(ry - textureCoord.y) * strokeNorm / sqrt(rk * rk + 1);
    double r = clamp((1.0 - 4 * rh * rh) * strokePeak, 0.0, 1.0);

    double ga = double(matrixColorG[i][0]);
    double gb = double(matrixColorG[i][1]);
    double gc = double(matrixColorG[i][2]);
    double gd = double(matrixColorG[i][3]);
    double gy = ga + gb * x + gc * x2 + gd * x3;
    double gk = gb + 2 * gc * x + 3 * gd * x2;
    double gh = abs(gy - textureCoord.y) * strokeNorm / sqrt(gk * gk + 1);
    double g = clamp((1.0 - 4 * gh * gh) * strokePeak, 0.0, 1.0);

    double ba = double(matrixColorB[i][0]);
    double bb = double(matrixColorB[i][1]);
    double bc = double(matrixColorB[i][2]);
    double bd = double(matrixColorB[i][3]);
    double by = ba + bb * x + bc * x2 + bd * x3;
    double bk = bb + 2 * bc * x + 3 * bd * x2;
    double bh = abs(by - textureCoord.y) * strokeNorm / sqrt(bk * bk + 1);
    double b = clamp((1.0 - 4 * bh * bh) * strokePeak, 0.0, 1.0);

    dvec3 rgb = dvec3(r, g, b);
    double l = length(rgb);
    if (l > 1.0) {
        rgb /= l;
    }

    color = vec4(rgb, 1.0);
}
